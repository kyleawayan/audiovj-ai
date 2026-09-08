"""Verify the session recorder end-to-end BEFORE trusting it with a real set.

Drives the REAL AudioCapture callback, the REAL model and the REAL recorder with
audio from data/smoke; only Carabiner / OSC / MIDI are stubbed. Checks the thing
that actually matters: that audio.wav is bit-exact against what the model was
fed, and that every event maps back to an exact audio sample.

Run from repo root:  uv run python experiments/_rectest.py
"""
import glob, json, struct, sys, threading, time, types
from pathlib import Path

import numpy as np

# stub only the hardware we cannot have here
mido = types.ModuleType("mido"); mido.get_input_names = lambda: []
mido.open_input = lambda *a, **k: None
sys.modules.setdefault("mido", mido)

import torchaudio
from audiovj.live import pipeline as PL
from audiovj.live.carabiner import DownbeatEvent
from audiovj.live.recorder import SessionRecorder

SR = 44100
OUT = Path("/tmp/audiovj-rectest")
BPM = 128.0
N_DOWNBEATS = 12

src = sorted(glob.glob("data/smoke/*.m4a"))[0]
wav, sr = torchaudio.load(src)
if sr != SR:
    wav = torchaudio.transforms.Resample(sr, SR)(wav)
mono = wav.mean(dim=0).numpy().astype(np.float32)
need = int(N_DOWNBEATS * 4 * 60.0 / BPM * SR) + SR
mono = mono[:need]
print(f"source: {Path(src).name}  {len(mono)} samples ({len(mono)/SR:.1f}s)")

class FakeCarabiner:
    def __init__(self, *a, **k):
        self._cb = k.get("on_downbeat"); self.bpm = BPM; self.beat_phase = 2.0
    alive = True; peers = 2
    def start(self): pass
    def stop(self): pass

sent = []
class FakeOSC:
    def __init__(self, *a, **k): pass
    def send_event(self, e): sent.append((e.kind, e.phrase))
    def send_beat(self, b): pass
    def send_status(self, s): pass

PL.CarabinerClient = FakeCarabiner
PL.OSCEmitter = FakeOSC
PL._setup_scroll_region = lambda: None
PL._teardown_scroll_region = lambda: None

p = PL.LivePipeline(
    checkpoint_path=Path("data/models/seq_unified.safetensors"),
    midi_port="", record_dir=SessionRecorder.new_dir(OUT), record_audio=True,
    drop_confirm=1, drop_release=2, force_drop_beats=8,
)
# real AudioCapture object, but no PortAudio stream: drive _callback directly
audio = p._audio
audio.start = lambda: None
audio.stop = lambda: None
type(audio).stream_latency = property(lambda self: 0.0117)

BLOCK = 2048
stop = threading.Event()
marks_injected = []
fed_samples = []          # what the callback ACTUALLY received (loop exits early)
armed_at = []             # downbeat index at which each arm press happened

def feed():
    """Mimic PortAudio: hand the real callback interleaved stereo blocks.

    The arm press is injected MID-BAR, which is what actually happens: the pad
    is pressed ~2 beats before the drop, so the press falls between two
    downbeats and the FOLLOWING downbeat is the labelled drop.
    """
    beat = 60.0 / BPM
    samples_per_bar = int(4 * beat * SR)
    pos = 0; downbeats = 0; armed_bars = set()
    while pos + BLOCK <= len(mono) and downbeats < N_DOWNBEATS:
        block = mono[pos:pos + BLOCK]
        indata = np.stack([block, block], axis=1)   # 2ch, mean() == block
        audio._callback(indata, BLOCK, None, None)
        pos += BLOCK
        fed_samples.append(BLOCK)

        # mid-bar arm press -> should snap to the NEXT downbeat
        if (downbeats in (2, 6) and downbeats not in armed_bars
                and pos % samples_per_bar > samples_per_bar // 2):
            armed_bars.add(downbeats)
            marks_injected.append(audio.write_pos)
            p._arm_drop()
            armed_at.append(downbeats + 1)        # expected labelled downbeat
            time.sleep(0.03)

        if pos // samples_per_bar > downbeats:
            downbeats += 1
            p._downbeat_queue.put(DownbeatEvent(
                time=time.monotonic(), bpm=BPM,
                beat_number=downbeats * 4.0, irregular=(downbeats == 5)))
            # Block until the pipeline has consumed it. Without this the feeder
            # (which pumps far faster than real time) leaves downbeats queued and
            # a press lands on a stale one.
            for _ in range(400):
                if p._downbeat_count >= downbeats: break
                time.sleep(0.005)
    time.sleep(0.5); stop.set()

threading.Thread(target=feed, daemon=True).start()
orig = p._downbeat_queue.get
def get(timeout=None):
    if stop.is_set(): raise KeyboardInterrupt
    return orig(timeout=timeout)
p._downbeat_queue.get = get
p.run()

# ---------------- verification ----------------
sess = sorted(d for d in OUT.glob("*") if d.is_dir())[-1]
print(f"\nsession dir: {sess}")
man = json.loads((sess / "manifest.json").read_text())
recs = [json.loads(l) for l in (sess / "events.jsonl").read_text().splitlines()]
rw, rsr = torchaudio.load(str(sess / "audio.wav"))
rec = rw[0].numpy()

fails = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok: fails.append(name)

print("\n--- verification ---")
check("wav sample rate", rsr == SR, f"{rsr}")
fed = sum(fed_samples)
n = min(len(rec), fed)
check("wav frames == samples fed to the callback", len(rec) == fed,
      f"rec={len(rec)} fed={fed}")
check("audio BIT-EXACT vs source", np.array_equal(rec[:n], mono[:n]),
      f"max|diff|={np.abs(rec[:n]-mono[:n]).max():.2e}" if n else "")
check("no dropped chunks", man.get("dropped_audio_chunks", -1) == 0,
      str(man.get("dropped_audio_chunks")))
db = [r for r in recs if r["kind"] == "downbeat"]
mk = [r for r in recs if r["kind"] == "mark"]  # separate non-forcing pad
check("downbeats logged", len(db) == N_DOWNBEATS, f"{len(db)}/{N_DOWNBEATS}")
check("downbeats logged after arm", len(db) == N_DOWNBEATS, f"{len(db)}")
check("irregular flag captured", any(r["irregular"] for r in db))
check("full prob vector per downbeat", all(len(r["probs"]) == 10 for r in db))
check("manifest has checkpoint digest", bool(man.get("checkpoint_sha256_16")))
check("manifest has audio_start_pos", "audio_start_pos" in man)
check("manifest has input latency", man.get("input_latency_s") == 0.0117)

# the alignment claim: audio_pos - audio_start_pos == sample offset in audio.wav
start = man["audio_start_pos"]
offs = [r["audio_pos"] - start for r in db]
check("event offsets monotonic", all(b > a for a, b in zip(offs, offs[1:])))
check("event offsets inside the wav", 0 <= offs[-1] <= len(rec) + BLOCK,
      f"last={offs[-1]} frames={len(rec)}")
lbl = [r for r in recs if r["kind"] == "drop_label"]
check("drop_label emitted per arm press", len(lbl) == len(armed_at),
      f"{len(lbl)}/{len(armed_at)}")
if lbl:
    check("label snapped to the NEXT downbeat",
          all(l["downbeat_index"] == a for l, a in zip(lbl, armed_at)),
          f"{[l['downbeat_index'] for l in lbl]} vs presses at {armed_at}")
    check("pad emits drop_start labels",
          all(l["label"] == "drop_start" for l in lbl),
          str([l["label"] for l in lbl]))
    check("forced hold starts at the label",
          db[lbl[0]["downbeat_index"] - 1]["forced"])
    i0 = lbl[0]["downbeat_index"] - 1        # 0-based index of the labelled downbeat
    check("forced hold lasts exactly 2 downbeats then releases",
          db[i0]["forced"] and db[i0 + 1]["forced"] and not db[i0 + 2]["forced"],
          f"forced={[db[i0 + k]['forced'] for k in range(3)]}")
    check("label carries UNFORCED model belief",
          all("p_drop_at_label" in l and len(l["probs_at_label"]) == 10 for l in lbl))
    check("label records press diagnostics",
          all("press_bar_phase" in l and "press_beats_early" in l for l in lbl),
          f"beats_early={lbl[0]['press_beats_early']:.1f}")
    check("press not flagged suspect at phase 2.0",
          all(not l["press_suspect"] for l in lbl))
    check("press audio_pos matches injection sample",
          [l["press_audio_pos"] for l in lbl] == marks_injected,
          f"{[l['press_audio_pos'] for l in lbl]} vs {marks_injected}")
    check("label downbeat is AFTER the press in samples",
          all(l["audio_pos"] > l["press_audio_pos"] for l in lbl))

if mk:
    moff = mk[0]["audio_pos"] - start
    exp = marks_injected[0] - start
    check("mark maps to exact injected sample", moff == exp, f"{moff} vs {exp}")
    check("mark carries model context", "p_drop_at_last_downbeat" in mk[0])

sec = len(rec) / SR
print(f"\naudio: {sec:.1f}s  {(sess/'audio.wav').stat().st_size/1e6:.1f} MB"
      f"  -> {(sess/'audio.wav').stat().st_size/1e6/max(sec,1)*3600:.0f} MB/hour")
print("RESULT:", "ALL PASSED" if not fails else f"{len(fails)} FAILED: {fails}")
