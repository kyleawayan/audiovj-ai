"""Replay a recorded session offline and test the LSTM-drift hypothesis.

The live run carried ONE hidden state across 379 consecutive downbeats, while
training sequences were per-track (~150-200 downbeats, always from h=0). The
session showed p(end+outro) climbing 0.002 -> 0.78 and every drop after the
halfway point going undetected, which is what state drift would look like.

Step 1 verifies the replay reproduces the live run from audio.wav alone (if it
does not, nothing downstream is trustworthy). Step 2 re-runs with periodic
resets and scores each against the human note-61 labels.

Usage: uv run python experiments/_replay.py <session-dir>
"""
import json, statistics, struct, sys
from pathlib import Path

import numpy as np
import torch

from audiovj.config import CONTEXT_BEATS, MODELS_DIR
from audiovj.data.features import extract_mel_spectrogram
from audiovj.live.cue import OnsetCueTracker
from audiovj.live.inference import SeqInferenceEngine

SR = 44100
DROP = 3


def read_wav(path: Path):
    raw = path.read_bytes()
    pos, sr, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            sr = struct.unpack("<HHI", raw[pos + 8:pos + 16])[2]
        elif cid == b"data":
            data = raw[pos + 8:pos + 8 + size] if size else raw[pos + 8:]
        pos += 8 + size + (size & 1)
    n = len(data) // 4
    return np.frombuffer(data[:n * 4], dtype="<f4"), sr


def run(audio, db, engine, reset_every=0, onset=0.30, shift=0):
    """Replay every downbeat; optionally reset state every N downbeats.

    ``shift`` (samples) moves the window END later in the recording. The capture
    path buffers ~465 ms, so the musical downbeat appears in the file that much
    AFTER the logged audio_pos; shifting by the measured input latency therefore
    simulates a zero-latency capture reading at the true downbeat.
    """
    engine.reset()
    cue = OnsetCueTracker(onset_threshold=onset, drop_confirm=1, drop_release=2)
    probs, fires = [], []
    since = 0
    for i, r in enumerate(db):
        if reset_every and since >= reset_every:
            engine.reset(); cue.reset(); since = 0
        since += 1
        need = int(CONTEXT_BEATS * (60.0 / r["bpm"]) * SR)
        end = min(len(audio), r["audio_pos"] + shift)
        win = audio[max(0, end - need):end]
        if len(win) < need:
            win = np.concatenate([np.zeros(need - len(win), dtype=np.float32), win])
        win = win * r["agc_gain"]           # reproduce the gain that was applied live
        mel = extract_mel_spectrogram(torch.from_numpy(win).unsqueeze(0))[0]
        pred = engine.step_window(mel)
        probs.append(pred.current_probs)
        for e in cue.update(pred):
            if e.kind == "drop_start":
                fires.append(i)
    return probs, fires


def score(probs, fires, db, labels_idx):
    caught = []
    for y in labels_idx:
        near = [f for f in fires if abs(f - y) <= 2]
        if near:
            b = min(near, key=lambda f: abs(f - y))
            caught.append((y, (b - y) * 4))
    at = {}
    for off in (0, 1, 2):
        v = [probs[y + off][DROP] for y in labels_idx if 0 <= y + off < len(probs)]
        at[off] = statistics.median(v) if v else 0.0
    n = len(probs)
    half = n // 2
    late_end = statistics.median(
        probs[i][7] + probs[i][8] + probs[i][9] for i in range(half, n))
    return {
        "caught": len(caught), "of": len(labels_idx),
        "lat": statistics.median([d for _, d in caught]) if caught else None,
        "first_half": sum(1 for y, _ in caught if y < half),
        "second_half": sum(1 for y, _ in caught if y >= half),
        "pD0": at[0], "pD1": at[1], "pD2": at[2],
        "fires": len(fires), "p_end_2nd_half": late_end,
    }



def load_labels(sess, db, pos):
    """Prefer hand-corrected labels when the review tool has produced them.

    Corrected labels carry times, not downbeat indices, so map each back to its
    nearest downbeat. The tool snaps to the grid, so this is exact in practice;
    a mismatch means the file was edited by something else.
    """
    import json as _j
    corr = sess / "labels_corrected.json"
    if corr.exists():
        d = _j.loads(corr.read_text())
        out = []
        for l in d["labels"]:
            if l["kind"] != "drop_start":
                continue
            s = l["t_sec"] * 44100
            out.append(min(range(len(db)), key=lambda i: abs(db[i]["audio_pos"] - s)))
        print(f"  using {corr.name} ({len(out)} drop_start labels)")
        return out
    return None

def main():
    sess = Path(sys.argv[1]).expanduser()
    recs = [json.loads(l) for l in (sess / "events.jsonl").read_text().splitlines() if l.strip()]
    db = [r for r in recs if r["kind"] == "downbeat"]
    pos = {r["audio_pos"]: i for i, r in enumerate(db)}
    labels_idx = load_labels(sess, db, pos) or [
        pos[l["audio_pos"]] for l in recs
        if l.get("label") == "drop_start" and not l.get("press_suspect")
        and l["audio_pos"] in pos]
    audio, sr = read_wav(sess / "audio.wav")
    print(f"{len(db)} downbeats, {len(labels_idx)} usable drop labels, "
          f"{len(audio)/sr:.0f}s audio")

    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    engine = SeqInferenceEngine(MODELS_DIR / "seq_unified.safetensors", dev)

    # ---- step 1: fidelity. replay must reproduce the live probabilities -----
    probs, fires = run(audio, db, engine, reset_every=0)
    live = [r["probs"] for r in db]
    err = [max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(probs, live)]
    agree = sum(1 for p, q in zip(probs, live)
                if max(range(10), key=lambda k: p[k]) == max(range(10), key=lambda k: q[k]))
    print(f"\nREPLAY FIDELITY vs the live run:")
    print(f"  argmax agreement {agree}/{len(db)} = {agree/len(db)*100:.1f}%")
    print(f"  max |prob delta|: median {statistics.median(err):.4f}  max {max(err):.4f}")
    if agree / len(db) < 0.9:
        print("  WARNING: replay diverges from live — treat results below as indicative only")

    # ---- step 2: does resetting the LSTM recover the second half? -----------
    print(f"\n{'reset every':>12} | {'caught':>7} | {'1st/2nd half':>12} | {'lat':>5} | "
          f"{'p(drop) D0/D1/D2':>22} | {'fires':>5} | {'p(end) 2nd half':>15}")
    print("-" * 104)
    lat_s = json.loads((sess / "manifest.json").read_text()).get("input_latency_s", 0.0)
    shift = int(lat_s * SR)
    for every in (0, 128, 64, 32, 16, 8):
        p2, f2 = run(audio, db, engine, reset_every=every)
        s = score(p2, f2, db, labels_idx)
        lab = "never (live)" if every == 0 else f"{every} downbeats"
        lat = f"{s['lat']:+.0f}b" if s["lat"] is not None else "  -"
        print(f"{lab:>12} | {s['caught']:>3}/{s['of']:<3} | "
              f"{s['first_half']:>5}/{s['second_half']:<6} | {lat:>5} | "
              f"{s['pD0']:>6.3f} {s['pD1']:>6.3f} {s['pD2']:>6.3f}   | "
              f"{s['fires']:>5} | {s['p_end_2nd_half']:>15.3f}")

    # ---- step 3: both fixes together ---------------------------------------
    print(f"\n=== + LOW-LATENCY CAPTURE (windows shifted +{lat_s*1000:.0f} ms "
          f"= {shift} samples) ===")
    print(f"{'reset every':>12} | {'caught':>7} | {'1st/2nd half':>12} | {'lat':>5} | "
          f"{'p(drop) D0/D1/D2':>22} | {'fires':>5} | {'p(end) 2nd half':>15}")
    print("-" * 104)
    for every in (0, 128, 64):
        p2, f2 = run(audio, db, engine, reset_every=every, shift=shift)
        s = score(p2, f2, db, labels_idx)
        lab = "never" if every == 0 else f"{every} downbeats"
        lat = f"{s['lat']:+.0f}b" if s["lat"] is not None else "  -"
        print(f"{lab:>12} | {s['caught']:>3}/{s['of']:<3} | "
              f"{s['first_half']:>5}/{s['second_half']:<6} | {lat:>5} | "
              f"{s['pD0']:>6.3f} {s['pD1']:>6.3f} {s['pD2']:>6.3f}   | "
              f"{s['fires']:>5} | {s['p_end_2nd_half']:>15.3f}")


if __name__ == "__main__":
    main()
