"""Measure f* for real: p(drop) vs where the analysis window ENDS.

Previously estimated by interpolating three downbeat-aligned points and assuming
linearity. With the recorded audio the curve can be measured directly: hold the
committed LSTM state fixed at the drop's downbeat and peek with windows ending
at sub-beat offsets after it.

Also answers the input-latency question. The capture path buffers ~465 ms, so
the musical downbeat appears in the recording ~465 ms AFTER the logged
audio_pos. A window ending at offset +465 ms is therefore what a zero-latency
capture would have seen at the downbeat -- it is one point on this same curve.

Usage: uv run python experiments/_fstar.py <session-dir>
"""
import json, statistics, struct, sys
from pathlib import Path

import numpy as np
import torch

from audiovj.config import CONTEXT_BEATS, MODELS_DIR
from audiovj.data.features import extract_mel_spectrogram
from audiovj.live.inference import SeqInferenceEngine

SR, DROP, THR = 44100, 3, 0.30
OFFSETS = [i * 0.25 for i in range(0, 33)]     # 0 .. 8 beats, quarter-beat steps
RESET_EVERY = 64


def read_wav(path):
    raw = path.read_bytes(); pos = 12; data = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos+4]; size = struct.unpack("<I", raw[pos+4:pos+8])[0]
        if cid == b"data": data = raw[pos+8:pos+8+size] if size else raw[pos+8:]
        pos += 8 + size + (size & 1)
    n = len(data)//4
    return np.frombuffer(data[:n*4], dtype="<f4")


def window(audio, end, need):
    w = audio[max(0, end-need):end]
    if len(w) < need:
        w = np.concatenate([np.zeros(need-len(w), dtype=np.float32), w])
    return w



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
    recs = [json.loads(l) for l in (sess/"events.jsonl").read_text().splitlines() if l.strip()]
    man = json.loads((sess/"manifest.json").read_text())
    db = [r for r in recs if r["kind"] == "downbeat"]
    pos = {r["audio_pos"]: i for i, r in enumerate(db)}
    labels = load_labels(sess, db, pos) or [
        pos[l["audio_pos"]] for l in recs
        if l.get("label") == "drop_start" and not l.get("press_suspect")
        and l["audio_pos"] in pos]
    audio = read_wav(sess/"audio.wav")
    lat_s = man.get("input_latency_s", 0.0)
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    eng = SeqInferenceEngine(MODELS_DIR/"seq_unified.safetensors", dev)

    curves = {o: [] for o in OFFSETS}
    eng.reset(); since = 0
    for i, r in enumerate(db):
        if since >= RESET_EVERY:
            eng.reset(); since = 0
        since += 1
        beat = 60.0/r["bpm"]
        need = int(CONTEXT_BEATS*beat*SR)
        if i in labels:
            # peek does not advance state, so the sweep cannot disturb the run
            for o in OFFSETS:
                end = r["audio_pos"] + int(o*beat*SR)
                if end > len(audio): continue
                w = window(audio, end, need)*r["agc_gain"]
                mel = extract_mel_spectrogram(torch.from_numpy(w).unsqueeze(0))[0]
                curves[o].append(eng.peek_window(mel).current_probs[DROP])
        w = window(audio, r["audio_pos"], need)*r["agc_gain"]
        eng.step_window(extract_mel_spectrogram(torch.from_numpy(w).unsqueeze(0))[0])

    beat_s = 60.0/statistics.median(r["bpm"] for r in db)
    lat_beats = lat_s/beat_s
    print(f"{len(labels)} drops.  input latency {lat_s*1000:.0f} ms = {lat_beats:.2f} beats\n")
    print(f"  {'window ends':>12} {'drop in win':>12} {'median':>8} {'mean':>8} "
          f"{'>=0.30':>7}")
    prev = None
    cross = None
    for o in OFFSETS:
        v = curves[o]
        if not v: continue
        med = statistics.median(v)
        frac = min(o/CONTEXT_BEATS, 1.0)
        n30 = sum(1 for x in v if x >= THR)
        mark = ""
        if abs(o - lat_beats) < 0.13: mark = "  <- zero-latency capture"
        if prev is not None and prev < THR <= med and cross is None:
            cross = o; mark += "  <<< CROSSES 0.30"
        print(f"  D0 +{o:>4.2f}b {frac*100:>10.0f}% {med:>8.3f} "
              f"{statistics.mean(v):>8.3f} {n30:>4}/{len(v)}{mark}")
        prev = med
    print()
    if cross is not None:
        print(f"f* = the window must be {cross/CONTEXT_BEATS*100:.0f}% drop audio.")
        print(f"Earliest possible fire: +{cross:.2f} beats  "
              f"(today: +8.00) -> saves {8.0-cross:.2f} beats")
    else:
        print("median p(drop) never crosses 0.30 within 8 beats of the drop")


if __name__ == "__main__":
    main()
