"""Offline drop/buildup/warning timeline for ear-checking.

Runs the detection seq model over each track, applies the debounced drop and
buildup state machines + the pre-drop (drop_incoming) warning, and writes a
per-track event timeline (times in seconds -> maps to <id>.wav) next to the
ground-truth cue points so you can scrub the audio and verify by ear.

Usage:
  uv run python _timeline.py            # val tracks
  uv run python _timeline.py 0018.xxx   # specific track id(s)
"""

import json
import os
import sys

import torch
from safetensors.torch import load_file

from _arch import SeqPhrasePredictor, build_seqs
from _drop import debounce_state, state_edges
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits
from audiovj.data.rekordbox import Track

DROP = PHRASE_TYPES.index("drop")
BUILDUP = PHRASE_TYPES.index("buildup")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "/mnt/scratch/data/loop/seq_predictor.safetensors"
OUT = "/mnt/scratch/data/timelines"


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def timeline_for(tid, model):
    seqs = build_seqs([tid])
    if not seqs:
        return None
    s = seqs[0]
    times = s["times"]
    with torch.no_grad():
        o = model(s["windows"].to(DEV).unsqueeze(0))
    pcur = o.current_phrase_logits[0].argmax(-1).cpu().tolist()
    nconf, nidx = torch.softmax(o.next_phrase_logits[0], -1).max(-1)
    pnext = nidx.cpu().tolist(); pconf = nconf.cpu().tolist()

    # minimum-span filter: real drops/buildups span several bars, so drop runs
    # shorter than min_len downbeats are flicker, not events.
    def filter_min(state, min_len):
        st = list(state); k = 0; n = len(st)
        while k < n:
            if st[k]:
                j = k
                while j < n and st[j]:
                    j += 1
                if j - k < min_len:
                    for x in range(k, j):
                        st[x] = False
                k = j
            else:
                k += 1
        return st

    drop_state = filter_min(debounce_state(pcur, DROP), 8)   # >= ~8 bars
    build_state = filter_min(debounce_state(pcur, BUILDUP), 2)

    # single chronological pass: idle -> warned -> drop -> idle, plus buildup spans
    events = []
    bs, be = state_edges(build_state)
    for k in bs:
        events.append((times[k], "buildup_start"))
    for k in be:
        events.append((times[k], "buildup_end"))

    phase = "idle"; consec = 0; cold = 0
    for k in range(len(times)):
        hot = pnext[k] == DROP and pconf[k] > 0.5
        consec = consec + 1 if hot else 0
        cold = 0 if hot else cold + 1
        if drop_state[k] and phase != "drop":
            events.append((times[k], "drop_start")); phase = "drop"
        elif not drop_state[k] and phase == "drop":
            events.append((times[k], "drop_end")); phase = "idle"; cold = 0
        elif phase == "idle" and consec >= 2:
            events.append((times[k], "drop_incoming")); phase = "warned"
        elif phase == "warned" and cold >= 8:
            phase = "idle"  # warning fizzled for 2 bars; allow a fresh one later

    events.sort(key=lambda e: e[0])
    track = Track.model_validate_json((TRACKS_DIR / f"{tid}.json").read_text())
    truth = [(c.start_time, c.phrase_type) for c in track.cue_points]
    return {
        "track_id": tid,
        "name": f"{track.artist} - {track.name}",
        "bpm": track.bpm,
        "audio": f"data/audio/{tid}.wav",
        "events": [{"t": round(t, 2), "mmss": mmss(t), "kind": k} for t, k in events],
        "ground_truth_cues": [{"t": round(t, 2), "mmss": mmss(t), "phrase": p} for t, p in truth],
    }


def main():
    ids = sys.argv[1:]
    if not ids:
        _, ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    m = SeqPhrasePredictor().to(DEV); m.load_state_dict(load_file(CKPT)); m.eval()
    os.makedirs(OUT, exist_ok=True)
    done = 0
    for tid in ids:
        tl = timeline_for(tid, m)
        if tl is None:
            continue
        with open(f"{OUT}/{tid}.json", "w") as f:
            json.dump(tl, f, indent=2)
        done += 1
        if done <= 3:  # readable preview for the first few
            print(f"\n=== {tl['name']}  ({tl['bpm']:.0f} BPM)  [{tid}.wav] ===")
            print("  PREDICTED                         |  GROUND TRUTH (drop cues)")
            gt_drops = [c for c in tl["ground_truth_cues"] if c["phrase"] == "drop"]
            ev = [e for e in tl["events"] if e["kind"] in ("drop_incoming", "drop_start", "drop_end", "buildup_start")]
            rows = max(len(ev), len(gt_drops))
            for i in range(rows):
                left = f"{ev[i]['mmss']:>7}  {ev[i]['kind']}" if i < len(ev) else ""
                right = f"drop @ {gt_drops[i]['mmss']}" if i < len(gt_drops) else ""
                print(f"  {left:<33} |  {right}")
    print(f"\nwrote {done} timelines to {OUT}/  (times in seconds; <id>.wav under data/audio)")


if __name__ == "__main__":
    main()
