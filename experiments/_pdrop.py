"""What is p(drop) at the drop's own downbeat?

The lights flip on a 10-class argmax (pipeline.py), the strictest possible rule.
The model was TRAINED to emit "drop" at the boundary downbeat from buildup
evidence alone (dataset.py generate_labels uses cue_time <= t, while
features.py slice_beat_windows ends the window AT t). So p(drop) can be high
but second-place, and the argmax throws that anticipation away.

This measures, on held-out fold 7:
  1. the p(drop) distribution at D-1 / D0 / D+1 / D+2 around every labeled drop
  2. what the argmax actually says at D0
  3. a threshold sweep: on-time recall at D0 vs the false-fire rate it costs

Run from the repo root:  uv run python experiments/_pdrop.py
"""
import statistics
import torch

from audiovj.config import FEATURES_DIR, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import load_tracks
from audiovj.evaluate import _track_windows
from audiovj.live.inference import SeqInferenceEngine

DROP = PHRASE_TYPES.index("drop")
OFFSETS = [-2, -1, 0, 1, 2]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FOLD = 7

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
engine = SeqInferenceEngine(MODELS_DIR / "seq_unified.safetensors", device)

tracks = [t for t in load_tracks(TRACKS_DIR)
          if t.cue_points and t.fold == FOLD
          and (FEATURES_DIR / f"{t.track_id}.safetensors").exists()]

at = {o: [] for o in OFFSETS}       # p(drop) samples per offset
argmax_at_d0 = {}                   # what class wins at D0
non_drop_pdrops = []                # p(drop) far from any drop -> false-fire base
pairs = []                          # per drop: (p at D0, p at D+1) for f* interpolation
n_drops = 0

for track in tracks:
    samples = _track_windows(track, FEATURES_DIR)
    if not samples:
        continue
    engine.reset()
    times, probs = [], []
    for t, _lbl, window in samples:
        pred = engine.step_window(window)
        times.append(t)
        probs.append(pred.current_probs)

    drop_idx = []
    for c in track.cue_points:
        if c.phrase_type != "drop":
            continue
        i = min(range(len(times)), key=lambda k: abs(times[k] - c.start_time))
        drop_idx.append(i)
        n_drops += 1
        for o in OFFSETS:
            j = i + o
            if 0 <= j < len(probs):
                at[o].append(probs[j][DROP])
        if i + 1 < len(probs):
            pairs.append((probs[i][DROP], probs[i + 1][DROP]))
        w = probs[i]
        argmax_at_d0[PHRASE_TYPES[max(range(len(w)), key=lambda k: w[k])]] = \
            argmax_at_d0.get(PHRASE_TYPES[max(range(len(w)), key=lambda k: w[k])], 0) + 1

    # False-fire base = downbeats whose ACTUAL phrase is not drop, and which are
    # not adjacent to a drop boundary. Using "far from a drop START" is wrong: a
    # drop section runs 16-24 downbeats and only its start is a cue point, so
    # mid-drop downbeats would be scored as false fires while correctly
    # detecting the drop they are inside of.
    cues = sorted(track.cue_points, key=lambda c: c.start_time)
    for k, (t, w) in enumerate(zip(times, probs)):
        label = None
        for c in cues:
            if c.start_time <= t:
                label = c.phrase_type
            else:
                break
        if label == "drop":
            continue
        if any(abs(k - i) <= 2 for i in drop_idx):
            continue
        non_drop_pdrops.append(w[DROP])

def q(xs, p):
    s = sorted(xs)
    return s[min(int(p * (len(s) - 1)), len(s) - 1)]

print(f"fold {FOLD}: {len(tracks)} tracks, {n_drops} labeled drops\n")
print("p(drop) around the drop downbeat  (D0 = the drop's own downbeat,")
print("window ends AT D0 so it contains ZERO drop audio)\n")
print(f"  {'pos':<5} {'median':>8} {'mean':>8} {'p25':>8} {'p75':>8} {'n':>5}")
for o in OFFSETS:
    v = at[o]
    if not v:
        continue
    lab = f"D{o:+d}" if o else "D0"
    print(f"  {lab:<5} {statistics.median(v):8.3f} {statistics.mean(v):8.3f} "
          f"{q(v,0.25):8.3f} {q(v,0.75):8.3f} {len(v):5d}")

print("\nargmax at D0 (what the lights currently key off):")
for k, v in sorted(argmax_at_d0.items(), key=lambda x: -x[1]):
    print(f"  {k:<12} {v:4d}  ({v / max(n_drops,1) * 100:.0f}%)")

print(f"\nthreshold sweep at D0   (base: {len(non_drop_pdrops)} downbeats >2 bars from any drop)")
print(f"  {'thr':>5} {'on-time recall':>16} {'false-fire rate':>17}")
d0 = at[0]
for T in THRESHOLDS:
    rec = sum(1 for p in d0 if p >= T) / max(len(d0), 1) * 100
    fp = sum(1 for p in non_drop_pdrops if p >= T) / max(len(non_drop_pdrops), 1) * 100
    print(f"  {T:5.2f} {rec:15.1f}% {fp:16.1f}%")

print("\nfor reference, the same sweep one bar later (D+1, today's earliest fire):")
d1 = at[1]
for T in (0.20, 0.30, 0.40):
    rec = sum(1 for p in d1 if p >= T) / max(len(d1), 1) * 100
    print(f"  {T:5.2f} {rec:15.1f}%")


# ---- f*: what fraction of the window must be drop before p(drop) crosses thr ----
# The window is 8 beats and ends at the read. At D0 it holds 0 beats of drop, at
# D+1 it holds 4 of 8 (=0.50). Assuming p rises linearly in that fraction
# (supported by the D0/D+1/D+2 medians), the crossing point gives the earliest
# beat a sub-beat speculative read could fire. LINEARITY IS AN ASSUMPTION --
# confirming it needs raw audio to build windows ending mid-bar.
print("\n" + "=" * 62)
print("f* ESTIMATE (interpolated between D0 and D+1, linearity assumed)")
for T in (0.20, 0.30, 0.40):
    fires = []
    for p0, p1 in pairs:
        if p1 < T:
            continue          # never crosses even at D+1 -> not caught this bar
        if p0 >= T:
            fires.append(0.0)  # already over at D0
        else:
            frac = 0.5 * (T - p0) / max(p1 - p0, 1e-9)   # fraction of window
            fires.append(frac * 8.0)                      # -> beats after the drop
    if not fires:
        continue
    srt = sorted(fires)
    med = srt[len(srt) // 2]
    print(f"  thr {T:.2f}: {len(fires)}/{len(pairs)} drops cross within the bar"
          f" | median fire at {med:+.2f} beats after D0"
          f" | saves {4.0 - med:+.2f} beats vs today")
