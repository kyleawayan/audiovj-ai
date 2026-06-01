"""Durable certification of the locked operating point: are the 2-bar
"near-misses" label-offset (model is right, label is quantized) or model error?

Objective stand-in for ear-checking: derive an AUDIO change-point signal from the
cached mel windows (no audio reload) and test whether the model's fires sit on
real audio changes BETTER than the labels do. If model fires land on stronger
novelty than the labels, and labels are systematically offset from the nearest
audio change, then the label-matched recall (58% @2-bar) UNDERSTATES the model
(true audio-aligned recall is higher) — which durably validates the operating
point.

novelty[i] = 1 - cos(meanmel[i], meanmel[i-1]) over downbeats (a timbral/energy
change-point curve). Peaks = where the audio actually changes.

  uv run python experiments/_validate.py [test|val]
"""

import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from _arch import build_seqs
from _full import _ckpt_for, fold_split
from _unified import UnifiedSeq
from audiovj.config import PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import load_tracks

LBN = ("intro", "buildup", "drop", "outro")
LB = set(LBN)
LB_IDX = [PHRASE_TYPES.index(p) for p in LBN]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split = sys.argv[1] if len(sys.argv) > 1 else "test"
ckpt = _ckpt_for("v2")
m = UnifiedSeq().to(DEV); m.load_state_dict(load_file(ckpt)); m.eval()
tr, va, te = fold_split()
ids = te if split == "test" else va
tracks = {t.track_id: t for t in load_tracks(TRACKS_DIR)}
print(f"validate ckpt={ckpt.split('/')[-1]} split={split} ({len(ids)} tracks)\n")


def novelty_curve(windows):
    """[T,128,Fr] -> per-downbeat timbral-change novelty (cosine distance to prev)."""
    mm = windows.mean(dim=-1)                       # [T,128] mean-mel profile
    mm = F.normalize(mm, dim=-1)
    nov = torch.zeros(mm.shape[0])
    nov[1:] = 1 - (mm[1:] * mm[:-1]).sum(-1)        # 1 - cos(adjacent)
    return nov


def peaks(nov, q=0.70):
    """downbeat indices that are local maxima above the per-track q-quantile."""
    if len(nov) < 3:
        return set()
    thr = torch.quantile(nov, q)
    out = set()
    for i in range(1, len(nov) - 1):
        if nov[i] >= thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]:
            out.add(i)
    return out


def nearest_bar_dist(idx, peakset):
    if not peakset:
        return None
    return min(abs(idx - p) for p in peakset)


# accumulators
lab_on_peak = lab_tot = 0          # labels sitting on an audio change (within 1 bar)
fire_on_peak = fire_tot = 0
lab_off = []                       # label -> nearest peak distance (bars)
fire_off = []                      # fire  -> nearest peak distance (bars)
nov_at_label = []; nov_at_fire = []; nov_at_rand = []
offset_nearmiss = 0; nearmiss_tot = 0   # 2-bar misses that are label-offset (model fired on a peak)

with torch.no_grad():
    for tid in ids:
        seqs = build_seqs([tid])
        if not seqs:
            continue
        s = seqs[0]; times = s["times"]; bd = 60.0 / s["bpm"]
        nov = novelty_curve(s["windows"])
        pk = peaks(nov)
        o = m(s["windows"].to(DEV).unsqueeze(0))
        cprob = torch.softmax(o.current_phrase_logits[0], -1).cpu()
        # model fires (onset@0.3) as downbeat indices
        fires = [i for i in range(1, len(times))
                 if any(cprob[i, c] >= 0.30 and cprob[i - 1, c] < 0.30 for c in LB_IDX)]
        # label LB boundary downbeat indices
        labs = []
        for ct, ph in [(c.start_time, c.phrase_type) for c in tracks[tid].cue_points][1:]:
            if ph not in LB:
                continue
            li = min(range(len(times)), key=lambda k: abs(times[k] - ct))
            labs.append(li)
        if not pk:
            continue
        rng = list(range(1, len(times)))
        rand = rng[:: max(len(rng) // max(len(labs), 1), 1)][:len(labs)]  # deterministic "random" sample

        for li in labs:
            d = nearest_bar_dist(li, pk); lab_off.append(d); lab_tot += 1
            lab_on_peak += int(d <= 1); nov_at_label.append(float(nov[li]))
        for fi in fires:
            d = nearest_bar_dist(fi, pk); fire_off.append(d); fire_tot += 1
            fire_on_peak += int(d <= 1); nov_at_fire.append(float(nov[fi]))
        for ri in rand:
            nov_at_rand.append(float(nov[ri]))

        # near-miss attribution: label missed at 2 bars, but a fire sits on a peak 2-4 bars away
        for li in labs:
            nf = min((abs(fi - li), fi) for fi in fires) if fires else None
            if nf is None:
                continue
            dist_bars = nf[0]  # downbeat steps == bars
            if dist_bars > 2:  # missed at 2-bar (8 beats ~ 2 bars)
                nearmiss_tot += 1
                fi = nf[1]
                if nearest_bar_dist(fi, pk) <= 1 and (nearest_bar_dist(li, pk) or 9) > 1:
                    offset_nearmiss += 1


def mean(x):
    return sum(x) / max(len(x), 1)

print("AUDIO-ALIGNMENT (does the model fire on real audio change-points?)")
print(f"  labels on an audio change (<=1 bar from a novelty peak): {lab_on_peak/max(lab_tot,1)*100:.0f}%")
print(f"  model fires on an audio change (<=1 bar from a peak):    {fire_on_peak/max(fire_tot,1)*100:.0f}%")
print(f"  mean novelty (z within track) at: label {mean(nov_at_label):.3f}  "
      f"fire {mean(nov_at_fire):.3f}  random {mean(nov_at_rand):.3f}")
print(f"  mean |offset to nearest audio peak|: labels {mean([d for d in lab_off if d is not None]):.1f} bars  "
      f"fires {mean([d for d in fire_off if d is not None]):.1f} bars")
print(f"\nLABEL-OFFSET near-miss attribution:")
print(f"  of {nearmiss_tot} boundaries missed at 2-bar, "
      f"{offset_nearmiss} ({offset_nearmiss/max(nearmiss_tot,1)*100:.0f}%) have the model firing on a "
      f"real audio change-point that the LABEL is offset from")
