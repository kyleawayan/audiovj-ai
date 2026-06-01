"""Are we under-measuring LB transition recall by using the wrong cueing method?

The onset detector (rising-edge of LB class prob crossing a threshold) needs a
SHARP prob jump at the boundary. A model with smooth/confident labels (esp. the
bidir one) has good frame labels but smeared onsets -> onset undercounts it.
This compares two cueing methods on the same model outputs:
  - onset@thr            : prob rising-edge (what we've been using)
  - argmax-flip(confirm) : debounced argmax current-phrase label CHANGES into an
                           LB class (uses the frame labels directly)

  uv run python experiments/_cue.py <base|v2|phase|bidir> [test|val]
"""

import sys

import torch
from safetensors.torch import load_file

from _arch import build_seqs
from _full import _ckpt_for, fold_split
from audiovj.config import PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import load_tracks

LBN = ("intro", "buildup", "drop", "outro")
LB = set(LBN)
LB_IDX = [PHRASE_TYPES.index(p) for p in LBN]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_key = sys.argv[1] if len(sys.argv) > 1 else "v2"
split = sys.argv[2] if len(sys.argv) > 2 else "test"

if model_key == "bidir":
    from _bidir import BiUnifiedSeq
    m = BiUnifiedSeq().to(DEV)
    ckpt = "/mnt/scratch/data/loop/seq_unified_bidir.safetensors"
else:
    from _unified import UnifiedSeq
    m = UnifiedSeq().to(DEV)
    ckpt = _ckpt_for("" if model_key == "base" else model_key)
m.load_state_dict(load_file(ckpt)); m.eval()

tr, va, te = fold_split()
ids = te if split == "test" else va
tracks = {t.track_id: t for t in load_tracks(TRACKS_DIR)}
print(f"model={model_key}  ckpt={ckpt.split('/')[-1]}  split={split} ({len(ids)} tracks)\n")

data = []
with torch.no_grad():
    for tid in ids:
        seqs = build_seqs([tid])
        if not seqs:
            continue
        s = seqs[0]
        o = m(s["windows"].to(DEV).unsqueeze(0))
        cprob = torch.softmax(o.current_phrase_logits[0], -1).cpu().tolist()
        data.append({"times": s["times"], "cprob": cprob, "bpm": s["bpm"],
                     "cues": [(c.start_time, c.phrase_type) for c in tracks[tid].cue_points]})


def score(fire_fn, near_beats=8.0):
    det = tot = fires = near = 0
    lat = []
    cd = {p: 0 for p in LBN}; ct_ = {p: 0 for p in LBN}
    for d in data:
        bd = 60.0 / d["bpm"]; ft = fire_fn(d)
        fires += len(ft)
        for f in ft:
            if d["cues"] and min(abs(c0 - f) / bd for c0, _ in d["cues"]) <= near_beats:
                near += 1
        for c0, ph in d["cues"][1:]:
            if ph not in LB:
                continue
            tot += 1; ct_[ph] += 1
            if ft and min(abs(f - c0) / bd for f in ft) <= near_beats:
                det += 1; cd[ph] += 1
                lat.append(min(abs(f - c0) / bd for f in ft))
    cls = {p: cd[p] / max(ct_[p], 1) * 100 for p in LBN}
    return (det / max(tot, 1) * 100, near / max(fires, 1) * 100,
            sum(lat) / max(len(lat), 1), fires, cls)


def onset(thr):
    def f(d):
        cp = d["cprob"]
        return [d["times"][i] for i in range(1, len(cp))
                if any(cp[i][c] >= thr and cp[i - 1][c] < thr for c in LB_IDX)]
    return f


def argmax_flip(confirm=2):
    def f(d):
        cp = d["cprob"]; times = d["times"]
        lab = [max(range(len(c)), key=lambda k: c[k]) for c in cp]
        out = []; cur = lab[0]; cand = None; cc = 0
        for i in range(1, len(lab)):
            if lab[i] == cur:
                cand = None; cc = 0
            elif lab[i] == cand:
                cc += 1
            else:
                cand = lab[i]; cc = 1
            if cand is not None and cc >= confirm:
                if cand in LB_IDX:
                    out.append(times[i])
                cur = cand; cand = None; cc = 0
        return out
    return f


def show(name, fn):
    r, p, l, fires, cls = score(fn)
    cb = " ".join(f"{k[:4]} {cls[k]:.0f}" for k in LBN)
    print(f"{name:26s} LB-rec {r:4.1f}%  prec {p:4.1f}%  lat {l:.1f}b  fires {fires:5d} | {cb}")


print("onset (prob rising-edge):")
for th in (0.4, 0.3, 0.25):
    show(f"  onset@{th}", onset(th))
print("argmax-flip (debounced label change):")
for cf in (1, 2, 3):
    show(f"  flip confirm={cf}", argmax_flip(cf))

print("\nTOLERANCE sweep (onset@0.3): are misses true-blind, or near-misses?")
for nb in (4, 8, 12, 16, 24):
    r, p, l, fires, cls = score(onset(0.3), near_beats=nb)
    cb = " ".join(f"{k[:4]} {cls[k]:.0f}" for k in LBN)
    print(f"  within {nb:2d} beats ({nb//4} bar): LB-rec {r:4.1f}%  | {cb}")
