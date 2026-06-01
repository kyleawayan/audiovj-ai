"""Human-like stateful phrase tracker (goal: break the flat ~58% LB transition
recall ceiling) — the mechanism Kyle described from how he DJs.

Two systems at once, both causal + live (stereo audio + Link beats only):
  - LISTEN: the model's per-downbeat current-phrase probabilities (content).
  - EXPECT: a soft, RE-SYNCABLE phrase-phase. Track "bars since the last
    boundary I HEARD" + an adaptive phrase-length estimate (init ~32 beats =
    8 bars, but EMA-updates from THIS track's recent boundary spacing). Near the
    expected boundary, LOWER the evidence bar (a faint riser is enough); far
    from it, RAISE it (kill mid-phrase false fires). Re-sync the clock on every
    detected boundary; strong audio (a hard drop) overrides expectation and
    re-anchors there — exactly "oh, something different... ok, picked it back up".

NOT absolute: nothing is anchored to track start or known length (those aren't
live-knowable and can't re-sync). See memory: raveform-live-no-phrase-grid.

This is an INFERENCE-LAYER wrapper over the existing model — no retraining. Runs
over a cached per-downbeat prediction set so configs sweep instantly.

  uv run python experiments/_tracker.py [test|val] [ckpt-tag]   # default: test v2
"""

import math
import sys

import torch
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
tag = sys.argv[2] if len(sys.argv) > 2 else "v2"
ckpt = _ckpt_for(tag)
tr, va, te = fold_split()
ids = te if split == "test" else va
tracks = {t.track_id: t for t in load_tracks(TRACKS_DIR)}
m = UnifiedSeq().to(DEV); m.load_state_dict(load_file(ckpt)); m.eval()
print(f"ckpt={ckpt.split('/')[-1]}  split={split}  ({len(ids)} tracks)\n")

# Cache per-downbeat current-phrase probs (GPU forward once; sweep on CPU).
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


def score(fire_fn):
    det = tot = fires = near = 0
    lat = []
    cls_det = {p: 0 for p in LBN}; cls_tot = {p: 0 for p in LBN}
    for d in data:
        bd = 60.0 / d["bpm"]; ft = fire_fn(d)
        fires += len(ft)
        for f in ft:
            if d["cues"] and min(abs(ct - f) / bd for ct, _ in d["cues"]) <= 8:
                near += 1
        for ct, ph in d["cues"][1:]:
            if ph not in LB:
                continue
            tot += 1; cls_tot[ph] += 1
            if ft:
                dd = min(abs(f - ct) / bd for f in ft)
                if dd <= 8:
                    det += 1; lat.append(dd); cls_det[ph] += 1
    rec = det / max(tot, 1) * 100
    prec = near / max(fires, 1) * 100
    ll = sum(lat) / max(len(lat), 1)
    cls = {p: cls_det[p] / max(cls_tot[p], 1) * 100 for p in LBN}
    return rec, prec, ll, fires, cls


def onset(thr):
    def f(d):
        cp = d["cprob"]; out = []
        for i in range(1, len(cp)):
            if any(cp[i][c] >= thr and cp[i - 1][c] < thr for c in LB_IDX):
                out.append(d["times"][i])
        return out
    return f


def tracker(base=0.4, gate=0.22, width=2.0, override=0.65, min_gap=3,
            len0=8.0, ema=0.6, adapt=True):
    """Re-syncable phrase tracker. len in BARS (downbeat steps); 8 bars = 32 beats."""
    def f(d):
        cp = d["cprob"]; times = d["times"]; out = []
        h = max(range(len(cp[0])), key=lambda c: cp[0][c])  # initial held phrase
        since = 0; lenest = len0
        for i in range(1, len(cp)):
            since += 1
            best_c, best_v = -1, -1.0
            for c in LB_IDX:           # evidence: best NEW load-bearing phrase
                if c == h:
                    continue
                if cp[i][c] > best_v:
                    best_v, best_c = cp[i][c], c
            exp_w = math.exp(-0.5 * ((since - lenest) / width) ** 2)  # peaks at expected boundary
            thr = max(base - gate * exp_w, 0.12)
            fire = (best_v >= thr and since >= min_gap) or best_v >= override
            if fire:
                out.append(times[i])
                if adapt and since >= 2:
                    lenest = min(max(ema * lenest + (1 - ema) * since, 4.0), 16.0)
                since = 0; h = best_c
        return out
    return f


def show(name, fire_fn):
    rec, prec, ll, fires, cls = score(fire_fn)
    cb = " ".join(f"{p[:4]} {cls[p]:.0f}" for p in LBN)
    print(f"{name:38s} LB-rec {rec:4.1f}%  prec {prec:4.1f}%  lat {ll:.1f}b  "
          f"fires {fires:5d}  | {cb}")


print("REFERENCE (no expectation):")
show("onset@0.40", onset(0.40))
show("onset@0.30", onset(0.30))
show("onset@0.25", onset(0.25))
print("\nPHRASE TRACKER (expectation-gated, re-syncable, adaptive len):")
show("base0.40 gate0.22 w2 ovr0.65", tracker(base=0.40, gate=0.22, width=2.0, override=0.65))
show("base0.40 gate0.30 w2 ovr0.65", tracker(base=0.40, gate=0.30, width=2.0, override=0.65))
show("base0.45 gate0.30 w2.5 ovr0.6", tracker(base=0.45, gate=0.30, width=2.5, override=0.60))
show("base0.45 gate0.35 w2.5 ovr0.6", tracker(base=0.45, gate=0.35, width=2.5, override=0.60))
show("base0.50 gate0.40 w3 ovr0.6", tracker(base=0.50, gate=0.40, width=3.0, override=0.60))
show("base0.50 gate0.40 w3 NOADAPT", tracker(base=0.50, gate=0.40, width=3.0, override=0.60, adapt=False))
