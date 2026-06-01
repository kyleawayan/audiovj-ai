"""Canonical State-Manager cueing eval for the full-scale UnifiedSeq, on the
held-out test fold (fold 7) — the goal-1 metric.

LB transition recall = fraction of load-bearing phrase boundaries (intro/
buildup/drop/outro) with an SM fire within 2 bars (8 beats); plus matched
latency and precision. This is "transition timing <=2 bars at >=70% recall on
load-bearing phrases." Mirrors _smfinal.py but on the clean fold split + the
unified production model, sweeping operating points to find one that clears 70%.

  uv run python experiments/_full_sm.py [test|val]
"""

import sys

import torch
from safetensors.torch import load_file

from _arch import build_seqs
from _full import _ckpt_for, fold_split
from _unified import UnifiedSeq
from audiovj.config import PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import load_tracks
from audiovj.live.inference import PredictionResult
from audiovj.live.state import PhraseStateManager as SM

LB = {"intro", "buildup", "drop", "outro"}
LB_IDX = [PHRASE_TYPES.index(p) for p in ("intro", "buildup", "drop", "outro")]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

split = sys.argv[1] if len(sys.argv) > 1 else "test"
ckpt = _ckpt_for(sys.argv[2] if len(sys.argv) > 2 else "")
tr, va, te = fold_split()
ids = te if split == "test" else va
tracks = {t.track_id: t for t in load_tracks(TRACKS_DIR)}
m = UnifiedSeq().to(DEV)
m.load_state_dict(load_file(ckpt)); m.eval()
print(f"ckpt={ckpt.split('/')[-1]}")

# Per-track: (bpm, rows[(time, PredictionResult, cur_prob_vec)], cues[(time, phrase)])
data = []
with torch.no_grad():
    for tid in ids:
        seqs = build_seqs([tid])
        if not seqs:
            continue
        s = seqs[0]
        o = m(s["windows"].to(DEV).unsqueeze(0))
        cp = torch.softmax(o.current_phrase_logits[0], -1).cpu()
        npb = torch.softmax(o.next_phrase_logits[0], -1).cpu()
        bu = torch.expm1(o.beats_until[0, :, 0]).cpu()
        rows = []
        for i in range(len(s["times"])):
            ci, ni = int(cp[i].argmax()), int(npb[i].argmax())
            rows.append((s["times"][i], PredictionResult(
                PHRASE_TYPES[ci], float(cp[i, ci]), PHRASE_TYPES[ni],
                float(npb[i, ni]), float(bu[i])), cp[i]))
        cues = [(c.start_time, c.phrase_type) for c in tracks[tid].cue_points]
        data.append((tracks[tid].bpm, rows, cues))


def eval_fires(fire_fn, lb_only=True):
    lb_det = lb_tot = fires = near = 0
    lat = []
    for bpm, rows, cues in data:
        bd = 60.0 / bpm
        fires_t = fire_fn(rows)
        fires += len(fires_t)
        for ft in fires_t:
            if cues and min(abs(ct - ft) / bd for ct, _ in cues) <= 8:
                near += 1
        for ct, ph in cues[1:]:
            if lb_only and ph not in LB:
                continue
            lb_tot += 1
            if fires_t:
                d = min(abs(ct - ft) / bd for ft in fires_t)
                if d <= 8:
                    lb_det += 1; lat.append(d)
    return (f"LB-recall {lb_det/max(lb_tot,1)*100:4.1f}% ({lb_det}/{lb_tot})  "
            f"prec {near/max(fires,1)*100:4.1f}%  lat {sum(lat)/max(len(lat),1):.1f}b  fires {fires}")


def sm_fires(**cfg):
    def f(rows):
        sm = SM(**cfg); out = []
        for t, pred, _ in rows:
            for e in sm.update(pred):
                if e.kind in ("transition", "correction"):
                    out.append(t)
        return out
    return f


def onset_fires(thresh=0.4):
    def f(rows):
        out = []
        for i in range(1, len(rows)):
            cp, cpp = rows[i][2], rows[i - 1][2]
            for c in LB_IDX:
                if cp[c] >= thresh and cpp[c] < thresh:
                    out.append(rows[i][0]); break
        return out
    return f


print(f"UNIFIED full model — SM cueing on {split} fold: {len(data)} tracks\n")
print("SM current defaults        ", eval_fires(sm_fires()))
print("SM ct0.4 sticky8 wu0       ", eval_fires(sm_fires(correction_threshold=0.4, sticky_beats=8.0, warmup_beats=0.0)))
print("SM ct0.3 sticky8 wu0       ", eval_fires(sm_fires(correction_threshold=0.3, sticky_beats=8.0, warmup_beats=0.0)))
print("SM ct0.3 sticky0 wu0       ", eval_fires(sm_fires(correction_threshold=0.3, sticky_beats=0.0, warmup_beats=0.0)))
print("SM ct0.25 sticky0 wu0      ", eval_fires(sm_fires(correction_threshold=0.25, sticky_beats=0.0, warmup_beats=0.0)))
print("SM ct0.2 sticky0 wu0 latch1", eval_fires(sm_fires(correction_threshold=0.2, sticky_beats=0.0, warmup_beats=0.0, latch_after=1)))
print("onset@0.4  (no SM)         ", eval_fires(onset_fires(0.4)))
print("onset@0.35 (no SM)         ", eval_fires(onset_fires(0.35)))
print("onset@0.3  (no SM)         ", eval_fires(onset_fires(0.3)))
print("onset@0.25 (no SM)         ", eval_fires(onset_fires(0.25)))
