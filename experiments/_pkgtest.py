"""End-to-end package-path validation: the production components
(SeqInferenceEngine stateful streaming + OnsetCueTracker @0.30) must reproduce
the locked operating point (~58% LB transition recall) on the held-out fold."""

import torch

from _arch import build_seqs
from _full import fold_split
from audiovj.config import PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import load_tracks
from audiovj.live.cue import OnsetCueTracker
from audiovj.live.inference import SeqInferenceEngine

LBN = ("intro", "buildup", "drop", "outro")
LB = set(LBN)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "/mnt/scratch/data/loop/seq_unified_full_v2.safetensors"

eng = SeqInferenceEngine(__import__("pathlib").Path(CKPT), DEV)
tr, va, te = fold_split()
tracks = {t.track_id: t for t in load_tracks(TRACKS_DIR)}

det = tot = fires = near = 0
lat = []
cd = {p: 0 for p in LBN}; ct_ = {p: 0 for p in LBN}
drop_starts = drop_ends = 0
for tid in te:
    seqs = build_seqs([tid])
    if not seqs:
        continue
    s = seqs[0]; times = s["times"]; bd = 60.0 / s["bpm"]
    cues = [(c.start_time, c.phrase_type) for c in tracks[tid].cue_points]
    eng.reset(); trk = OnsetCueTracker(onset_threshold=0.30)
    ft = []
    for i in range(len(times)):
        pred = eng.step_window(s["windows"][i])
        for e in trk.update(pred):
            if e.kind == "transition":
                ft.append(times[i])
            elif e.kind == "drop_start":
                drop_starts += 1
            elif e.kind == "drop_end":
                drop_ends += 1
    fires += len(ft)
    for f in ft:
        if cues and min(abs(c0 - f) / bd for c0, _ in cues) <= 8:
            near += 1
    for c0, ph in cues[1:]:
        if ph not in LB:
            continue
        tot += 1; ct_[ph] += 1
        if ft and min(abs(f - c0) / bd for f in ft) <= 8:
            det += 1; cd[ph] += 1; lat.append(min(abs(f - c0) / bd for f in ft))

cls = " ".join(f"{p[:4]} {cd[p]/max(ct_[p],1)*100:.0f}" for p in LBN)
print(f"PACKAGE PATH (SeqInferenceEngine + OnsetCueTracker@0.30) on test fold ({len(te)} tracks):")
print(f"  LB transition recall {det/max(tot,1)*100:.1f}%  prec {near/max(fires,1)*100:.1f}%  "
      f"lat {sum(lat)/max(len(lat),1):.1f}b  fires {fires}")
print(f"  per-class: {cls}")
print(f"  drop_start events {drop_starts}  drop_end events {drop_ends}")
print("\n  expected (from experiments/_cue.py onset@0.30): ~57.7% recall, ~50% prec  -> MATCH?")
