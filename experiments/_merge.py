"""Iteration 5: merged 5-class vocab + longer-context seq model.

KA-233's proposed merge consolidates confusable classes so the model isn't
splitting probability mass (esp. the ending region outro/altoutro/end, which
crushed outro-boundary recall). Merged vocab:
  intro     <- intro, altintro
  buildup   <- buildup
  drop      <- drop
  breakdown <- breakdown, bridge, cooldown
  outro     <- outro, altoutro, end
Load-bearing = intro/buildup/drop/outro (4 of 5). Retrains the seq model and
re-measures load-bearing boundary recall (the one metric short of the goal).
"""

import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from _arch import SeqPhrasePredictor
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits, generate_labels
from audiovj.data.rekordbox import Track, build_downbeat_times, load_tracks
from audiovj.training import (
    PhraseLoss, SpecAugment, _compute_class_weights, _get_device, _macro_f1_key_classes,
)

MERGE = {
    "intro": "intro", "altintro": "intro",
    "buildup": "buildup",
    "drop": "drop",
    "breakdown": "breakdown", "bridge": "breakdown", "cooldown": "breakdown",
    "outro": "outro", "altoutro": "outro", "end": "outro",
}
MTYPES = ["intro", "buildup", "drop", "breakdown", "outro"]
MP2I = {p: i for i, p in enumerate(MTYPES)}
NUMP = len(MTYPES)
KEY = ["intro", "buildup", "drop", "outro"]
KEY_IDX = [MP2I[p] for p in KEY]
LB_IDX = KEY_IDX
CKPT = "/mnt/scratch/data/loop/seq_merged.safetensors"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_seqs(ids):
    seqs = []
    for tid in ids:
        tp = TRACKS_DIR / f"{tid}.json"; fp = FEATURES_DIR / f"{tid}.safetensors"
        if not tp.exists() or not fp.exists():
            continue
        track = Track.model_validate_json(tp.read_text())
        if not track.cue_points:
            continue
        data = load_file(str(fp)); windows = data["windows"]; kept = data["kept_indices"].tolist()
        downbeats = build_downbeat_times(track); labels = generate_labels(track, downbeats)
        if not labels:
            continue
        wl, cur, nxt, beats, times = [], [], [], [], []
        for i, db in enumerate(kept):
            if i >= windows.shape[0] or db >= len(labels):
                break
            lbl = labels[db]
            if lbl is None:
                continue
            wl.append(windows[i])
            cur.append(MP2I[MERGE[lbl["current_phrase"]]])
            nxt.append(MP2I[MERGE[lbl["next_phrase"]]])
            beats.append(float(lbl["beats_until"])); times.append(downbeats[db])
        if len(wl) < 2:
            continue
        seqs.append({"windows": torch.stack(wl), "current": torch.tensor(cur),
                     "next": torch.tensor(nxt), "beats": torch.tensor(beats), "times": times,
                     "cues": [(c.start_time, MERGE[c.phrase_type]) for c in track.cue_points],
                     "bpm": track.bpm})
    return seqs


def train(epochs=30, lr=1e-3, dropout=0.3, wd=1e-4, wp=0.5):
    _get_device()
    tr_ids, va_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    tr, va = build_seqs(tr_ids), build_seqs(va_ids)
    print(f"train {len(tr)} val {len(va)}  downbeats {sum(len(s['current']) for s in tr)}")
    cw = _compute_class_weights([c for s in tr for c in s["current"].tolist()], NUMP, cap=5.0, power=wp).to(DEV)
    model = SeqPhrasePredictor(dropout=dropout, num_phrases=NUMP).to(DEV)
    aug = SpecAugment().to(DEV); crit = PhraseLoss(class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    g = torch.Generator().manual_seed(0); best = -1.0
    for ep in range(1, epochs + 1):
        model.train(); aug.train()
        for j in torch.randperm(len(tr), generator=g).tolist():
            s = tr[j]; w = aug(s["windows"].to(DEV)).unsqueeze(0); o = model(w)
            loss = crit(o.next_phrase_logits.reshape(-1, NUMP), o.current_phrase_logits.reshape(-1, NUMP),
                        o.beats_until.reshape(-1, 1), s["next"].to(DEV), s["current"].to(DEV), s["beats"].float().to(DEV))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); aug.eval()
        tp = torch.zeros(NUMP); fp = torch.zeros(NUMP); fn = torch.zeros(NUMP); vl = 0.0; cc = tot = 0
        with torch.no_grad():
            for s in va:
                o = model(s["windows"].to(DEV).unsqueeze(0))
                cl = o.current_phrase_logits.reshape(-1, NUMP)
                vl += crit(o.next_phrase_logits.reshape(-1, NUMP), cl, o.beats_until.reshape(-1, 1),
                           s["next"].to(DEV), s["current"].to(DEV), s["beats"].float().to(DEV)).item()
                cp = cl.argmax(-1).cpu(); gt = s["current"]; cc += (cp == gt).sum().item(); tot += len(gt)
                for c in range(NUMP):
                    pc = cp == c; gc = gt == c
                    tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
        f1, per = _macro_f1_key_classes(tp, fp, fn, KEY_IDX); sch.step(vl / max(len(va), 1))
        tag = ""
        if f1 > best:
            best = f1; save_file(model.state_dict(), CKPT); tag = " *"
        brk = " ".join(f"{KEY[i]}={per[i]:.2f}" for i in range(len(KEY)))
        print(f"ep {ep:3d} val {vl/max(len(va),1):.3f} cur_acc {cc/max(tot,1)*100:.1f}% mF1 {f1:.3f} [{brk}]{tag}", flush=True)
    print(f"best mF1 {best:.3f} -> {CKPT}")


def detect_onset(times, cprob, thresh):
    f = []
    for t in range(1, len(times)):
        for c in LB_IDX:
            if cprob[t, c] >= thresh and cprob[t - 1, c] < thresh:
                f.append(times[t]); break
    return f


def detect_flip(times, cur):
    f = []; prev = None
    for t, c in zip(times, cur):
        if prev is not None and c != prev:
            f.append(t)
        prev = c
    return f


def evaluate():
    _, va_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    model = SeqPhrasePredictor(num_phrases=NUMP).to(DEV); model.load_state_dict(load_file(CKPT)); model.eval()
    preds = []
    with torch.no_grad():
        for s in build_seqs(va_ids):
            o = model(s["windows"].to(DEV).unsqueeze(0))
            cprob = torch.softmax(o.current_phrase_logits[0], -1).cpu()
            cur = [MTYPES[i] for i in cprob.argmax(-1).tolist()]
            preds.append((s["bpm"], s["times"], cur, cprob, s["cues"]))

    def measure(detect):
        det, tot, lat, fires = defaultdict(int), defaultdict(int), [], 0
        for bpm, times, cur, cprob, cues in preds:
            bd = 60.0 / bpm; f = detect(times, cur, cprob); fires += len(f)
            for ct, ph in cues[1:]:
                tot[ph] += 1
                if f:
                    d = min(abs(ct - x) / bd for x in f)
                    if d <= 8:
                        det[ph] += 1; lat.append(d)
        lb_d = sum(det[p] for p in KEY); lb_t = sum(tot[p] for p in KEY)
        return det, tot, lb_d, lb_t, lat, fires

    print(f"\nval tracks {len(preds)}  (merged 5-class vocab)\n")
    for name, fn in [("current-flip", lambda ti, c, cp: detect_flip(ti, c)),
                     ("onset 0.35", lambda ti, c, cp: detect_onset(ti, cp, 0.35)),
                     ("onset 0.40", lambda ti, c, cp: detect_onset(ti, cp, 0.40)),
                     ("onset 0.45", lambda ti, c, cp: detect_onset(ti, cp, 0.45)),
                     ("flip OR onset 0.40", lambda ti, c, cp: sorted(set(detect_flip(ti, c)) | set(detect_onset(ti, cp, 0.40))))]:
        det, tot, lb_d, lb_t, lat, fires = measure(fn)
        per = "  ".join(f"{p}={det[p]/tot[p]*100:.0f}%({det[p]}/{tot[p]})" for p in KEY if tot[p])
        print(f"== {name} ==  LB-recall {lb_d/max(lb_t,1)*100:.1f}% ({lb_d}/{lb_t})  "
              f"lat {sum(lat)/max(len(lat),1):.1f}b  fires {fires}")
        print(f"     {per}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    else:
        evaluate()
