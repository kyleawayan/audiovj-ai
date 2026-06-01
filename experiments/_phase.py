"""Lever 2 (goal: break the LB transition-recall ceiling): bar-phase /
periodicity features.

The model currently sees only mel content per 8-beat window — it has no idea
WHERE in the phrase grid it sits. But EDM transitions are quantized to 8/16/32-
bar boundaries, and intro/outro are position-locked (track start / end). So we
fuse cheap, beat-grid-derived features into each downbeat embedding before the
cross-bar LSTM:

  - phrase-grid phase: (db mod P) as sin/cos for P in {4,8,16,32}  -> 8 feats
  - position in track: db/(T-1) and (T-1-db)/(T-1)                 -> 2 feats

These are legitimately available live: djay Pro beatgrids + Ableton Link give
~99% beat / ~90% downbeat accuracy, so the downbeat index (hence grid phase) is
known at runtime. No audio reprocessing — derived from the downbeat index.

RAM-safe lazy training like _full.py. Clean fold split (train 0-5 / val 6 /
test 7 held out). Reports frame metrics AND the honest onset-cueing operating
point (the goal metric).

  uv run python experiments/_phase.py train [epochs]
  uv run python experiments/_phase.py eval  <val|test>
"""

import math
import os
import sys
import time

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from audiovj.config import (
    ENCODER_CHANNELS, FEATURES_DIR, LSTM_HIDDEN, LSTM_LAYERS, N_MELS,
    NUM_PHRASES, PHRASE_TYPES, TRACKS_DIR,
)
from audiovj.data.dataset import generate_labels
from audiovj.data.rekordbox import Track, build_downbeat_times, load_tracks
from audiovj.live.inference import PredictionResult
from audiovj.model import ModelOutput, SpectrogramEncoder
from audiovj.training import (
    KEY_CLASSES, PhraseLoss, SpecAugment, _compute_class_weights,
    _get_device, _macro_f1_key_classes,
)

CKPT = "/mnt/scratch/data/loop/seq_unified_phase.safetensors"
P2I = {p: i for i, p in enumerate(PHRASE_TYPES)}
DROP = PHRASE_TYPES.index("drop")
LB = {"intro", "buildup", "drop", "outro"}
LB_IDX = [PHRASE_TYPES.index(p) for p in ("intro", "buildup", "drop", "outro")]
GRIDS = (4, 8, 16, 32)
N_PHASE = 2 * len(GRIDS) + 2  # sin/cos per grid + 2 position feats
W_REG, CAP = 0.3, 12.0
# v2 showed reweight helps frame F1/buildup; keep it (wp 0.75, cw_cap 8) + light reg.
WP, CW_CAP, DROPOUT, WD = 0.75, 8.0, 0.4, 3e-4


def phase_feats(dbidx: list[int], total: int) -> torch.Tensor:
    denom = max(total - 1, 1)
    out = []
    for db in dbidx:
        f = []
        for P in GRIDS:
            ph = (db % P) / P
            f += [math.sin(2 * math.pi * ph), math.cos(2 * math.pi * ph)]
        f += [db / denom, (denom - db) / denom]
        out.append(f)
    return torch.tensor(out, dtype=torch.float32)  # [T, N_PHASE]


class UnifiedSeqPhase(nn.Module):
    def __init__(self, dropout=0.3, detach=False):
        super().__init__()
        self.encoder = SpectrogramEncoder(N_MELS, 128, ENCODER_CHANNELS)
        ch = self.encoder.out_channels
        self.ctx_lstm = nn.LSTM(ch + N_PHASE, LSTM_HIDDEN, LSTM_LAYERS,
                                batch_first=True, dropout=dropout if LSTM_LAYERS > 1 else 0.0)
        self.head_dropout = nn.Dropout(dropout)
        self.next_phrase_head = nn.Linear(LSTM_HIDDEN, NUM_PHRASES)
        self.current_phrase_head = nn.Linear(LSTM_HIDDEN, NUM_PHRASES)
        self.beats_branch = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(LSTM_HIDDEN, 1))
        self.detach = detach

    def forward(self, x, phase):
        b, t = x.shape[0], x.shape[1]
        enc = self.encoder(x.reshape(b * t, x.shape[2], x.shape[3]))
        win = enc.mean(dim=1).reshape(b, t, -1)
        h_in = torch.cat([win, phase], dim=-1)
        ctx, _ = self.ctx_lstm(h_in)
        h = self.head_dropout(ctx)
        beats_in = h.detach() if self.detach else h
        return ModelOutput(
            next_phrase_logits=self.next_phrase_head(h),
            current_phrase_logits=self.current_phrase_head(h),
            beats_until=self.beats_branch(beats_in))


def _track_steps(tid):
    """Shared keep/label logic -> (fp, keep_pos, cur, nxt, beats, dbidx, total)."""
    tp = TRACKS_DIR / f"{tid}.json"
    fp = FEATURES_DIR / f"{tid}.safetensors"
    if not tp.exists() or not fp.exists():
        return None
    track = Track.model_validate_json(tp.read_text())
    if not track.cue_points:
        return None
    with safe_open(str(fp), framework="pt") as f:
        kept = f.get_tensor("kept_indices").tolist()
        nwin = f.get_slice("windows").get_shape()[0]
    downbeats = build_downbeat_times(track)
    labels = generate_labels(track, downbeats)
    if not labels:
        return None
    keep_pos, cur, nxt, beats, dbidx = [], [], [], [], []
    for i, db in enumerate(kept):
        if i >= nwin or db >= len(labels):
            break
        lbl = labels[db]
        if lbl is None:
            continue
        keep_pos.append(i); cur.append(P2I[lbl["current_phrase"]])
        nxt.append(P2I[lbl["next_phrase"]]); beats.append(float(lbl["beats_until"]))
        dbidx.append(db)
    if len(keep_pos) < 2:
        return None
    return {
        "fp": str(fp), "keep": torch.tensor(keep_pos),
        "current": torch.tensor(cur), "next": torch.tensor(nxt),
        "beats": torch.tensor(beats),
        "phase": phase_feats(dbidx, len(downbeats)),
        "times": [downbeats[d] for d in dbidx],
        "cue_times": [(c.start_time, c.phrase_type) for c in track.cue_points],
        "bpm": track.bpm,
    }


def fold_split():
    tr, va, te = [], [], []
    for t in load_tracks(TRACKS_DIR):
        if not t.cue_points or t.fold is None:
            continue
        if not (FEATURES_DIR / f"{t.track_id}.safetensors").exists():
            continue
        (te if t.fold == 7 else va if t.fold == 6 else tr).append(t.track_id)
    return tr, va, te


def build_meta(ids):
    return [m for m in (_track_steps(t) for t in ids) if m is not None]


def _load_windows(m):
    return load_file(m["fp"])["windows"][m["keep"]]


def train(epochs=40, lr=1e-3):
    dev = _get_device()
    tr_ids, va_ids, te_ids = fold_split()
    print(f"folds: train {len(tr_ids)} val {len(va_ids)} test {len(te_ids)} (HELD OUT)", flush=True)
    tr = build_meta(tr_ids); va = build_meta(va_ids)
    print(f"train {len(tr)} val {len(va)}  N_PHASE={N_PHASE}", flush=True)
    cw = _compute_class_weights([c for s in tr for c in s["current"].tolist()],
                                NUM_PHRASES, cap=CW_CAP, power=WP).to(dev)
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]
    model = UnifiedSeqPhase(dropout=DROPOUT).to(dev); aug = SpecAugment().to(dev)
    crit = PhraseLoss(w_regression=W_REG, class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    g = torch.Generator().manual_seed(0)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    best = (-1.0, 99.0)
    for ep in range(1, epochs + 1):
        model.train(); aug.train(); t0 = time.time(); tl = 0.0
        for j in torch.randperm(len(tr), generator=g).tolist():
            m = tr[j]; W = _load_windows(m).to(dev)
            w = aug(W).unsqueeze(0); ph = m["phase"].to(dev).unsqueeze(0)
            o = model(w, ph)
            beats = m["beats"].float().clamp(max=CAP).to(dev)
            loss = crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.current_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.beats_until.reshape(-1, 1),
                        m["next"].to(dev), m["current"].to(dev), beats)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item()
        model.eval(); aug.eval()
        tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
        vl = 0.0; mae = []
        with torch.no_grad():
            for s in va:
                o = model(s["windows"].to(dev).unsqueeze(0) if "windows" in s
                          else _load_windows(s).to(dev).unsqueeze(0), s["phase"].to(dev).unsqueeze(0))
                cl = o.current_phrase_logits.reshape(-1, NUM_PHRASES)
                beats = s["beats"].float().clamp(max=CAP).to(dev)
                vl += crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES), cl, o.beats_until.reshape(-1, 1),
                           s["next"].to(dev), s["current"].to(dev), beats).item()
                cp = cl.argmax(-1).cpu(); gt = s["current"]
                for c in range(NUM_PHRASES):
                    pc = cp == c; gc = gt == c
                    tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
                pbu = torch.expm1(o.beats_until[0, :, 0]).cpu(); tb = s["beats"]
                for kk in range(len(tb)):
                    if 0 < tb[kk] <= 8:
                        mae.append(abs(pbu[kk].item() - float(tb[kk])))
        f1, per = _macro_f1_key_classes(tp, fp, fn, key); cmae = sum(mae) / max(len(mae), 1)
        sch.step(vl / max(len(va), 1)); tag = ""
        if f1 >= 0.55 and (f1 > best[0] + 1e-6 or (abs(f1 - best[0]) <= 0.01 and cmae < best[1])):
            best = (f1, cmae); save_file(model.state_dict(), CKPT); tag = " *"
        brk = " ".join(f"{KEY_CLASSES[i]}={per[i]:.2f}" for i in range(len(KEY_CLASSES)))
        print(f"ep {ep:3d}/{epochs} {time.time()-t0:4.0f}s val {vl/max(len(va),1):.3f} "
              f"mF1 {f1:.3f} cd_mae {cmae:4.1f} [{brk}]{tag}", flush=True)
    print(f"\nsaved best mF1 {best[0]:.3f} cd_mae {best[1]:.1f} -> {CKPT}", flush=True)


def _windows_for(s):
    return _load_windows(s)


def evaluate(split="test"):
    dev = _get_device()
    tr, va, te = fold_split()
    ids = te if split == "test" else va
    seqs = build_meta(ids)
    print(f"=== PHASE model EVAL on {split} fold ({len(seqs)} tracks) ===", flush=True)
    m = UnifiedSeqPhase().to(dev); m.load_state_dict(load_file(CKPT)); m.eval()

    tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]
    dtp = dfp = dfn = dtn = 0
    # onset cueing accumulators per threshold
    THR = [0.4, 0.35, 0.3, 0.25]
    lb_det = {th: 0 for th in THR}; near = {th: 0 for th in THR}; fires = {th: 0 for th in THR}
    lat = {th: [] for th in THR}; lb_tot = 0
    mae = []; mono_ok = mono_tot = 0; warn_det = warn_tot = 0

    with torch.no_grad():
        for s in seqs:
            W = _windows_for(s).to(dev)
            o = m(W.unsqueeze(0), s["phase"].to(dev).unsqueeze(0))
            cprob = torch.softmax(o.current_phrase_logits[0], -1).cpu()
            pcur = cprob.argmax(-1).tolist()
            pbu = torch.expm1(o.beats_until[0, :, 0]).cpu().tolist()
            nconf, nidx = torch.softmax(o.next_phrase_logits[0], -1).max(-1)
            pnext = nidx.tolist(); pconf = nconf.tolist()
            tcur = s["current"].tolist(); tnext = s["next"].tolist(); tbu = s["beats"].tolist()
            times = s["times"]; bd = 60.0 / s["bpm"]; cues = s["cue_times"]

            cp = torch.tensor(pcur); gt = s["current"]
            for c in range(NUM_PHRASES):
                pc = cp == c; gc = gt == c
                tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
            ds = [c == DROP for c in pcur]; ts = [c == DROP for c in tcur]
            for k in range(len(tcur)):
                dtp += ds[k] and ts[k]; dfp += ds[k] and not ts[k]
                dfn += (not ds[k]) and ts[k]; dtn += (not ds[k]) and (not ts[k])

            # onset cueing at several thresholds
            lb_bound = [ct for ct, ph in cues[1:] if ph in LB]
            lb_tot += len(lb_bound)
            for th in THR:
                ft = []
                for i in range(1, len(times)):
                    if any(cprob[i, c] >= th and cprob[i - 1, c] < th for c in LB_IDX):
                        ft.append(times[i])
                fires[th] += len(ft)
                for f in ft:
                    if cues and min(abs(ct - f) / bd for ct, _ in cues) <= 8:
                        near[th] += 1
                for ct in lb_bound:
                    if ft:
                        d = min(abs(f - ct) / bd for f in ft)
                        if d <= 8:
                            lb_det[th] += 1; lat[th].append(d)

            # countdown + warning
            cd = None
            for k in range(len(times)):
                hot = pnext[k] == DROP and pconf[k] > 0.5
                r4 = max(round(pbu[k] / 4) * 4, 0); prev = cd
                cd = (max(r4, 4) if hot else None) if cd is None else (min(cd - 4, r4) if hot else cd - 4)
                if cd is not None and cd <= 0:
                    cd = None
                if cd is not None and prev is not None and 0 < tbu[k] <= 8:
                    mono_tot += 1; mono_ok += int(cd <= prev)
                if cd is not None and 0 < tbu[k] <= 8:
                    mae.append(abs(cd - tbu[k]))
            i = 0
            while i < len(tnext):
                if tnext[i] == DROP:
                    j = i
                    while j < len(tnext) and tnext[j] == DROP:
                        j += 1
                    warn_tot += 1
                    if any(pnext[k] == DROP and pconf[k] > 0.5 and tbu[k] >= 4 for k in range(i, j)):
                        warn_det += 1
                    i = j
                else:
                    i += 1

    f1, _ = _macro_f1_key_classes(tp, fp, fn, key)
    rec = {KEY_CLASSES[i]: (tp[key[i]] / max(tp[key[i]] + fn[key[i]], 1)).item() for i in range(len(KEY_CLASSES))}
    acc = (dtp + dtn) / max(dtp + dfp + dfn + dtn, 1) * 100
    print("\n-- DETECTION --")
    print(f"  load-bearing macro-F1: {f1:.3f}")
    print("  per-class recall (frame): " + "  ".join(f"{k} {v*100:.0f}%" for k, v in rec.items()))
    print(f"  in-drop frame: acc {acc:.1f}%  recall {dtp/max(dtp+dfn,1)*100:.1f}%  prec {dtp/max(dtp+dfp,1)*100:.1f}%")
    print("\n-- LB TRANSITION CUEING (onset, honest operating point) --")
    for th in THR:
        print(f"  onset@{th:<4}: LB-recall {lb_det[th]/max(lb_tot,1)*100:4.1f}%  "
              f"prec {near[th]/max(fires[th],1)*100:4.1f}%  "
              f"lat {sum(lat[th])/max(len(lat[th]),1):.1f}b  fires {fires[th]}")
    print("\n-- ANTICIPATION --")
    print(f"  pre-drop warning (>=1 bar): recall {warn_det/max(warn_tot,1)*100:.1f}%  (n={warn_tot})")
    print(f"  countdown MAE (final 2 bars): {sum(mae)/max(len(mae),1):.2f}b  monotone {mono_ok/max(mono_tot,1)*100:.0f}%")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "test")
