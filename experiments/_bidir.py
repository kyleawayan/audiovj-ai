"""Diagnostic (goal: "if 70% is a hard ceiling, prove it"): does FULL FUTURE
context break the ~58% LB transition-recall ceiling?

A buildup/outro is defined by what comes AFTER it. A causal (live) model can't
see the future, so it must recognize these from their own audio. This variant
makes the cross-bar LSTM **bidirectional** — full-track context, both
directions — to separate two hypotheses:

  - bidir ALSO caps ~58%  -> boundaries aren't in the audio  -> HARD CEILING
                             (lock the honest operating point; goal satisfied by proof)
  - bidir jumps to ~75%+  -> the limiter is CAUSALITY         -> lever is the
                             anticipation/countdown head, not detection

NOT live-deployable for real-time detection (needs the future), but it is a
clean diagnostic AND directly usable for the offline timeline tool (full-track
context is legitimate there). Content-only — no phrase-grid / position
assumptions (see memory: raveform-live-no-phrase-grid). Reuses _full.py's clean
fold split + RAM-safe lazy loader.

  uv run python experiments/_bidir.py train [epochs]
  uv run python experiments/_bidir.py eval  <val|test>
"""

import os
import sys
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from _full import _load_windows, build_meta, build_seqs, fold_split
from audiovj.config import (
    ENCODER_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS, N_MELS, NUM_PHRASES, PHRASE_TYPES,
)
from audiovj.model import ModelOutput, SpectrogramEncoder
from audiovj.training import (
    KEY_CLASSES, PhraseLoss, SpecAugment, _compute_class_weights,
    _get_device, _macro_f1_key_classes,
)

CKPT = "/mnt/scratch/data/loop/seq_unified_bidir.safetensors"
DROP = PHRASE_TYPES.index("drop")
LB_IDX = [PHRASE_TYPES.index(p) for p in ("intro", "buildup", "drop", "outro")]
LB = {"intro", "buildup", "drop", "outro"}
W_REG, CAP = 0.3, 12.0
WP, CW_CAP, DROPOUT, WD = 0.75, 8.0, 0.4, 3e-4  # match v2 recipe


class BiUnifiedSeq(nn.Module):
    def __init__(self, dropout=0.3, detach=False):
        super().__init__()
        self.encoder = SpectrogramEncoder(N_MELS, 128, ENCODER_CHANNELS)
        ch = self.encoder.out_channels
        self.ctx_lstm = nn.LSTM(ch, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True,
                                bidirectional=True,
                                dropout=dropout if LSTM_LAYERS > 1 else 0.0)
        hid = LSTM_HIDDEN * 2
        self.head_dropout = nn.Dropout(dropout)
        self.next_phrase_head = nn.Linear(hid, NUM_PHRASES)
        self.current_phrase_head = nn.Linear(hid, NUM_PHRASES)
        self.beats_branch = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1))
        self.detach = detach

    def forward(self, x):
        b, t = x.shape[0], x.shape[1]
        enc = self.encoder(x.reshape(b * t, x.shape[2], x.shape[3]))
        win = enc.mean(dim=1).reshape(b, t, -1)
        ctx, _ = self.ctx_lstm(win)
        h = self.head_dropout(ctx)
        beats_in = h.detach() if self.detach else h
        return ModelOutput(
            next_phrase_logits=self.next_phrase_head(h),
            current_phrase_logits=self.current_phrase_head(h),
            beats_until=self.beats_branch(beats_in))


def train(epochs=40, lr=1e-3):
    dev = _get_device()
    tr_ids, va_ids, te_ids = fold_split()
    print(f"folds: train {len(tr_ids)} val {len(va_ids)} test {len(te_ids)} (HELD OUT)", flush=True)
    tr = build_meta(tr_ids); va = build_seqs(va_ids)
    print(f"train {len(tr)} val {len(va)}  [BIDIRECTIONAL diagnostic]", flush=True)
    cw = _compute_class_weights([c for s in tr for c in s["current"].tolist()],
                                NUM_PHRASES, cap=CW_CAP, power=WP).to(dev)
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]
    model = BiUnifiedSeq(dropout=DROPOUT).to(dev); aug = SpecAugment().to(dev)
    crit = PhraseLoss(w_regression=W_REG, class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    print(f"params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    g = torch.Generator().manual_seed(0)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    best = (-1.0, 99.0)
    for ep in range(1, epochs + 1):
        model.train(); aug.train(); t0 = time.time()
        for j in torch.randperm(len(tr), generator=g).tolist():
            m = tr[j]; W = _load_windows(m).to(dev)
            o = model(aug(W).unsqueeze(0))
            beats = m["beats"].float().clamp(max=CAP).to(dev)
            loss = crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.current_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.beats_until.reshape(-1, 1),
                        m["next"].to(dev), m["current"].to(dev), beats)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); aug.eval()
        tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
        vl = 0.0; mae = []
        with torch.no_grad():
            for s in va:
                o = model(s["windows"].to(dev).unsqueeze(0))
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


def evaluate(split="test"):
    from audiovj.config import TRACKS_DIR
    from audiovj.data.rekordbox import load_tracks
    dev = _get_device()
    tr, va, te = fold_split()
    ids = te if split == "test" else va
    tracks = {t.track_id: t for t in load_tracks(TRACKS_DIR)}
    print(f"=== BIDIR diagnostic EVAL on {split} fold ({len(ids)} tracks) ===", flush=True)
    m = BiUnifiedSeq().to(dev); m.load_state_dict(load_file(CKPT)); m.eval()

    tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]
    dtp = dfp = dfn = dtn = 0
    THR = [0.4, 0.35, 0.3, 0.25]
    lb_det = {th: 0 for th in THR}; near = {th: 0 for th in THR}; fires = {th: 0 for th in THR}
    lat = {th: [] for th in THR}; lb_tot = 0
    with torch.no_grad():
        for tid in ids:
            seqs = build_seqs([tid])
            if not seqs:
                continue
            s = seqs[0]
            o = m(s["windows"].to(dev).unsqueeze(0))
            cprob = torch.softmax(o.current_phrase_logits[0], -1).cpu()
            pcur = cprob.argmax(-1).tolist()
            tcur = s["current"].tolist(); times = s["times"]; bd = 60.0 / s["bpm"]
            cues = [(c.start_time, c.phrase_type) for c in tracks[tid].cue_points]
            cp = torch.tensor(pcur); gt = s["current"]
            for c in range(NUM_PHRASES):
                pc = cp == c; gc = gt == c
                tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
            ds = [c == DROP for c in pcur]; ts = [c == DROP for c in tcur]
            for k in range(len(tcur)):
                dtp += ds[k] and ts[k]; dfp += ds[k] and not ts[k]
                dfn += (not ds[k]) and ts[k]; dtn += (not ds[k]) and (not ts[k])
            lb_bound = [ct for ct, ph in cues[1:] if ph in LB]
            lb_tot += len(lb_bound)
            for th in THR:
                ft = [times[i] for i in range(1, len(times))
                      if any(cprob[i, c] >= th and cprob[i - 1, c] < th for c in LB_IDX)]
                fires[th] += len(ft)
                for f in ft:
                    if cues and min(abs(ct - f) / bd for ct, _ in cues) <= 8:
                        near[th] += 1
                for ct in lb_bound:
                    if ft:
                        d = min(abs(f - ct) / bd for f in ft)
                        if d <= 8:
                            lb_det[th] += 1; lat[th].append(d)

    f1, _ = _macro_f1_key_classes(tp, fp, fn, key)
    rec = {KEY_CLASSES[i]: (tp[key[i]] / max(tp[key[i]] + fn[key[i]], 1)).item() for i in range(len(KEY_CLASSES))}
    acc = (dtp + dtn) / max(dtp + dfp + dfn + dtn, 1) * 100
    print("\n-- DETECTION (bidir, full-context upper bound) --")
    print(f"  load-bearing macro-F1: {f1:.3f}")
    print("  per-class recall (frame): " + "  ".join(f"{k} {v*100:.0f}%" for k, v in rec.items()))
    print(f"  in-drop frame: acc {acc:.1f}%  recall {dtp/max(dtp+dfn,1)*100:.1f}%  prec {dtp/max(dtp+dfp,1)*100:.1f}%")
    print("\n-- LB TRANSITION CUEING (onset, honest operating point) --")
    for th in THR:
        print(f"  onset@{th:<4}: LB-recall {lb_det[th]/max(lb_tot,1)*100:4.1f}%  "
              f"prec {near[th]/max(fires[th],1)*100:4.1f}%  lat {sum(lat[th])/max(len(lat[th]),1):.1f}b  fires {fires[th]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    elif cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "test")
