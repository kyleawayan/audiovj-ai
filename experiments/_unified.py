"""Unified model: one network good at BOTH detection and anticipation.

The anticipation retrain regressed detection (F1 0.62->0.60) because the heavy
beats_until regression distorts the shared LSTM backbone. Fix: a dedicated
beats_until branch (deeper MLP) fed from the backbone, optionally DETACHED so
the regression gradient never touches the classification features -> detection
F1 is fully protected while the branch still learns the countdown from the
(ramp-encoding) features.

  uv run python _unified.py <detach:0|1> <w_reg> <cap> [epochs]
"""

import sys

import torch
import torch.nn as nn
from safetensors.torch import save_file

from _arch import build_seqs
from audiovj.config import (
    ENCODER_CHANNELS, FEATURES_DIR, LSTM_HIDDEN, LSTM_LAYERS, N_MELS, NUM_PHRASES,
    PHRASE_TYPES, TRACKS_DIR,
)
from audiovj.data.dataset import create_splits
from audiovj.model import ModelOutput, SpectrogramEncoder
from audiovj.training import (
    KEY_CLASSES, PhraseLoss, SpecAugment, _compute_class_weights, _get_device,
    _macro_f1_key_classes,
)

CKPT = "/mnt/scratch/data/loop/seq_unified.safetensors"


class UnifiedSeq(nn.Module):
    def __init__(self, dropout=0.3, detach=True):
        super().__init__()
        self.encoder = SpectrogramEncoder(N_MELS, 128, ENCODER_CHANNELS)
        ch = self.encoder.out_channels
        self.ctx_lstm = nn.LSTM(ch, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True,
                                dropout=dropout if LSTM_LAYERS > 1 else 0.0)
        self.head_dropout = nn.Dropout(dropout)
        self.next_phrase_head = nn.Linear(LSTM_HIDDEN, NUM_PHRASES)
        self.current_phrase_head = nn.Linear(LSTM_HIDDEN, NUM_PHRASES)
        # dedicated deeper beats branch
        self.beats_branch = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(LSTM_HIDDEN, 1))
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


def train(detach, w_reg, cap, epochs=30, lr=1e-3, wp=0.5):
    dev = _get_device()
    tr_ids, va_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    tr, va = build_seqs(tr_ids), build_seqs(va_ids)
    print(f"detach={detach} w_reg={w_reg} cap={cap}  train {len(tr)} val {len(va)}")
    cw = _compute_class_weights([c for s in tr for c in s["current"].tolist()], NUM_PHRASES, cap=5.0, power=wp).to(dev)
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]
    model = UnifiedSeq(detach=detach).to(dev); aug = SpecAugment().to(dev)
    crit = PhraseLoss(w_regression=w_reg, class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    g = torch.Generator().manual_seed(0)
    best_f1 = -1; best_pick = (0, 99)  # (f1, cd) of saved
    for ep in range(1, epochs + 1):
        model.train(); aug.train()
        for j in torch.randperm(len(tr), generator=g).tolist():
            s = tr[j]; w = aug(s["windows"].to(dev)).unsqueeze(0); o = model(w)
            beats = s["beats"].float().clamp(max=cap).to(dev)
            loss = crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES), o.current_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.beats_until.reshape(-1, 1), s["next"].to(dev), s["current"].to(dev), beats)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); aug.eval()
        tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES); vl = 0; mae = []
        with torch.no_grad():
            for s in va:
                o = model(s["windows"].to(dev).unsqueeze(0)); cl = o.current_phrase_logits.reshape(-1, NUM_PHRASES)
                beats = s["beats"].float().clamp(max=cap).to(dev)
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
        f1, _ = _macro_f1_key_classes(tp, fp, fn, key); cmae = sum(mae) / max(len(mae), 1)
        sch.step(vl / max(len(va), 1))
        tag = ""
        # save the epoch maximizing F1 then (tiebreak) low cd_mae among F1>=0.58
        if f1 >= 0.58 and (f1 > best_pick[0] + 1e-6 or (abs(f1 - best_pick[0]) <= 0.01 and cmae < best_pick[1])):
            best_pick = (f1, cmae); save_file(model.state_dict(), CKPT); tag = " *"
        print(f"ep {ep:3d} mF1 {f1:.3f} cd_mae(2bar) {cmae:5.1f}{tag}", flush=True)
    print(f"\nsaved F1 {best_pick[0]:.3f} cd_mae {best_pick[1]:.1f} -> {CKPT}")


if __name__ == "__main__":
    train(detach=bool(int(sys.argv[1])) if len(sys.argv) > 1 else True,
          w_reg=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0,
          cap=float(sys.argv[3]) if len(sys.argv) > 3 else 12.0,
          epochs=int(sys.argv[4]) if len(sys.argv) > 4 else 30)
