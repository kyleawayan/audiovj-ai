"""Retrain the seq model with anticipation-focused beats_until supervision.

Baseline countdown is broken (MAE ~30 beats, monotone ~55%) because PhraseLoss
weights regression at 0.01 and the target spans 4..96+ beats. Here we:
  - raise the regression weight (w_reg)
  - CAP the beats_until target (we only need an accurate countdown in the final
    ~8 bars; beyond that "far" is fine), so the head's capacity goes to the
    approach where the goal is measured.
Checkpoint selection keeps detection (macro-F1) from regressing while reporting
countdown MAE so we can pick a model good at BOTH.
"""

import sys

import torch
from safetensors.torch import load_file, save_file

from _arch import SeqPhrasePredictor, build_seqs
from audiovj.config import NUM_PHRASES, PHRASE_TYPES, TRACKS_DIR, FEATURES_DIR
from audiovj.data.dataset import create_splits
from audiovj.training import (
    KEY_CLASSES, PhraseLoss, SpecAugment, _compute_class_weights,
    _get_device, _macro_f1_key_classes,
)

CKPT = "/mnt/scratch/data/loop/seq_antic.safetensors"
DROP = PHRASE_TYPES.index("drop")


def train(epochs=30, w_reg=0.3, bu_cap=32.0, lr=1e-3, dropout=0.3, wd=1e-4, wp=0.5):
    dev = _get_device()
    tr_ids, va_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    tr, va = build_seqs(tr_ids), build_seqs(va_ids)
    print(f"train {len(tr)} val {len(va)}  w_reg={w_reg} bu_cap={bu_cap}")
    cw = _compute_class_weights([c for s in tr for c in s["current"].tolist()], NUM_PHRASES, cap=5.0, power=wp).to(dev)
    key_idx = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]
    model = SeqPhrasePredictor(dropout=dropout).to(dev)
    aug = SpecAugment().to(dev)
    crit = PhraseLoss(w_regression=w_reg, class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    g = torch.Generator().manual_seed(0)
    # Select the lowest countdown MAE among epochs that keep detection F1 >= floor,
    # so we improve anticipation without regressing detection.
    f1_floor = 0.58
    best_cd = 1e9; best_f1_at = -1.0

    for ep in range(1, epochs + 1):
        model.train(); aug.train()
        for j in torch.randperm(len(tr), generator=g).tolist():
            s = tr[j]
            w = aug(s["windows"].to(dev)).unsqueeze(0)
            o = model(w)
            beats = s["beats"].float().clamp(max=bu_cap).to(dev)
            loss = crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.current_phrase_logits.reshape(-1, NUM_PHRASES),
                        o.beats_until.reshape(-1, 1),
                        s["next"].to(dev), s["current"].to(dev), beats)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

        # val: macro-F1 + countdown MAE (final 2 bars)
        model.eval(); aug.eval()
        tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
        vl = 0.0; mae = []
        with torch.no_grad():
            for s in va:
                o = model(s["windows"].to(dev).unsqueeze(0))
                cl = o.current_phrase_logits.reshape(-1, NUM_PHRASES)
                beats = s["beats"].float().clamp(max=bu_cap).to(dev)
                vl += crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES), cl, o.beats_until.reshape(-1, 1),
                           s["next"].to(dev), s["current"].to(dev), beats).item()
                cp = cl.argmax(-1).cpu(); gt = s["current"]
                for c in range(NUM_PHRASES):
                    pc = cp == c; gc = gt == c
                    tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
                pbu = torch.expm1(o.beats_until[0, :, 0]).cpu()
                tbu = s["beats"]
                for k in range(len(tbu)):
                    if 0 < tbu[k] <= 8:
                        mae.append(abs(pbu[k].item() - float(tbu[k])))
        f1, per = _macro_f1_key_classes(tp, fp, fn, key_idx)
        cmae = sum(mae) / max(len(mae), 1)
        sch.step(vl / max(len(va), 1))
        tag = ""
        if f1 >= f1_floor and cmae < best_cd:
            best_cd = cmae; best_f1_at = f1; save_file(model.state_dict(), CKPT); tag = " *"
        print(f"ep {ep:3d} mF1 {f1:.3f} cd_mae(2bar) {cmae:5.1f} "
              f"[{' '.join(f'{KEY_CLASSES[i]}={per[i]:.2f}' for i in range(4))}]{tag}", flush=True)
    print(f"\nbest cd_mae {best_cd:.1f} (mF1 {best_f1_at:.3f}) -> {CKPT}")


if __name__ == "__main__":
    train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 30,
          w_reg=float(sys.argv[1]) if len(sys.argv) > 1 else 0.3,
          bu_cap=float(sys.argv[3]) if len(sys.argv) > 3 else 32.0)
