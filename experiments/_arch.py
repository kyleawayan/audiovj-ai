"""Iteration 3: longer-context (cross-downbeat) sequence model.

The committed model encodes one 8-beat window and classifies that downbeat in
isolation (its LSTM only spans the 32 frames *within* the window). This variant
adds a second LSTM *across* downbeats so the model can see the multi-bar ramp
that defines a buildup and latch transitions earlier.

  per-window:  mel window -> SpectrogramEncoder -> mean-pool -> 128-d embedding
  cross-bar:   LSTM over the per-downbeat embeddings (whole track) -> per-step heads

Trained as sequence labeling (one track = one sequence, batch=1), reusing the
committed loss/weighting recipe. Self-contained so the committed code stays put
until this proves out. GPU via the package CUDA bootstrap.
"""

import os
import pickle
import sys
import time

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from audiovj.config import (
    ENCODER_CHANNELS, FEATURES_DIR, LSTM_HIDDEN, LSTM_LAYERS,
    N_MELS, NUM_PHRASES, PHRASE_TYPES, TRACKS_DIR,
)
from audiovj.data.dataset import generate_labels, create_splits
from audiovj.data.rekordbox import build_downbeat_times, load_tracks
from audiovj.live.inference import PredictionResult
from audiovj.model import ModelOutput, SpectrogramEncoder
from audiovj.training import (
    KEY_CLASSES, PhraseLoss, SpecAugment, _compute_class_weights,
    _get_device, _macro_f1_key_classes,
)

CKPT = "/mnt/scratch/data/loop/seq_predictor.safetensors"
SEQ_CACHE = "/mnt/scratch/data/loop/seq_pred_cache.pkl"
P2I = {p: i for i, p in enumerate(PHRASE_TYPES)}


class SeqPhrasePredictor(nn.Module):
    def __init__(self, dropout: float = 0.3, num_phrases: int = NUM_PHRASES) -> None:
        super().__init__()
        self.encoder = SpectrogramEncoder(N_MELS, 128, ENCODER_CHANNELS)
        ch = self.encoder.out_channels
        self.ctx_lstm = nn.LSTM(
            ch, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True,
            dropout=dropout if LSTM_LAYERS > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.next_phrase_head = nn.Linear(LSTM_HIDDEN, num_phrases)
        self.current_phrase_head = nn.Linear(LSTM_HIDDEN, num_phrases)
        self.beats_until_head = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x: [B, T, n_mels, frames]
        b, t = x.shape[0], x.shape[1]
        enc = self.encoder(x.reshape(b * t, x.shape[2], x.shape[3]))  # [B*T, 32, ch]
        win = enc.mean(dim=1).reshape(b, t, -1)                       # [B, T, ch]
        ctx, _ = self.ctx_lstm(win)                                  # [B, T, hidden]
        h = self.head_dropout(ctx)
        return ModelOutput(
            next_phrase_logits=self.next_phrase_head(h),
            current_phrase_logits=self.current_phrase_head(h),
            beats_until=self.beats_until_head(h),
        )


def build_seqs(ids):
    """Per-track contiguous labeled sequences: windows + per-step labels + meta."""
    seqs = []
    for tid in ids:
        tp = TRACKS_DIR / f"{tid}.json"
        fp = FEATURES_DIR / f"{tid}.safetensors"
        if not tp.exists() or not fp.exists():
            continue
        from audiovj.data.rekordbox import Track
        track = Track.model_validate_json(tp.read_text())
        if not track.cue_points:
            continue
        data = load_file(str(fp))
        windows = data["windows"]
        kept = data["kept_indices"].tolist()
        downbeats = build_downbeat_times(track)
        labels = generate_labels(track, downbeats)
        if not labels:
            continue
        wlist, cur, nxt, beats, times = [], [], [], [], []
        for i, db in enumerate(kept):
            if i >= windows.shape[0] or db >= len(labels):
                break
            lbl = labels[db]
            if lbl is None:
                continue
            wlist.append(windows[i])
            cur.append(P2I[lbl["current_phrase"]])
            nxt.append(P2I[lbl["next_phrase"]])
            beats.append(float(lbl["beats_until"]))
            times.append(downbeats[db])
        if len(wlist) < 2:
            continue
        seqs.append({
            "windows": torch.stack(wlist),                # [T, n_mels, frames]
            "current": torch.tensor(cur),
            "next": torch.tensor(nxt),
            "beats": torch.tensor(beats),
            "times": times,
            "cue_times": [c.start_time for c in track.cue_points],
            "actual_transitions": max(len(track.cue_points) - 1, 0),
            "bpm": track.bpm,
        })
    return seqs


def train(epochs=30, lr=1e-3, dropout=0.3, weight_decay=1e-4, grad_clip=1.0, weight_power=0.5):
    device = _get_device()
    train_ids, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    print(f"train tracks {len(train_ids)}  val tracks {len(val_ids)}")
    train_seqs = build_seqs(train_ids)
    val_seqs = build_seqs(val_ids)
    print(f"train seqs {len(train_seqs)}  val seqs {len(val_seqs)}  "
          f"train downbeats {sum(len(s['current']) for s in train_seqs)}")

    all_current = [c for s in train_seqs for c in s["current"].tolist()]
    cw = _compute_class_weights(all_current, NUM_PHRASES, cap=5.0, power=weight_power).to(device)
    key_idx = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]

    model = SeqPhrasePredictor(dropout=dropout).to(device)
    aug = SpecAugment().to(device)
    crit = PhraseLoss(class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    print(f"params {sum(p.numel() for p in model.parameters()):,}  weight_power {weight_power}")

    g = torch.Generator().manual_seed(0)
    best_f1 = -1.0
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train(); aug.train()
        order = torch.randperm(len(train_seqs), generator=g).tolist()
        tl, t0 = 0.0, time.time()
        for j in order:
            s = train_seqs[j]
            w = aug(s["windows"].to(device)).unsqueeze(0)  # [1, T, mel, fr]
            out = model(w)
            loss = crit(
                out.next_phrase_logits.reshape(-1, NUM_PHRASES),
                out.current_phrase_logits.reshape(-1, NUM_PHRASES),
                out.beats_until.reshape(-1, 1),
                s["next"].to(device), s["current"].to(device), s["beats"].float().to(device),
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); tl += loss.item()

        # val
        model.eval(); aug.eval()
        tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
        vl, cn, cc, tot = 0.0, 0, 0, 0
        with torch.no_grad():
            for s in val_seqs:
                w = s["windows"].to(device).unsqueeze(0)
                out = model(w)
                nl = out.next_phrase_logits.reshape(-1, NUM_PHRASES)
                cl = out.current_phrase_logits.reshape(-1, NUM_PHRASES)
                vl += crit(nl, cl, out.beats_until.reshape(-1, 1),
                           s["next"].to(device), s["current"].to(device), s["beats"].float().to(device)).item()
                cp = cl.argmax(-1).cpu(); gt = s["current"]
                cc += (cp == gt).sum().item(); tot += len(gt)
                cn += (nl.argmax(-1).cpu() == s["next"]).sum().item()
                for c in range(NUM_PHRASES):
                    pc = cp == c; gc = gt == c
                    tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
        f1, per = _macro_f1_key_classes(tp, fp, fn, key_idx)
        sched.step(vl / max(len(val_seqs), 1))
        brk = " ".join(f"{KEY_CLASSES[i]}={per[i]:.2f}" for i in range(len(KEY_CLASSES)))
        tag = ""
        if f1 > best_f1:
            best_f1 = f1; save_file(model.state_dict(), CKPT); tag = " *saved*"
        print(f"ep {ep:3d}/{epochs} train {tl/max(len(train_seqs),1):.3f} val {vl/max(len(val_seqs),1):.3f} "
              f"cur_acc {cc/max(tot,1)*100:.1f}% macro_f1 {f1:.3f} [{brk}]{tag}", flush=True)
    print(f"\nbest macro_f1 {best_f1:.3f}  -> {CKPT}")
    return best_f1


def build_cache():
    device = _get_device()
    model = SeqPhrasePredictor().to(device)
    model.load_state_dict(load_file(CKPT)); model.eval()
    # cache over ALL tracks that have features (same set as _loop) for SM eval parity
    ids = [t.track_id for t in load_tracks(TRACKS_DIR)
           if t.cue_points and (FEATURES_DIR / f"{t.track_id}.safetensors").exists()]
    seqs = build_seqs(ids)
    cache = []
    with torch.no_grad():
        for s in seqs:
            out = model(s["windows"].to(device).unsqueeze(0))
            npb = torch.softmax(out.next_phrase_logits[0], -1).cpu()
            cpb = torch.softmax(out.current_phrase_logits[0], -1).cpu()
            bu = torch.expm1(out.beats_until[0, :, 0]).cpu()
            rows = []
            for i in range(len(s["times"])):
                ci = int(cpb[i].argmax()); ni = int(npb[i].argmax())
                pred = PredictionResult(PHRASE_TYPES[ci], float(cpb[i, ci]),
                                        PHRASE_TYPES[ni], float(npb[i, ni]), float(bu[i]))
                lbl = {"current_phrase": PHRASE_TYPES[int(s["current"][i])],
                       "beats_until": float(s["beats"][i])}
                rows.append((s["times"][i], lbl, pred))
            cache.append({"bpm": s["bpm"], "cue_times": s["cue_times"],
                          "actual_transitions": s["actual_transitions"], "rows": rows})
    with open(SEQ_CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == "cache":
        c = build_cache()
        print(f"seq cache: {len(c)} tracks, {sum(len(t['rows']) for t in c)} downbeats")
