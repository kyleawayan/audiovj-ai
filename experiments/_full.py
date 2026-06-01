"""Full-scale (1,391-track) unified-model training with a CLEAN fold split.

This is the run FINDINGS flagged as the real unlock: all 1,391 tracks are now
present locally (audio + pre-computed features + labels), so we retrain the
production-candidate UnifiedSeq from scratch at full scale and certify on a
held-out fold the model never saw.

Why this fixes the core caveat: the 150-subset's val split overlapped the seed
checkpoint's training data. Raveform ships an official 8-fold assignment; we use
  train = folds 0..5   (~1043 tracks)
  val   = fold 6        (~174, early-stop / model selection)
  test  = fold 7        (~174, HELD OUT — touched once, at the very end)

RAM-safe: 47GB of features can't fit in 31GB RAM. The seq model trains batch=1
(one track = one sequence), so windows are lazy-loaded per-track in the loop and
freed immediately; only tiny per-track labels/meta are held. Eval folds (~174
tracks, ~6.6GB) still fit eagerly via the committed build_seqs.

  uv run python experiments/_full.py train [epochs]
  uv run python experiments/_full.py eval  <val|test>
"""

import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from _arch import build_seqs
from _drop import debounce_state, score_edges, state_edges
from _unified import UnifiedSeq
from audiovj.config import (
    FEATURES_DIR, NUM_PHRASES, PHRASE_TYPES, TRACKS_DIR,
)
from audiovj.data.dataset import generate_labels
from audiovj.data.rekordbox import Track, build_downbeat_times, load_tracks
from audiovj.training import (
    KEY_CLASSES, PhraseLoss, SpecAugment, _compute_class_weights,
    _get_device, _macro_f1_key_classes,
)

CKPT = "/mnt/scratch/data/loop/seq_unified_full.safetensors"
P2I = {p: i for i, p in enumerate(PHRASE_TYPES)}
DROP = PHRASE_TYPES.index("drop")
BUILD = PHRASE_TYPES.index("buildup")

# Production-candidate config (FINDINGS winner): coupled (non-detached) beats
# branch, light regression weight, capped countdown target.
DETACH = False
W_REG = 0.3
CAP = 12.0


def fold_split() -> tuple[list[str], list[str], list[str]]:
    """Clean split by Raveform's official fold field. test=7 is held out."""
    tr, va, te = [], [], []
    for t in load_tracks(TRACKS_DIR):
        if not t.cue_points:
            continue
        if not (FEATURES_DIR / f"{t.track_id}.safetensors").exists():
            continue
        if t.fold is None:
            continue
        bucket = te if t.fold == 7 else va if t.fold == 6 else tr
        bucket.append(t.track_id)
    return tr, va, te


def build_meta(ids: list[str]) -> list[dict]:
    """Light per-track metadata (NO window tensors) for RAM-safe lazy training.

    Mirrors _arch.build_seqs' keep/label logic exactly, but stores only the
    positions to slice out of the windows tensor at load time + the labels.
    Reads just the safetensors header (kept_indices + windows shape), not the
    38MB window payload.
    """
    meta = []
    for tid in ids:
        tp = TRACKS_DIR / f"{tid}.json"
        fp = FEATURES_DIR / f"{tid}.safetensors"
        if not tp.exists() or not fp.exists():
            continue
        track = Track.model_validate_json(tp.read_text())
        if not track.cue_points:
            continue
        with safe_open(str(fp), framework="pt") as f:
            kept = f.get_tensor("kept_indices").tolist()
            nwin = f.get_slice("windows").get_shape()[0]
        downbeats = build_downbeat_times(track)
        labels = generate_labels(track, downbeats)
        if not labels:
            continue
        keep_pos, cur, nxt, beats = [], [], [], []
        for i, db in enumerate(kept):
            if i >= nwin or db >= len(labels):
                break
            lbl = labels[db]
            if lbl is None:
                continue
            keep_pos.append(i)
            cur.append(P2I[lbl["current_phrase"]])
            nxt.append(P2I[lbl["next_phrase"]])
            beats.append(float(lbl["beats_until"]))
        if len(keep_pos) < 2:
            continue
        meta.append({
            "fp": str(fp),
            "keep": torch.tensor(keep_pos),
            "current": torch.tensor(cur),
            "next": torch.tensor(nxt),
            "beats": torch.tensor(beats),
        })
    return meta


def _load_windows(m: dict) -> torch.Tensor:
    """Lazy: load one track's windows from NVMe and slice the kept rows."""
    return load_file(m["fp"])["windows"][m["keep"]]


def train(epochs=30, lr=1e-3, wp=0.5, dropout=0.3, w_reg=W_REG, cap=CAP,
          cw_cap=5.0, weight_decay=1e-4, ckpt=CKPT):
    dev = _get_device()
    tr_ids, va_ids, te_ids = fold_split()
    print(f"folds: train {len(tr_ids)}  val {len(va_ids)}  test {len(te_ids)} (HELD OUT)",
          flush=True)
    tr = build_meta(tr_ids)            # lazy meta (RAM-safe)
    va = build_seqs(va_ids)            # eager OK (~174 tracks)
    print(f"train seqs {len(tr)}  val seqs {len(va)}  "
          f"train downbeats {sum(len(s['current']) for s in tr)}", flush=True)

    cw = _compute_class_weights(
        [c for s in tr for c in s["current"].tolist()], NUM_PHRASES, cap=cw_cap, power=wp
    ).to(dev)
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]

    model = UnifiedSeq(dropout=dropout, detach=DETACH).to(dev)
    aug = SpecAugment().to(dev)
    crit = PhraseLoss(w_regression=w_reg, class_weights=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    print(f"params {sum(p.numel() for p in model.parameters()):,}  "
          f"detach={DETACH} w_reg={W_REG} cap={CAP}", flush=True)

    g = torch.Generator().manual_seed(0)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    best = (-1.0, 99.0)  # (mF1, cd_mae) of saved checkpoint
    for ep in range(1, epochs + 1):
        model.train(); aug.train()
        t0 = time.time(); tl = 0.0
        for j in torch.randperm(len(tr), generator=g).tolist():
            m = tr[j]
            W = _load_windows(m).to(dev)
            w = aug(W).unsqueeze(0)
            o = model(w)
            beats = m["beats"].float().clamp(max=cap).to(dev)
            loss = crit(
                o.next_phrase_logits.reshape(-1, NUM_PHRASES),
                o.current_phrase_logits.reshape(-1, NUM_PHRASES),
                o.beats_until.reshape(-1, 1),
                m["next"].to(dev), m["current"].to(dev), beats,
            )
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item()

        # validation (eager seqs)
        model.eval(); aug.eval()
        tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
        vl = 0.0; mae = []
        with torch.no_grad():
            for s in va:
                o = model(s["windows"].to(dev).unsqueeze(0))
                cl = o.current_phrase_logits.reshape(-1, NUM_PHRASES)
                beats = s["beats"].float().clamp(max=cap).to(dev)
                vl += crit(o.next_phrase_logits.reshape(-1, NUM_PHRASES), cl,
                           o.beats_until.reshape(-1, 1),
                           s["next"].to(dev), s["current"].to(dev), beats).item()
                cp = cl.argmax(-1).cpu(); gt = s["current"]
                for c in range(NUM_PHRASES):
                    pc = cp == c; gc = gt == c
                    tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()
                pbu = torch.expm1(o.beats_until[0, :, 0]).cpu(); tb = s["beats"]
                for kk in range(len(tb)):
                    if 0 < tb[kk] <= 8:
                        mae.append(abs(pbu[kk].item() - float(tb[kk])))
        f1, per = _macro_f1_key_classes(tp, fp, fn, key)
        cmae = sum(mae) / max(len(mae), 1)
        sch.step(vl / max(len(va), 1))
        tag = ""
        if f1 >= 0.55 and (f1 > best[0] + 1e-6 or (abs(f1 - best[0]) <= 0.01 and cmae < best[1])):
            best = (f1, cmae); save_file(model.state_dict(), ckpt); tag = " *"
        brk = " ".join(f"{KEY_CLASSES[i]}={per[i]:.2f}" for i in range(len(KEY_CLASSES)))
        print(f"ep {ep:3d}/{epochs} {time.time()-t0:5.0f}s "
              f"train {tl/max(len(tr),1):.3f} val {vl/max(len(va),1):.3f} "
              f"mF1 {f1:.3f} cd_mae {cmae:4.1f} [{brk}]{tag}", flush=True)
    print(f"\nsaved best mF1 {best[0]:.3f} cd_mae {best[1]:.1f} -> {ckpt}", flush=True)
    return best


def evaluate(split="test", ckpt=CKPT):
    """Headline goal metrics on a held-out fold (val or test)."""
    dev = _get_device()
    tr, va, te = fold_split()
    ids = te if split == "test" else va
    print(f"=== EVAL on {split} fold ({len(ids)} tracks)  ckpt={ckpt.split('/')[-1]} ===", flush=True)
    m = UnifiedSeq(detach=DETACH).to(dev)
    m.load_state_dict(load_file(ckpt)); m.eval()

    # detection (frame-level + drop-start edges)
    tp = torch.zeros(NUM_PHRASES); fp = torch.zeros(NUM_PHRASES); fn = torch.zeros(NUM_PHRASES)
    dtp = dfp = dfn = dtn = 0
    st = [0, 0, [], 0, 0]  # drop-start: detected, total, latencies, false_fires, fires
    warn_det = warn_tot = 0
    mae = []; mono_ok = mono_tot = 0
    key = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]

    with torch.no_grad():
        for tid in ids:
            seqs = build_seqs([tid])
            if not seqs:
                continue
            s = seqs[0]; times = s["times"]; bd = 60.0 / s["bpm"]
            o = m(s["windows"].to(dev).unsqueeze(0))
            cl = o.current_phrase_logits[0]
            pcur = cl.argmax(-1).cpu().tolist()
            pbu = torch.expm1(o.beats_until[0, :, 0]).cpu().tolist()
            nconf, nidx = torch.softmax(o.next_phrase_logits[0], -1).max(-1)
            pnext = nidx.cpu().tolist(); pconf = nconf.cpu().tolist()
            tcur = s["current"].tolist(); tnext = s["next"].tolist(); tbu = s["beats"].tolist()

            cp = torch.tensor(pcur); gt = s["current"]
            for c in range(NUM_PHRASES):
                pc = cp == c; gc = gt == c
                tp[c] += (pc & gc).sum(); fp[c] += (pc & ~gc).sum(); fn[c] += (~pc & gc).sum()

            ds = debounce_state(pcur, DROP)
            ts = [c == DROP for c in tcur]
            for k in range(len(tcur)):
                dtp += ds[k] and ts[k]; dfp += ds[k] and not ts[k]
                dfn += (not ds[k]) and ts[k]; dtn += (not ds[k]) and (not ts[k])
            tss, _ = state_edges(ts); pss, _ = state_edges(ds)
            d, t, lat, nf, f = score_edges(tss, pss, times, bd)
            st[0] += d; st[1] += t; st[2] += lat; st[3] += nf; st[4] += f

            # clamped, monotone-by-construction countdown + pre-drop warning
            cd = None
            for k in range(len(times)):
                hot = pnext[k] == DROP and pconf[k] > 0.5
                r4 = max(round(pbu[k] / 4) * 4, 0)
                prev = cd
                if cd is None:
                    cd = max(r4, 4) if hot else None
                else:
                    cd = min(cd - 4, r4) if hot else cd - 4
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

    f1, per = _macro_f1_key_classes(tp, fp, fn, key)
    rec = {KEY_CLASSES[i]: (tp[key[i]] / max(tp[key[i]] + fn[key[i]], 1)).item() for i in range(len(KEY_CLASSES))}
    acc = (dtp + dtn) / max(dtp + dfp + dfn + dtn, 1) * 100
    print("\n-- DETECTION (goal 1) --")
    print(f"  load-bearing macro-F1: {f1:.3f}")
    print("  per-class recall (frame): " + "  ".join(f"{k} {v*100:.0f}%" for k, v in rec.items()))
    print(f"  drop on/off: acc {acc:.1f}%  recall {dtp/max(dtp+dfn,1)*100:.1f}%  precision {dtp/max(dtp+dfp,1)*100:.1f}%")
    print(f"  drop START : recall {st[0]/max(st[1],1)*100:.1f}%  "
          f"latency {sum(st[2])/max(len(st[2]),1):.1f}b  precision {st[3]/max(st[4],1)*100:.1f}%")
    print("\n-- ANTICIPATION (goal 2) --")
    print(f"  pre-drop warning (>=1 bar early): recall {warn_det/max(warn_tot,1)*100:.1f}%  (n={warn_tot})")
    print(f"  countdown MAE (final 2 bars): {sum(mae)/max(len(mae),1):.2f}b  (n={len(mae)})")
    print(f"  countdown monotonic (final 2 bars): {mono_ok/max(mono_tot,1)*100:.1f}%  (n={mono_tot})")


def _ckpt_for(tag):
    if not tag:
        return CKPT
    if tag.endswith(".safetensors"):
        return tag
    return CKPT.replace(".safetensors", f"_{tag}.safetensors")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif cmd == "train_v2":
        # rare-class reweight + anti-overfit regularization (attack buildup/outro recall)
        train(epochs=int(sys.argv[2]) if len(sys.argv) > 2 else 40,
              wp=0.75, cw_cap=8.0, dropout=0.4, weight_decay=3e-4,
              ckpt=_ckpt_for("v2"))
    elif cmd == "eval":
        evaluate(sys.argv[2] if len(sys.argv) > 2 else "test",
                 ckpt=_ckpt_for(sys.argv[3] if len(sys.argv) > 3 else ""))
