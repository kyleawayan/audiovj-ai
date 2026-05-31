"""Training loop: loss function, SpecAugment, and train_model entry point."""

import time

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader, WeightedRandomSampler

from audiovj.config import FEATURES_DIR, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import PhraseDataset, create_splits
from audiovj.model import PhrasePredictor

KEY_CLASSES = ["intro", "buildup", "drop", "outro"]


def _collate_variable_width(
    batch: list[tuple[torch.Tensor, int, int, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-width mel windows to a size divisible by FIXED_FRAMES.

    MPS requires AdaptiveAvgPool1d input to be divisible by the output size.
    """
    from audiovj.config import FIXED_FRAMES

    windows, current, next_, beats = zip(*batch)
    max_frames = max(w.shape[-1] for w in windows)
    # Round up to nearest multiple of FIXED_FRAMES for MPS compatibility
    max_frames = ((max_frames + FIXED_FRAMES - 1) // FIXED_FRAMES) * FIXED_FRAMES
    padded = torch.zeros(len(windows), windows[0].shape[0], max_frames)
    for i, w in enumerate(windows):
        padded[i, :, : w.shape[-1]] = w
    return (
        padded,
        torch.tensor(current),
        torch.tensor(next_),
        torch.tensor(beats),
    )


class SpecAugment(nn.Module):
    """SpecAugment data augmentation: time and frequency masking."""

    def __init__(self, time_mask_pct: float = 0.2, freq_mask_pct: float = 0.2) -> None:
        super().__init__()
        self.time_mask_pct = time_mask_pct
        self.freq_mask_pct = freq_mask_pct

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [batch, n_mels, frames]. Masks in-place."""
        if not self.training:
            return x

        _, n_mels, frames = x.shape

        # Time masking
        max_time = int(frames * self.time_mask_pct)
        if max_time > 0:
            t_len = torch.randint(1, max_time + 1, (1,)).item()
            t_start = torch.randint(0, frames - t_len + 1, (1,)).item()
            x[:, :, t_start : t_start + t_len] = 0

        # Frequency masking
        max_freq = int(n_mels * self.freq_mask_pct)
        if max_freq > 0:
            f_len = torch.randint(1, max_freq + 1, (1,)).item()
            f_start = torch.randint(0, n_mels - f_len + 1, (1,)).item()
            x[:, f_start : f_start + f_len, :] = 0

        return x


class PhraseLoss(nn.Module):
    """Combined loss: CE(next) + CE(current) + w_reg*Huber(beats_until on transitions) + w_con*consistency."""

    def __init__(
        self,
        w_regression: float = 0.01,
        w_consistency: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.huber = nn.SmoothL1Loss()
        self.w_regression = w_regression
        self.w_consistency = w_consistency

    def forward(
        self,
        next_logits: torch.Tensor,
        current_logits: torch.Tensor,
        beats_until_pred: torch.Tensor,
        next_target: torch.Tensor,
        current_target: torch.Tensor,
        beats_until_target: torch.Tensor,
    ) -> torch.Tensor:
        loss_next = self.ce(next_logits, next_target)
        loss_current = self.ce(current_logits, current_target)

        # Masked regression: only learn beats_until from samples that are actually
        # near a transition (next != current). Otherwise the head learns to predict
        # the long-tail "next phrase is far away" value and the backbone gets corrupted.
        transition_mask = next_target != current_target
        if transition_mask.any():
            log_target = torch.log1p(beats_until_target[transition_mask])
            loss_beats = self.huber(
                beats_until_pred.squeeze(-1)[transition_mask], log_target
            )
        else:
            loss_beats = torch.tensor(0.0, device=next_logits.device)

        # Consistency penalty: penalize when next_phrase prediction flips
        # but current_phrase stays the same between consecutive samples in batch.
        consistency = torch.tensor(0.0, device=next_logits.device)
        if next_logits.shape[0] > 1:
            next_pred = next_logits.argmax(dim=-1)
            current_pred = current_logits.argmax(dim=-1)
            next_flipped = (next_pred[1:] != next_pred[:-1]).float()
            current_same = (current_pred[1:] == current_pred[:-1]).float()
            consistency = (next_flipped * current_same).mean()

        return (
            loss_next
            + loss_current
            + self.w_regression * loss_beats
            + self.w_consistency * consistency
        )


def _get_device() -> torch.device:
    """Select best available device, logging explicitly."""
    if torch.cuda.is_available():
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using device: MPS (Apple Silicon)")
        return torch.device("mps")
    print("WARNING: no GPU available, falling back to CPU")
    return torch.device("cpu")


def _compute_class_weights(
    labels: list[int], num_classes: int, cap: float, power: float = 1.0
) -> torch.Tensor:
    """Frequency-based class weights, capped at `cap`.

    power=1.0 -> full inverse-frequency (w_c = N/(K*n_c)); power=0.5 -> sqrt
    scheme, which lifts minorities without crushing the majority; power=0 ->
    uniform. Classes absent from `labels` get weight = cap.
    """
    counts = torch.zeros(num_classes)
    for c in labels:
        counts[c] += 1
    total = counts.sum().item()
    # Inverse freq raised to `power`. Avoid div-by-zero.
    inv = total / (counts.clamp(min=1.0) * num_classes)
    weights = inv ** power
    # Classes with zero examples get cap (penalize heavily, even though gradient
    # won't flow if they're never in a batch).
    weights = torch.where(counts > 0, weights, torch.full_like(weights, cap))
    weights = weights.clamp(max=cap)
    return weights


def _macro_f1_key_classes(
    tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, key_indices: list[int]
) -> tuple[float, list[float]]:
    """Macro-F1 averaged across the load-bearing classes.

    Returns (macro_f1, per_key_class_f1_list).
    """
    f1s = []
    for c in key_indices:
        tp_c = tp[c].item()
        fp_c = fp[c].item()
        fn_c = fn[c].item()
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s), f1s


def train_model(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    lr_patience: int = 5,
    lr_factor: float = 0.5,
    class_weight_cap: float = 5.0,
    weight_power: float = 1.0,
    f1_save_threshold: float = 0.0,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    log_interval: int = 200,
    balance: str = "sampler",
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
) -> None:
    """Full training loop: split data, train, save best checkpoint by macro-F1.

    Data loading is parallelized (num_workers) and overlapped with GPU compute
    (pin_memory + prefetch): the model is tiny, so wall-clock is dominated by the
    data pipeline, not GPU compute. In-epoch progress logs every log_interval
    batches. f1_save_threshold=0 always keeps the best checkpoint seen.
    """
    device = _get_device()

    # Data splits
    train_ids, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    print(f"Train tracks: {len(train_ids)}, Val tracks: {len(val_ids)}")

    train_ds = PhraseDataset(train_ids, TRACKS_DIR, FEATURES_DIR)
    print(f"Train samples: {len(train_ds)}")

    if not train_ds:
        print("Error: No training samples. Run import-rekordbox and preprocess first.")
        return

    val_ds = PhraseDataset(val_ids, TRACKS_DIR, FEATURES_DIR) if val_ids else None
    if val_ds:
        print(f"Val samples: {len(val_ds)}")

    num_phrases = len(PHRASE_TYPES)
    key_indices = [PHRASE_TYPES.index(p) for p in KEY_CLASSES]

    # Class weights from training-set current_phrase distribution
    class_weights = _compute_class_weights(
        train_ds._current_phrase, num_phrases, cap=class_weight_cap, power=weight_power
    )
    print(f"Class weights (cap={class_weight_cap}, power={weight_power}):")
    for i, p in enumerate(PHRASE_TYPES):
        count = sum(1 for c in train_ds._current_phrase if c == i)
        print(f"  {p:<10} count={count:>6}  weight={class_weights[i].item():.3f}")

    # Parallel + overlapped loading. The feature set (mmap'd safetensors) is far
    # larger than RAM, so workers prefetch/fault the next batch while the GPU runs.
    loader_kwargs: dict = {"collate_fn": _collate_variable_width}
    if num_workers > 0:
        loader_kwargs.update(
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
        )

    # Single balancing mechanism. Applying BOTH a weighted sampler AND weighted
    # loss double-corrects: it suppresses the majority class (drop) in both the
    # data distribution and the gradient, collapsing its F1. Pick exactly one.
    #   sampler -> WeightedRandomSampler + unweighted CE
    #   loss    -> natural distribution + class-weighted CE
    #   none    -> natural distribution + unweighted CE
    loss_class_weights = None
    if balance == "sampler":
        sample_weights = [class_weights[c].item() for c in train_ds._current_phrase]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs
        )
    elif balance == "loss":
        loss_class_weights = class_weights.to(device)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
        )
    else:  # "none"
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
        )
    print(f"Balancing: {balance}  |  dropout={dropout}  weight_decay={weight_decay}")

    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, **loader_kwargs) if val_ds else None
    )

    # Model, loss, optimizer, scheduler
    model = PhrasePredictor(dropout=dropout).to(device)
    augment = SpecAugment().to(device)
    criterion = PhraseLoss(class_weights=loss_class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Phrase classes: {num_phrases} {PHRASE_TYPES}")
    print(f"F1 ckpt selection: macro-F1 over {KEY_CLASSES} (threshold={f1_save_threshold})")
    print()

    best_macro_f1 = -1.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "phrase_predictor.safetensors"

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        augment.train()
        train_loss = 0.0
        train_batches = 0
        total_batches = len(train_loader)
        samples_seen = 0
        epoch_t0 = time.time()

        for windows, current_idx, next_idx, beats_until in train_loader:
            windows = augment(windows.to(device, non_blocking=True))
            current_idx = current_idx.to(device, non_blocking=True)
            next_idx = next_idx.to(device, non_blocking=True)
            beats_until = beats_until.float().to(device, non_blocking=True)

            out = model(windows)
            loss = criterion(
                out.next_phrase_logits,
                out.current_phrase_logits,
                out.beats_until,
                next_idx,
                current_idx,
                beats_until,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            samples_seen += windows.shape[0]

            if log_interval and train_batches % log_interval == 0:
                elapsed = time.time() - epoch_t0
                sps = samples_seen / max(elapsed, 1e-6)
                eta = elapsed / max(train_batches, 1) * (total_batches - train_batches)
                print(
                    f"  epoch {epoch:3d} [{train_batches:>5d}/{total_batches}] "
                    f"loss={train_loss / train_batches:.3f}  "
                    f"{sps:,.0f} samp/s  ETA {eta:4.0f}s",
                    flush=True,
                )

        avg_train = train_loss / max(train_batches, 1)

        # --- Validate ---
        val_msg = ""
        if val_loader is not None:
            model.eval()
            augment.eval()
            val_loss = 0.0
            val_batches = 0
            correct_next = 0
            correct_current = 0
            total = 0

            tp = torch.zeros(num_phrases)
            fp = torch.zeros(num_phrases)
            fn = torch.zeros(num_phrases)

            with torch.no_grad():
                for windows, current_idx, next_idx, beats_until in val_loader:
                    windows = windows.to(device)
                    current_idx = current_idx.to(device)
                    next_idx = next_idx.to(device)
                    beats_until = beats_until.float().to(device)

                    out = model(windows)
                    loss = criterion(
                        out.next_phrase_logits,
                        out.current_phrase_logits,
                        out.beats_until,
                        next_idx,
                        current_idx,
                        beats_until,
                    )
                    val_loss += loss.item()
                    val_batches += 1

                    current_pred = out.current_phrase_logits.argmax(-1)
                    correct_next += (
                        out.next_phrase_logits.argmax(-1) == next_idx
                    ).sum().item()
                    correct_current += (current_pred == current_idx).sum().item()
                    total += windows.shape[0]

                    # Per-class tp/fp/fn for current_phrase (on CPU to accumulate)
                    cp_cpu = current_pred.cpu()
                    gt_cpu = current_idx.cpu()
                    for c in range(num_phrases):
                        pred_c = cp_cpu == c
                        gt_c = gt_cpu == c
                        tp[c] += (pred_c & gt_c).sum()
                        fp[c] += (pred_c & ~gt_c).sum()
                        fn[c] += (~pred_c & gt_c).sum()

            avg_val = val_loss / max(val_batches, 1)
            acc_next = correct_next / max(total, 1) * 100
            acc_current = correct_current / max(total, 1) * 100
            macro_f1, per_key_f1 = _macro_f1_key_classes(tp, fp, fn, key_indices)

            scheduler.step(avg_val)
            lr_now = optimizer.param_groups[0]["lr"]

            f1_breakdown = " ".join(
                f"{KEY_CLASSES[i]}={per_key_f1[i]:.2f}" for i in range(len(KEY_CLASSES))
            )
            val_msg = (
                f"  val_loss={avg_val:.4f}  "
                f"next_acc={acc_next:.1f}%  current_acc={acc_current:.1f}%  "
                f"macro_f1={macro_f1:.3f}  [{f1_breakdown}]  lr={lr_now:.1e}"
            )

            # Save best by macro-F1 (with floor to avoid early-epoch noise)
            if macro_f1 > best_macro_f1 and macro_f1 >= f1_save_threshold:
                best_macro_f1 = macro_f1
                save_file(model.state_dict(), str(checkpoint_path))
                val_msg += "  *saved*"
        else:
            # No val set — save every epoch
            save_file(model.state_dict(), str(checkpoint_path))

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}{val_msg}", flush=True)

    print(f"\nTraining complete. Best checkpoint: {checkpoint_path}  best_macro_f1={best_macro_f1:.3f}")
