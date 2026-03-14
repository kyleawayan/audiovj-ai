"""Training loop: loss function, SpecAugment, and train_model entry point."""

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from audiovj.config import FEATURES_DIR, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import PhraseDataset, create_splits
from audiovj.model import PhrasePredictor


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
    """Combined loss: CE(next) + CE(current) + w_reg*MSE(beats_until) + w_con*consistency."""

    def __init__(
        self, w_regression: float = 0.01, w_consistency: float = 0.5
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
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
        # Log-scale targets to compress range (4-64 -> ~1.6-4.2),
        # keeping MSE magnitude comparable to CE (~1.6).
        log_target = torch.log1p(beats_until_target)
        loss_beats = self.mse(beats_until_pred.squeeze(-1), log_target)

        # Consistency penalty: penalize when next_phrase prediction flips
        # but current_phrase stays the same between consecutive samples in batch.
        # Only meaningful when batch is from the same sequence (sorted by time).
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
    if torch.backends.mps.is_available():
        print("Using device: MPS (Apple Silicon)")
        return torch.device("mps")
    print("WARNING: MPS not available, falling back to CPU")
    return torch.device("cpu")


def train_model(
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> None:
    """Full training loop: split data, train, save best checkpoint."""
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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_variable_width
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, collate_fn=_collate_variable_width)
        if val_ds
        else None
    )

    # Model, loss, optimizer
    model = PhrasePredictor().to(device)
    augment = SpecAugment().to(device)
    criterion = PhraseLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_phrases = len(PHRASE_TYPES)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Phrase classes: {num_phrases} {PHRASE_TYPES}")
    print()

    best_val_loss = float("inf")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / "phrase_predictor.safetensors"

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        augment.train()
        train_loss = 0.0
        train_batches = 0

        for windows, current_idx, next_idx, beats_until in train_loader:
            windows = augment(windows.to(device))
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

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

                    correct_next += (
                        out.next_phrase_logits.argmax(-1) == next_idx
                    ).sum().item()
                    correct_current += (
                        out.current_phrase_logits.argmax(-1) == current_idx
                    ).sum().item()
                    total += windows.shape[0]

            avg_val = val_loss / max(val_batches, 1)
            acc_next = correct_next / max(total, 1) * 100
            acc_current = correct_current / max(total, 1) * 100
            val_msg = (
                f"  val_loss={avg_val:.4f}  "
                f"next_acc={acc_next:.1f}%  current_acc={acc_current:.1f}%"
            )

            # Save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                save_file(model.state_dict(), str(checkpoint_path))
                val_msg += "  *saved*"
        else:
            # No val set — save every epoch
            save_file(model.state_dict(), str(checkpoint_path))

        print(f"Epoch {epoch:3d}/{epochs}  train_loss={avg_train:.4f}{val_msg}")

    print(f"\nTraining complete. Best checkpoint: {checkpoint_path}")
