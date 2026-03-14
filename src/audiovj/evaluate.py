"""Evaluation: accuracy, MAE, flip-flop rate, per-class breakdown."""

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from audiovj.config import FEATURES_DIR, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import PhraseDataset, create_splits
from audiovj.model import PhrasePredictor
from audiovj.training import _collate_variable_width


def evaluate_model(
    checkpoint: str | None = None,
    batch_size: int = 8,
) -> dict:
    """Evaluate a trained model on the validation split.

    Returns dict of metrics:
      - next_phrase_accuracy
      - current_phrase_accuracy
      - beats_until_mae
      - flip_flop_rate
      - per_class_accuracy (dict by phrase type)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    ckpt_path = checkpoint or str(MODELS_DIR / "phrase_predictor.safetensors")
    model = PhrasePredictor()
    state = load_file(ckpt_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Validation data
    _, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    if not val_ids:
        print("Warning: No validation tracks. Evaluating on training set.")
        val_ids, _ = create_splits(TRACKS_DIR, FEATURES_DIR)

    val_ds = PhraseDataset(val_ids, TRACKS_DIR, FEATURES_DIR)
    if len(val_ds) == 0:
        return {"error": "No evaluation samples"}

    loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=_collate_variable_width)

    # Accumulators
    correct_next = 0
    correct_current = 0
    total = 0
    mae_sum = 0.0
    flip_flops = 0
    flip_opportunities = 0

    per_class_correct: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_total: dict[str, int] = {p: 0 for p in PHRASE_TYPES}

    prev_next_pred = None
    prev_current_pred = None

    with torch.no_grad():
        for windows, current_idx, next_idx, beats_until in loader:
            windows = windows.to(device)
            current_idx = current_idx.to(device)
            next_idx = next_idx.to(device)
            beats_until = beats_until.float().to(device)

            out = model(windows)

            next_pred = out.next_phrase_logits.argmax(-1)
            current_pred = out.current_phrase_logits.argmax(-1)

            correct_next += (next_pred == next_idx).sum().item()
            correct_current += (current_pred == current_idx).sum().item()
            total += windows.shape[0]

            # Convert log-space predictions back to beat-space for MAE
            pred_beats = torch.expm1(out.beats_until.squeeze(-1))
            mae_sum += (pred_beats - beats_until).abs().sum().item()

            # Per-class accuracy (current phrase)
            for i in range(windows.shape[0]):
                phrase = PHRASE_TYPES[current_idx[i].item()]
                per_class_total[phrase] += 1
                if current_pred[i] == current_idx[i]:
                    per_class_correct[phrase] += 1

            # Flip-flop: next_phrase changes but current_phrase doesn't
            if prev_next_pred is not None:
                # Compare last element of previous batch with first of current
                if prev_current_pred == current_pred[0].item():
                    flip_opportunities += 1
                    if prev_next_pred != next_pred[0].item():
                        flip_flops += 1

            # Within-batch flip-flop
            for i in range(1, windows.shape[0]):
                if current_pred[i] == current_pred[i - 1]:
                    flip_opportunities += 1
                    if next_pred[i] != next_pred[i - 1]:
                        flip_flops += 1

            prev_next_pred = next_pred[-1].item()
            prev_current_pred = current_pred[-1].item()

    per_class_acc = {}
    for p in PHRASE_TYPES:
        if per_class_total[p] > 0:
            per_class_acc[p] = per_class_correct[p] / per_class_total[p] * 100

    return {
        "next_phrase_accuracy": correct_next / max(total, 1) * 100,
        "current_phrase_accuracy": correct_current / max(total, 1) * 100,
        "beats_until_mae": mae_sum / max(total, 1),
        "flip_flop_rate": flip_flops / max(flip_opportunities, 1) * 100,
        "per_class_accuracy": per_class_acc,
        "total_samples": total,
    }
