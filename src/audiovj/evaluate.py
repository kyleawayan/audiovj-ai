"""Evaluation: accuracy, MAE, flip-flop rate, per-class breakdown."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from audiovj.config import FEATURES_DIR, FIXED_FRAMES, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
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


def evaluate_pipeline(
    checkpoint: str | None = None,
    correction_threshold: float = 0.7,
    transition_beats: float = 4.0,
    anticipate_beats: float = 8.0,
) -> list[dict]:
    """Evaluate the full pipeline (model + State Manager) on all labeled tracks.

    Processes each track left-to-right through the model and State Manager,
    comparing running_phrase against ground-truth cue points.

    Returns a list of per-track metric dicts.
    """
    from audiovj.data.dataset import generate_labels
    from audiovj.data.features import extract_mel_spectrogram, load_audio, slice_beat_windows
    from audiovj.data.rekordbox import Track, build_downbeat_times, load_tracks
    from audiovj.live.inference import PredictionResult
    from audiovj.live.state import PhraseStateManager

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt_path = checkpoint or str(MODELS_DIR / "phrase_predictor.safetensors")
    model = PhrasePredictor()
    state = load_file(ckpt_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tracks = load_tracks(TRACKS_DIR)
    labeled = [t for t in tracks if t.cue_points and t.audio_path and Path(t.audio_path).exists()]

    if not labeled:
        return [{"error": "No labeled tracks with audio found"}]

    phrase_set = set(PHRASE_TYPES)
    results = []

    for track in labeled:
        waveform, duration = load_audio(Path(track.audio_path))
        mel_spec = extract_mel_spectrogram(waveform)
        downbeats = build_downbeat_times(track, total_duration=duration)
        labels = generate_labels(track, downbeats)

        if not labels:
            continue

        # Build ground-truth cue boundaries (downbeat indices where phrase changes)
        cue_times = [c.start_time for c in track.cue_points]

        sm = PhraseStateManager(
            correction_threshold=correction_threshold,
            transition_beats=transition_beats,
            anticipate_beats=anticipate_beats,
        )

        raw_correct = 0
        sm_correct = 0
        labeled_count = 0
        corrections = 0
        transitions_fired = 0
        transition_timing_errors: list[float] = []
        transition_fire_times: list[float] = []

        with torch.no_grad():
            for i, t in enumerate(downbeats):
                lbl = labels[i] if i < len(labels) else None
                if lbl is None:
                    continue

                window, kept = slice_beat_windows(mel_spec, [t], track.bpm)
                if window.shape[0] == 0:
                    continue

                # Pad for MPS
                frames = window.shape[-1]
                pad_to = ((frames + FIXED_FRAMES - 1) // FIXED_FRAMES) * FIXED_FRAMES
                if pad_to > frames:
                    window = torch.nn.functional.pad(window, (0, pad_to - frames))

                window = window.to(device)
                out = model(window)

                next_probs = torch.softmax(out.next_phrase_logits, dim=-1)
                current_probs = torch.softmax(out.current_phrase_logits, dim=-1)
                next_idx = next_probs.argmax(-1).item()
                current_idx = current_probs.argmax(-1).item()

                prediction = PredictionResult(
                    current_phrase=PHRASE_TYPES[current_idx],
                    current_confidence=current_probs[0, current_idx].item(),
                    next_phrase=PHRASE_TYPES[next_idx],
                    next_confidence=next_probs[0, next_idx].item(),
                    beats_until=torch.expm1(out.beats_until[0, 0]).item(),
                )

                events = sm.update(prediction)

                gt_current = lbl["current_phrase"]
                # Remap ground truth to match vocabulary
                if gt_current not in phrase_set:
                    gt_current = "other" if "other" in phrase_set else gt_current
                labeled_count += 1

                if prediction.current_phrase == gt_current:
                    raw_correct += 1
                if sm.running_phrase == gt_current:
                    sm_correct += 1

                for event in events:
                    if event.kind == "correction":
                        corrections += 1
                    elif event.kind == "transition":
                        transitions_fired += 1
                        transition_fire_times.append(t)
                        # Find nearest actual cue boundary
                        beat_duration = 60.0 / track.bpm
                        min_dist = float("inf")
                        for ct in cue_times:
                            dist_beats = abs(ct - t) / beat_duration
                            if dist_beats < min_dist:
                                min_dist = dist_beats
                        transition_timing_errors.append(min_dist)

        # Count actual transitions (boundaries where phrase type changes in our vocabulary)
        remapped_cues = []
        for cp in track.cue_points:
            remapped = cp.phrase_type if cp.phrase_type in phrase_set else "other"
            remapped_cues.append((cp.start_time, remapped))
        actual_transitions = sum(
            1 for i in range(1, len(remapped_cues))
            if remapped_cues[i][1] != remapped_cues[i - 1][1]
        )

        # Transition precision: how many fired transitions were near an actual boundary?
        near_threshold = 8.0  # within 8 beats counts as "correct"
        precise_transitions = sum(1 for e in transition_timing_errors if e <= near_threshold)

        # Transition recall: how many actual boundaries had at least one nearby fired transition?
        beat_duration = 60.0 / track.bpm
        detected_boundaries = 0
        for i in range(1, len(remapped_cues)):
            if remapped_cues[i][1] != remapped_cues[i - 1][1]:
                ct = remapped_cues[i][0]
                for ft in transition_fire_times:
                    if abs(ct - ft) / beat_duration <= near_threshold:
                        detected_boundaries += 1
                        break

        results.append({
            "track_id": track.track_id,
            "name": f"{track.artist} - {track.name}",
            "labeled_downbeats": labeled_count,
            "raw_accuracy": raw_correct / max(labeled_count, 1) * 100,
            "sm_accuracy": sm_correct / max(labeled_count, 1) * 100,
            "transitions_fired": transitions_fired,
            "actual_transitions": actual_transitions,
            "transition_precision": precise_transitions / max(transitions_fired, 1) * 100,
            "transition_recall": detected_boundaries / max(actual_transitions, 1) * 100,
            "corrections": corrections,
            "correction_rate": corrections / max(labeled_count, 1),
            "mean_timing_error": sum(transition_timing_errors) / max(len(transition_timing_errors), 1),
        })

    return results
