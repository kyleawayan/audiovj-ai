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
    _, val_ids, _ = create_splits(TRACKS_DIR, FEATURES_DIR)
    if not val_ids:
        print("Warning: No validation tracks. Evaluating on training set.")
        val_ids, _, _ = create_splits(TRACKS_DIR, FEATURES_DIR)

    val_ds = PhraseDataset(val_ids, TRACKS_DIR, FEATURES_DIR)
    if len(val_ds) == 0:
        return {"error": "No evaluation samples"}

    loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=_collate_variable_width)

    # Accumulators
    correct_next = 0
    correct_current = 0
    total = 0
    mae_sum = 0.0
    mae_transition_count = 0
    flip_flops = 0
    flip_opportunities = 0

    per_class_correct: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_total: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_tp: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_fp: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_fn: dict[str, int] = {p: 0 for p in PHRASE_TYPES}

    next_tp: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    next_fp: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    next_fn: dict[str, int] = {p: 0 for p in PHRASE_TYPES}

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
            # Only compute on transition samples (where next != current)
            pred_beats = torch.expm1(out.beats_until.squeeze(-1))
            transition_mask = next_idx != current_idx
            if transition_mask.any():
                mae_sum += (pred_beats[transition_mask] - beats_until[transition_mask]).abs().sum().item()
                mae_transition_count += transition_mask.sum().item()

            # Per-class accuracy and P/R/F1 (current and next phrase)
            for i in range(windows.shape[0]):
                gt_phrase = PHRASE_TYPES[current_idx[i].item()]
                pred_phrase = PHRASE_TYPES[current_pred[i].item()]
                per_class_total[gt_phrase] += 1
                if current_pred[i] == current_idx[i]:
                    per_class_correct[gt_phrase] += 1
                    per_class_tp[gt_phrase] += 1
                else:
                    per_class_fn[gt_phrase] += 1
                    per_class_fp[pred_phrase] += 1

                gt_next = PHRASE_TYPES[next_idx[i].item()]
                pred_next = PHRASE_TYPES[next_pred[i].item()]
                if next_pred[i] == next_idx[i]:
                    next_tp[gt_next] += 1
                else:
                    next_fn[gt_next] += 1
                    next_fp[pred_next] += 1

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
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    next_precision = {}
    next_recall = {}
    next_f1 = {}
    for p in PHRASE_TYPES:
        if per_class_total[p] > 0:
            per_class_acc[p] = per_class_correct[p] / per_class_total[p] * 100
        tp = per_class_tp[p]
        fp = per_class_fp[p]
        fn = per_class_fn[p]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_precision[p] = prec * 100
        per_class_recall[p] = rec * 100
        per_class_f1[p] = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0.0

        ntp = next_tp[p]
        nfp = next_fp[p]
        nfn = next_fn[p]
        nprec = ntp / (ntp + nfp) if (ntp + nfp) > 0 else 0.0
        nrec = ntp / (ntp + nfn) if (ntp + nfn) > 0 else 0.0
        next_precision[p] = nprec * 100
        next_recall[p] = nrec * 100
        next_f1[p] = 2 * nprec * nrec / (nprec + nrec) * 100 if (nprec + nrec) > 0 else 0.0

    return {
        "next_phrase_accuracy": correct_next / max(total, 1) * 100,
        "current_phrase_accuracy": correct_current / max(total, 1) * 100,
        "beats_until_mae": mae_sum / max(mae_transition_count, 1),
        "beats_until_transition_samples": mae_transition_count,
        "flip_flop_rate": flip_flops / max(flip_opportunities, 1) * 100,
        "per_class_accuracy": per_class_acc,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "next_precision": next_precision,
        "next_recall": next_recall,
        "next_f1": next_f1,
        "total_samples": total,
    }


def evaluate_pipeline(
    checkpoint: str | None = None,
    correction_threshold: float = 0.5,
    transition_beats: float = 4.0,
    anticipate_beats: float = 8.0,
) -> tuple[list[dict], dict]:
    """Evaluate the full pipeline (model + State Manager) on held-out test tracks.

    Processes each track left-to-right through the model and State Manager,
    comparing running_phrase against ground-truth cue points.

    Returns (per_track_results, drop_metrics).
    """
    from audiovj.data.dataset import create_splits, generate_labels
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

    # Only evaluate on held-out test tracks
    _, _, test_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    test_set = set(test_ids)
    tracks = load_tracks(TRACKS_DIR)
    labeled = [
        t for t in tracks
        if t.track_id in test_set and t.cue_points and t.audio_path and Path(t.audio_path).exists()
    ]

    if not labeled:
        return [{"error": "No labeled tracks with audio found"}]

    phrase_set = set(PHRASE_TYPES)
    results = []

    # Aggregate current drop P/R/F1 across all tracks
    agg_drop_tp = 0
    agg_drop_fp = 0
    agg_drop_fn = 0

    # Aggregate next_phrase drop P/R/F1 across all tracks
    agg_next_drop_tp = 0
    agg_next_drop_fp = 0
    agg_next_drop_fn = 0

    for track in labeled:
        waveform, duration = load_audio(Path(track.audio_path))
        mel_spec = extract_mel_spectrogram(waveform)
        downbeats = build_downbeat_times(track, total_duration=duration)
        labels = generate_labels(track, downbeats)

        if not labels:
            continue

        # Build ground-truth boundaries from drop cues + assumed duration
        from audiovj.config import DROP_LENGTH_BEATS
        beat_dur = 60.0 / track.bpm
        drop_cues = [c for c in track.cue_points if c.phrase_type == "drop"]
        # Synthesized boundary times (drop start + drop end)
        cue_times = []
        for c in drop_cues:
            cue_times.append(c.start_time)
            cue_times.append(c.start_time + DROP_LENGTH_BEATS * beat_dur)
        cue_times.sort()

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

        # Countdown quality: track (gt, pred) beats_until for transition samples
        countdown_gt: list[float] = []
        countdown_pred: list[float] = []

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

                # Aggregate current drop detection metrics (raw model)
                if gt_current == "drop" and prediction.current_phrase == "drop":
                    agg_drop_tp += 1
                elif gt_current != "drop" and prediction.current_phrase == "drop":
                    agg_drop_fp += 1
                elif gt_current == "drop" and prediction.current_phrase != "drop":
                    agg_drop_fn += 1

                # Aggregate next_phrase drop detection metrics
                gt_next = lbl["next_phrase"]
                if gt_next not in phrase_set:
                    gt_next = "other" if "other" in phrase_set else gt_next
                if gt_next == "drop" and prediction.next_phrase == "drop":
                    agg_next_drop_tp += 1
                elif gt_next != "drop" and prediction.next_phrase == "drop":
                    agg_next_drop_fp += 1
                elif gt_next == "drop" and prediction.next_phrase != "drop":
                    agg_next_drop_fn += 1

                # Countdown quality: only when SM has active countdown
                # for a transition sample (avoids noisy raw model fallback)
                gt_beats = lbl["beats_until"]
                sm_cd = sm.countdown
                if gt_current != gt_next and gt_beats < 999 and sm_cd is not None:
                    countdown_gt.append(gt_beats)
                    countdown_pred.append(sm_cd[1])

                for event in events:
                    if event.kind == "correction":
                        corrections += 1
                    # Count both transitions and corrections as phrase changes
                    # for timing/recall evaluation
                    if event.kind in ("transition", "correction"):
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

        # Count actual transitions from synthesized boundaries
        # Each drop cue produces 2 transitions: other→drop and drop→other
        actual_transitions = len(drop_cues) * 2

        # Transition precision: how many fired transitions were near an actual boundary?
        near_threshold = 8.0  # within 8 beats counts as "correct"
        precise_transitions = sum(1 for e in transition_timing_errors if e <= near_threshold)

        # Transition recall: how many actual boundaries had at least one nearby fired transition?
        detected_boundaries = 0
        for ct in cue_times:
            for ft in transition_fire_times:
                if abs(ct - ft) / beat_dur <= near_threshold:
                    detected_boundaries += 1
                    break

        # Countdown quality metrics
        countdown_mae = 0.0
        countdown_corr = 0.0
        countdown_mono = 1.0  # Vacuously true for 0-1 samples
        countdown_mono_pairs = 0
        countdown_mono_correct = 0
        countdown_n = len(countdown_gt)
        if countdown_n > 0:
            countdown_mae = sum(abs(g - p) for g, p in zip(countdown_gt, countdown_pred)) / countdown_n
            # Pearson correlation
            if countdown_n > 1:
                mean_g = sum(countdown_gt) / countdown_n
                mean_p = sum(countdown_pred) / countdown_n
                cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(countdown_gt, countdown_pred))
                std_g = (sum((g - mean_g) ** 2 for g in countdown_gt)) ** 0.5
                std_p = (sum((p - mean_p) ** 2 for p in countdown_pred)) ** 0.5
                if std_g > 0 and std_p > 0:
                    countdown_corr = cov / (std_g * std_p)
            # Monotonicity: fraction of consecutive pairs where pred decreases
            if countdown_n > 1:
                for j in range(1, countdown_n):
                    if countdown_gt[j] < countdown_gt[j - 1]:
                        countdown_mono_pairs += 1
                        if countdown_pred[j] < countdown_pred[j - 1]:
                            countdown_mono_correct += 1
                if countdown_mono_pairs > 0:
                    countdown_mono = countdown_mono_correct / countdown_mono_pairs

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
            "countdown_mae": countdown_mae,
            "countdown_corr": countdown_corr,
            "countdown_mono": countdown_mono,
            "countdown_mono_pairs": countdown_mono_pairs,
            "countdown_mono_correct": countdown_mono_correct,
            "countdown_samples": countdown_n,
        })

    # Compute aggregate current drop P/R/F1
    drop_prec = agg_drop_tp / (agg_drop_tp + agg_drop_fp) if (agg_drop_tp + agg_drop_fp) > 0 else 0.0
    drop_rec = agg_drop_tp / (agg_drop_tp + agg_drop_fn) if (agg_drop_tp + agg_drop_fn) > 0 else 0.0
    drop_f1 = 2 * drop_prec * drop_rec / (drop_prec + drop_rec) if (drop_prec + drop_rec) > 0 else 0.0

    # Compute aggregate next drop P/R/F1
    nd_prec = agg_next_drop_tp / (agg_next_drop_tp + agg_next_drop_fp) if (agg_next_drop_tp + agg_next_drop_fp) > 0 else 0.0
    nd_rec = agg_next_drop_tp / (agg_next_drop_tp + agg_next_drop_fn) if (agg_next_drop_tp + agg_next_drop_fn) > 0 else 0.0
    nd_f1 = 2 * nd_prec * nd_rec / (nd_prec + nd_rec) if (nd_prec + nd_rec) > 0 else 0.0

    # Aggregate countdown metrics across all tracks
    total_cd_samples = sum(r["countdown_samples"] for r in results)
    agg_cd_mae = (
        sum(r["countdown_mae"] * r["countdown_samples"] for r in results)
        / max(total_cd_samples, 1)
    )
    agg_cd_corr = (
        sum(r["countdown_corr"] * r["countdown_samples"] for r in results)
        / max(total_cd_samples, 1)
    )
    # Aggregate monotonicity at pair level (not sample level)
    total_mono_pairs = sum(r["countdown_mono_pairs"] for r in results)
    total_mono_correct = sum(r["countdown_mono_correct"] for r in results)
    agg_cd_mono = total_mono_correct / max(total_mono_pairs, 1)

    return results, {
        "drop_precision": drop_prec * 100,
        "drop_recall": drop_rec * 100,
        "drop_f1": drop_f1 * 100,
        "next_drop_precision": nd_prec * 100,
        "next_drop_recall": nd_rec * 100,
        "next_drop_f1": nd_f1 * 100,
        "countdown_mae": agg_cd_mae,
        "countdown_corr": agg_cd_corr,
        "countdown_mono": agg_cd_mono,
        "countdown_samples": total_cd_samples,
    }
