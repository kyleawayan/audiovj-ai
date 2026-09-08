"""Evaluation: accuracy, MAE, flip-flop rate, per-class breakdown."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from audiovj.config import FEATURES_DIR, FIXED_FRAMES, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import PhraseDataset, create_splits
from audiovj.model import PhrasePredictor
from audiovj.training import _collate_variable_width

# The 4 classes that carry the live VJ use case (intro / buildup / drop / outro).
# Macro-F1 over these is the headline quality metric under class imbalance.
LOAD_BEARING = ["intro", "buildup", "drop", "outro"]


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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
    # Transition-only MAE: beats_until is only meaningful on transition samples
    # (next != current); placeholder/long-tail samples otherwise inflate it.
    mae_transition_sum = 0.0
    mae_transition_count = 0
    flip_flops = 0
    flip_opportunities = 0

    per_class_correct: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_total: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    # Confusion counts for per-class precision / recall / F1 (current phrase)
    per_class_tp: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_fp: dict[str, int] = {p: 0 for p in PHRASE_TYPES}
    per_class_fn: dict[str, int] = {p: 0 for p in PHRASE_TYPES}

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
            abs_err = (pred_beats - beats_until).abs()
            mae_sum += abs_err.sum().item()

            # Transition-only MAE: restrict to samples where a transition is
            # actually pending (ground-truth next != current).
            transition_mask = next_idx != current_idx
            if transition_mask.any():
                mae_transition_sum += abs_err[transition_mask].sum().item()
                mae_transition_count += int(transition_mask.sum().item())

            # Per-class accuracy + confusion counts (current phrase)
            for i in range(windows.shape[0]):
                phrase = PHRASE_TYPES[current_idx[i].item()]
                pred_phrase = PHRASE_TYPES[current_pred[i].item()]
                per_class_total[phrase] += 1
                if current_pred[i] == current_idx[i]:
                    per_class_correct[phrase] += 1
                    per_class_tp[phrase] += 1
                else:
                    per_class_fn[phrase] += 1
                    per_class_fp[pred_phrase] += 1

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
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    for p in PHRASE_TYPES:
        if per_class_total[p] > 0:
            per_class_acc[p] = per_class_correct[p] / per_class_total[p] * 100
        tp, fp, fn = per_class_tp[p], per_class_fp[p], per_class_fn[p]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_precision[p] = prec * 100
        per_class_recall[p] = rec * 100
        per_class_f1[p] = f1 * 100

    # Macro-F1 over the 4 load-bearing classes (matches training ckpt selection)
    macro_f1 = sum(per_class_f1[p] for p in LOAD_BEARING) / len(LOAD_BEARING)

    return {
        "next_phrase_accuracy": correct_next / max(total, 1) * 100,
        "current_phrase_accuracy": correct_current / max(total, 1) * 100,
        "beats_until_mae": mae_sum / max(total, 1),
        "beats_until_mae_transition": mae_transition_sum / max(mae_transition_count, 1),
        "transition_samples": mae_transition_count,
        "flip_flop_rate": flip_flops / max(flip_opportunities, 1) * 100,
        "per_class_accuracy": per_class_acc,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "macro_f1_load_bearing": macro_f1,
        "total_samples": total,
    }


def _track_windows(track, features_dir) -> list[tuple[float, dict, torch.Tensor]]:
    """Build per-downbeat (time, label, mel-window) samples for a track.

    Prefers precomputed features (``features/<id>.safetensors``, audio-free and
    fast); falls back to decoding the WAV + slicing on the fly. ``kept_indices``
    maps each precomputed window back to its downbeat position.
    """
    from audiovj.data.dataset import generate_labels
    from audiovj.data.features import extract_mel_spectrogram, load_audio, slice_beat_windows
    from audiovj.data.rekordbox import build_downbeat_times

    samples: list[tuple[float, dict, torch.Tensor]] = []
    feature_path = features_dir / f"{track.track_id}.safetensors"

    if feature_path.exists():
        downbeats = build_downbeat_times(track)
        labels = generate_labels(track, downbeats)
        if not labels:
            return []
        data = load_file(str(feature_path))
        windows = data["windows"]  # [num_windows, n_mels, frames]
        kept = data["kept_indices"].tolist()
        for k, idx in enumerate(kept):
            lbl = labels[idx] if idx < len(labels) else None
            if lbl is None:
                continue
            samples.append((downbeats[idx], lbl, windows[k]))
        return samples

    if track.audio_path and Path(track.audio_path).exists():
        waveform, duration = load_audio(Path(track.audio_path))
        mel_spec = extract_mel_spectrogram(waveform)
        downbeats = build_downbeat_times(track, total_duration=duration)
        labels = generate_labels(track, downbeats)
        if not labels:
            return []
        windows, kept = slice_beat_windows(mel_spec, downbeats, track.bpm)
        for k, idx in enumerate(kept):
            lbl = labels[idx] if idx < len(labels) else None
            if lbl is None:
                continue
            samples.append((downbeats[idx], lbl, windows[k]))

    return samples


def evaluate_pipeline(
    checkpoint: str | None = None,
    correction_threshold: float = 0.5,
    transition_beats: float = 4.0,
    anticipate_beats: float = 8.0,
    latch_after: int = 2,
    sticky_beats: float = 32.0,
    warmup_beats: float = 16.0,
    limit: int | None = None,
) -> list[dict]:
    """Evaluate the full pipeline (model + State Manager) on labeled tracks.

    Processes each track left-to-right through the model and State Manager,
    comparing running_phrase against ground-truth cue points. `limit` caps the
    number of tracks (quick eval); each track re-decodes its WAV + mel-spec, so
    a full pass over the corpus is slow.

    Returns a list of per-track metric dicts.
    """
    from audiovj.data.rekordbox import load_tracks
    from audiovj.live.inference import PredictionResult
    from audiovj.live.state import PhraseStateManager

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt_path = checkpoint or str(MODELS_DIR / "phrase_predictor.safetensors")
    model = PhrasePredictor()
    state = load_file(ckpt_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    def _has_inputs(t) -> bool:
        return (FEATURES_DIR / f"{t.track_id}.safetensors").exists() or bool(
            t.audio_path and Path(t.audio_path).exists()
        )

    tracks = load_tracks(TRACKS_DIR)
    labeled = [t for t in tracks if t.cue_points and _has_inputs(t)]

    if not labeled:
        return [{"error": "No labeled tracks with features or audio found"}]

    if limit is not None:
        labeled = labeled[:limit]

    results = []
    total = len(labeled)

    for i, track in enumerate(labeled, 1):
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  [pipeline] {i}/{total} tracks...", flush=True)

        # Per-downbeat (time, label, window) samples. Prefer precomputed
        # features (audio-free, fast); fall back to decoding the WAV.
        samples = _track_windows(track, FEATURES_DIR)
        if not samples:
            continue

        # Build ground-truth cue boundaries (downbeat indices where phrase changes)
        cue_times = [c.start_time for c in track.cue_points]

        sm = PhraseStateManager(
            correction_threshold=correction_threshold,
            transition_beats=transition_beats,
            anticipate_beats=anticipate_beats,
            latch_after=latch_after,
            sticky_beats=sticky_beats,
            warmup_beats=warmup_beats,
        )

        raw_correct = 0
        sm_correct = 0
        labeled_count = 0
        corrections = 0
        transitions_fired = 0
        transition_timing_errors: list[float] = []
        # Phrase-change fire times = transitions AND corrections. A correction
        # that lands on a boundary is a legitimate detection, so it counts
        # toward recall/precision/timing (KA-167 choice).
        change_fire_times: list[float] = []
        # Countdown quality: when the SM has an active mechanical countdown,
        # compare its remaining-beats against the ground-truth beats_until.
        countdown_gt: list[float] = []
        countdown_pred: list[float] = []

        with torch.no_grad():
            for t, lbl, window in samples:
                # window: [n_mels, frames] — add batch dim and pad to a multiple
                # of FIXED_FRAMES (matches the prior pipeline; MPS-friendly).
                window = window.unsqueeze(0)
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

                gt_phrase = lbl["current_phrase"]
                labeled_count += 1

                if prediction.current_phrase == gt_phrase:
                    raw_correct += 1
                if sm.running_phrase == gt_phrase:
                    sm_correct += 1

                # Countdown quality: GT beats_until vs the SM's mechanical
                # countdown, sampled whenever a countdown is active.
                cd = sm.countdown
                if cd is not None and lbl.get("beats_until") is not None:
                    countdown_gt.append(float(lbl["beats_until"]))
                    countdown_pred.append(float(cd[1]))

                beat_duration = 60.0 / track.bpm
                for event in events:
                    if event.kind == "transition":
                        transitions_fired += 1
                    elif event.kind == "correction":
                        corrections += 1
                    if event.kind in ("transition", "correction"):
                        change_fire_times.append(t)
                        # Distance (in beats) to the nearest actual cue boundary
                        min_dist = float("inf")
                        for ct in cue_times:
                            dist_beats = abs(ct - t) / beat_duration
                            if dist_beats < min_dist:
                                min_dist = dist_beats
                        transition_timing_errors.append(min_dist)

        # Count actual transitions (cue point boundaries)
        actual_transitions = max(len(track.cue_points) - 1, 0)

        # Precision: how many fired phrase-changes (transitions + corrections)
        # landed near an actual boundary?
        near_threshold = 8.0  # within 8 beats counts as "correct"
        total_fires = len(change_fire_times)
        precise_transitions = sum(1 for e in transition_timing_errors if e <= near_threshold)

        # Recall + matched latency: for each actual boundary, distance to the
        # nearest fired change. A boundary is "detected" if that distance is
        # within near_threshold; matched latency averages those distances.
        # NOTE: matched latency (boundary->nearest fire) is the real cueing
        # precision. It is NOT mean_timing_error, which averages over all fires
        # and is inflated by mid-phrase false fires.
        beat_duration = 60.0 / track.bpm
        detected_boundaries = 0
        matched_latencies: list[float] = []
        for ct in cue_times[1:]:  # skip first cue (no transition into it)
            if not change_fire_times:
                continue
            d = min(abs(ct - ft) / beat_duration for ft in change_fire_times)
            if d <= near_threshold:
                detected_boundaries += 1
                matched_latencies.append(d)

        # Countdown quality (this track): MAE, Pearson correlation, monotonicity.
        cq = _countdown_quality(countdown_gt, countdown_pred)

        results.append({
            "track_id": track.track_id,
            "name": f"{track.artist} - {track.name}",
            "labeled_downbeats": labeled_count,
            "raw_accuracy": raw_correct / max(labeled_count, 1) * 100,
            "sm_accuracy": sm_correct / max(labeled_count, 1) * 100,
            "transitions_fired": transitions_fired,
            "actual_transitions": actual_transitions,
            "transition_precision": precise_transitions / max(total_fires, 1) * 100,
            "transition_recall": detected_boundaries / max(actual_transitions, 1) * 100,
            "corrections": corrections,
            "correction_rate": corrections / max(labeled_count, 1),
            "mean_timing_error": sum(transition_timing_errors) / max(len(transition_timing_errors), 1),
            "matched_latency": sum(matched_latencies) / max(len(matched_latencies), 1),
            "detected_boundaries": detected_boundaries,
            "countdown_samples": cq["n"],
            "countdown_mae": cq["mae"],
            "countdown_corr": cq["corr"],
            "countdown_monotonicity": cq["monotonicity"],
        })

    return results


def _countdown_quality(gt: list[float], pred: list[float]) -> dict:
    """Countdown-quality metrics over paired (ground-truth, predicted) beats.

    - mae: mean absolute error in beats.
    - corr: Pearson correlation between GT and predicted countdown.
    - monotonicity: fraction of consecutive steps where the predicted countdown
      decreases whenever the ground-truth countdown decreases (the mechanical
      countdown should be strictly monotone as a transition approaches).
    """
    n = len(gt)
    if n == 0:
        return {"n": 0, "mae": 0.0, "corr": 0.0, "monotonicity": 0.0}

    mae = sum(abs(g - p) for g, p in zip(gt, pred)) / n

    corr = 0.0
    if n > 1:
        mean_g = sum(gt) / n
        mean_p = sum(pred) / n
        cov = sum((g - mean_g) * (p - mean_p) for g, p in zip(gt, pred))
        std_g = sum((g - mean_g) ** 2 for g in gt) ** 0.5
        std_p = sum((p - mean_p) ** 2 for p in pred) ** 0.5
        if std_g > 0 and std_p > 0:
            corr = cov / (std_g * std_p)

    mono_pairs = 0
    mono_correct = 0
    for j in range(1, n):
        if gt[j] < gt[j - 1]:
            mono_pairs += 1
            if pred[j] < pred[j - 1]:
                mono_correct += 1
    monotonicity = mono_correct / mono_pairs if mono_pairs > 0 else 0.0

    return {"n": n, "mae": mae, "corr": corr, "monotonicity": monotonicity}


def evaluate_seq(
    checkpoint: str | None = None,
    onset_threshold: float = 0.30,
    fold: int | None = None,
    limit: int | None = None,
    tolerance: float = 2.0,
    drop_confirm: int = 1,
    drop_release: int = 2,
) -> dict:
    """Offline twin of the live path: drives the PRODUCTION components
    (SeqInferenceEngine stateful streaming + OnsetCueTracker @onset_threshold)
    over labeled tracks and reports the locked operating-point metrics
    (LB transition recall / precision / latency, per-class, drop events).

    This is the same inference + cueing run-live uses, minus the audio-capture /
    Carabiner / OSC-transport layers. Pass fold=7 for the held-out certification
    number. Reproduces experiments/_cue.py onset@0.30 (~58% LB recall).
    """
    from audiovj.data.rekordbox import load_tracks
    from audiovj.live.cue import OnsetCueTracker
    from audiovj.live.inference import SeqInferenceEngine

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = checkpoint or str(MODELS_DIR / "seq_unified.safetensors")
    if not Path(ckpt).exists():
        return {"error": f"checkpoint not found: {ckpt}"}
    engine = SeqInferenceEngine(Path(ckpt), device)

    # Import the live tracker's own class list so evaluate-seq stays a true
    # offline twin of run-live. Hardcoding it here silently decouples the two the
    # moment the live cue set changes.
    from audiovj.live.cue import _LB as lb
    lb_set = set(lb)
    tracks = [
        t for t in load_tracks(TRACKS_DIR)
        if t.cue_points and (FEATURES_DIR / f"{t.track_id}.safetensors").exists()
    ]
    if fold is not None:
        tracks = [t for t in tracks if t.fold == fold]
    if limit is not None:
        tracks = tracks[:limit]

    # Collect RAW SIGNED deltas once, derive every metric post-hoc. The previous
    # version folded matching, tolerance and averaging into one pass with abs()
    # and a hardcoded 8-beat window, which made it blind to the very thing the
    # live path was changed to fix (a 4-beat lateness) and unable to distinguish
    # "fired 4 beats early" from "fired 4 beats late".
    #
    # Sign convention throughout: POSITIVE = LATE (fire after the boundary).
    boundaries: list[tuple[str, float | None]] = []   # (phrase, signed delta | None)
    fire_offsets: list[float] = []                    # per fire -> signed dist to nearest boundary
    drop_start_offsets: list[float | None] = []       # per actual drop -> signed delta
    drop_end_offsets: list[float | None] = []         # per actual drop exit -> signed delta
    n_drop_start_fires = n_drop_end_fires = 0
    fires = 0
    n_tracks = 0

    def _nearest(targets: list[float], x: float, bd: float) -> float | None:
        """Signed distance (beats) from x to the nearest element of targets."""
        if not targets:
            return None
        best = min(targets, key=lambda y: abs(y - x))
        return (x - best) / bd

    def _fire_delta(fire_times: list[float], boundary: float, bd: float) -> float | None:
        """Signed beats from a boundary to the nearest fire. POSITIVE = LATE.

        Distinct from _nearest's argument order on purpose: every boundary metric
        asks "how late was the cue", so the boundary is the reference, not the fire.
        """
        if not fire_times:
            return None
        best = min(fire_times, key=lambda f: abs(f - boundary))
        return (best - boundary) / bd

    for track in tracks:
        samples = _track_windows(track, FEATURES_DIR)
        if not samples:
            continue
        n_tracks += 1
        engine.reset()
        cue = OnsetCueTracker(
            onset_threshold=onset_threshold,
            drop_confirm=drop_confirm,
            drop_release=drop_release,
        )
        bd = 60.0 / track.bpm
        fire_times: list[float] = []
        drop_start_times: list[float] = []
        drop_end_times: list[float] = []
        for t, _lbl, window in samples:
            pred = engine.step_window(window)
            for e in cue.update(pred):
                if e.kind == "transition":
                    fire_times.append(t)
                elif e.kind == "drop_start":
                    drop_start_times.append(t)
                elif e.kind == "drop_end":
                    drop_end_times.append(t)
        n_drop_start_fires += len(drop_start_times)
        n_drop_end_fires += len(drop_end_times)

        cues = [(c.start_time, c.phrase_type) for c in track.cue_points]
        cue_times = [c0 for c0, _ in cues]
        fires += len(fire_times)
        for f in fire_times:
            d = _nearest(cue_times, f, bd)
            if d is not None:
                fire_offsets.append(d)

        # Transition boundaries. cues[1:] because the first cue is the track's
        # own start, not a transition anyone could cue on.
        for c0, ph in cues[1:]:
            if ph not in lb_set:
                continue
            boundaries.append((ph, _fire_delta(fire_times, c0, bd)))

        # Drop entries, and drop EXITS (the cue immediately following a drop).
        # These were previously only COUNTED, so the two events the rig cares
        # most about had no timing number at all.
        for i, (c0, ph) in enumerate(cues):
            if ph != "drop":
                continue
            drop_start_offsets.append(_fire_delta(drop_start_times, c0, bd))
            if i + 1 < len(cues):
                drop_end_offsets.append(_fire_delta(drop_end_times, cues[i + 1][0], bd))

    def _stats(deltas: list[float | None], tol: float,
               match_window: float = 8.0) -> dict:
        """Recall at ``tol``, latency distribution over a WIDER ``match_window``.

        These must be separate. Measuring latency only over items already inside
        a tight tolerance is circular: a +/-2 beat tolerance can never reveal a
        4-beat lateness, because everything that survives the filter is within 2
        beats by construction. So: recall answers "did it fire near this
        boundary at all", the distribution answers "and how late was it".
        """
        matched = [d for d in deltas if d is not None and abs(d) <= tol]
        paired = [d for d in deltas if d is not None and abs(d) <= match_window]
        n = len(deltas)
        out = {
            "n": n,
            "matched": len(matched),
            "recall": len(matched) / max(n, 1) * 100,
            "paired": len(paired),
        }
        if paired:
            srt = sorted(paired)
            out["median_beats"] = srt[len(srt) // 2]
            out["mean_beats"] = sum(srt) / len(srt)
            out["p90_abs_beats"] = sorted(abs(d) for d in srt)[int(0.9 * (len(srt) - 1))]
            out["pct_late"] = sum(1 for d in srt if d > 0) / len(srt) * 100
            out["pct_early"] = sum(1 for d in srt if d < 0) / len(srt) * 100
        else:
            out |= {"median_beats": 0.0, "mean_beats": 0.0, "p90_abs_beats": 0.0,
                    "pct_late": 0.0, "pct_early": 0.0}
        return out

    trans = _stats([d for _, d in boundaries], tolerance)
    per_class = {}
    for p in lb:
        ds = [d for ph, d in boundaries if ph == p]
        st = _stats(ds, tolerance)
        # Report the denominator: a 0% with n=0 is undefined, not a failure.
        per_class[p] = {"recall": st["recall"], "n": st["n"],
                        "median_beats": st["median_beats"]}

    near = sum(1 for d in fire_offsets if abs(d) <= tolerance)

    # Recall is a strong function of tolerance; a single number hides that the
    # misses are mostly near-misses with timing scatter rather than blindness.
    sweep = {
        t: _stats([d for _, d in boundaries], t)["recall"]
        for t in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    }

    return {
        "n_tracks": n_tracks,
        "fold": fold,
        "tolerance_beats": tolerance,
        "lb_transition_recall": trans["recall"],
        "fire_precision": near / max(fires, 1) * 100,
        # Kept for continuity with older runs: mean ABSOLUTE matched latency.
        "matched_latency_beats": abs(trans["mean_beats"]),
        "transition": trans,
        "fires": fires,
        "per_class_recall": {p: per_class[p]["recall"] for p in lb},
        "per_class": per_class,
        "recall_by_tolerance": sweep,
        "drop_start": _stats(drop_start_offsets, tolerance),
        "drop_end": _stats(drop_end_offsets, tolerance),
        "drop_start_events": n_drop_start_fires,
        "drop_end_events": n_drop_end_fires,
    }
