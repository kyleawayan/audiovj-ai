"""Training data preparation: label generation and PyTorch dataset."""

import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import Track, build_downbeat_times, load_tracks


def generate_labels(
    track: Track, downbeat_times: list[float]
) -> list[dict | None]:
    """Generate training labels for each downbeat in a labeled track.

    Returns a list with one entry per downbeat (same length as downbeat_times).
    Entries are None for downbeats outside cue-point range, otherwise a dict with
    current_phrase, next_phrase, beats_until. Returns empty list if no cue points.
    """
    if not track.cue_points:
        return []

    beat_duration = 60.0 / track.bpm
    cue_times = [(c.start_time, c.phrase_type) for c in track.cue_points]

    labels: list[dict | None] = []
    for t in downbeat_times:
        # Current phrase: largest cue start_time <= t
        current_phrase = None
        for cue_time, phrase in cue_times:
            if cue_time <= t:
                current_phrase = phrase
            else:
                break

        # If we're before the first cue point, no label for this downbeat
        if current_phrase is None:
            labels.append(None)
            continue

        # Next phrase: smallest cue start_time > t
        next_phrase = None
        next_cue_time = None
        for cue_time, phrase in cue_times:
            if cue_time > t:
                next_phrase = phrase
                next_cue_time = cue_time
                break

        # If no next cue (past the last one), no label
        if next_phrase is None:
            labels.append(None)
            continue

        # Beats until next transition
        beats_until = (next_cue_time - t) / beat_duration
        # Quantize to nearest downbeat count (multiples of 4)
        beats_until = round(beats_until / 4) * 4

        labels.append(
            {
                "downbeat_time": t,
                "current_phrase": current_phrase,
                "next_phrase": next_phrase,
                "beats_until": beats_until,
            }
        )

    return labels


class PhraseDataset(Dataset):
    """PyTorch dataset that loads preprocessed windows and labels.

    Builds a flat index across all labeled tracks.
    """

    def __init__(
        self,
        track_ids: list[str],
        tracks_dir: Path = TRACKS_DIR,
        features_dir: Path = FEATURES_DIR,
    ) -> None:
        self.phrase_to_idx = {p: i for i, p in enumerate(PHRASE_TYPES)}

        # Build flat index: list of (windows_tensor, label_dict) per item
        self._windows: list[torch.Tensor] = []
        self._current_phrase: list[int] = []
        self._next_phrase: list[int] = []
        self._beats_until: list[float] = []

        for tid in track_ids:
            track_path = tracks_dir / f"{tid}.json"
            features_path = features_dir / f"{tid}.safetensors"

            if not track_path.exists() or not features_path.exists():
                continue

            track = Track.model_validate_json(track_path.read_text())
            if not track.cue_points:
                continue

            data = load_file(str(features_path))
            windows = data["windows"]
            duration = data["duration"].item()
            kept_indices = data["kept_indices"].tolist()

            downbeats = build_downbeat_times(track, total_duration=duration)
            labels = generate_labels(track, downbeats)
            if not labels:
                continue

            # Match windows to labels by downbeat index.
            # kept_indices[i] tells us which downbeat produced windows[i].
            for i, db_idx in enumerate(kept_indices):
                if i >= windows.shape[0]:
                    break
                if db_idx >= len(labels):
                    break
                lbl = labels[db_idx]
                if lbl is None:
                    continue
                self._windows.append(windows[i])
                self._current_phrase.append(
                    self.phrase_to_idx[lbl["current_phrase"]]
                )
                self._next_phrase.append(
                    self.phrase_to_idx[lbl["next_phrase"]]
                )
                self._beats_until.append(lbl["beats_until"])

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, int, float]:
        """Returns (window, current_phrase_idx, next_phrase_idx, beats_until)."""
        return (
            self._windows[idx],
            self._current_phrase[idx],
            self._next_phrase[idx],
            self._beats_until[idx],
        )


def create_splits(
    tracks_dir: Path = TRACKS_DIR,
    features_dir: Path = FEATURES_DIR,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split labeled tracks 80/20 by track for train/validation.

    Returns (train_track_ids, val_track_ids).
    """
    tracks = load_tracks(tracks_dir)

    # Only tracks with both cue points and preprocessed features
    labeled_ids: list[str] = []
    for t in tracks:
        if t.cue_points and (features_dir / f"{t.track_id}.safetensors").exists():
            labeled_ids.append(t.track_id)

    if len(labeled_ids) <= 1:
        if labeled_ids:
            print(
                f"Warning: Only {len(labeled_ids)} labeled track(s) — "
                "all assigned to training, validation set is empty"
            )
        return labeled_ids, []

    rng = random.Random(seed)
    rng.shuffle(labeled_ids)

    split_idx = max(1, int(len(labeled_ids) * 0.8))
    return labeled_ids[:split_idx], labeled_ids[split_idx:]
