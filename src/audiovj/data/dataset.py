"""Training data preparation: label generation and PyTorch dataset."""

import random
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from audiovj.config import DROP_LENGTH_BEATS, FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import Track, build_downbeat_times, load_tracks


def generate_labels(
    track: Track, downbeat_times: list[float]
) -> list[dict | None]:
    """Generate training labels for each downbeat from drop cue points.

    Only requires "drop" cue points (hot cue D). Each drop is assumed to be
    DROP_LENGTH_BEATS long. Synthesizes boundaries:
      - drop start at cue time
      - drop end at cue time + DROP_LENGTH_BEATS * beat_duration

    Returns a list with one entry per downbeat (same length as downbeat_times).
    Entries are None for downbeats outside any labeled region. Returns empty list
    if no drop cue points.
    """
    # Filter to drop cues only
    drop_cues = [c for c in track.cue_points if c.phrase_type == "drop"]
    if not drop_cues:
        return []

    beat_duration = 60.0 / track.bpm
    drop_length_secs = DROP_LENGTH_BEATS * beat_duration

    # Build sorted list of (start_time, end_time) for each drop region
    drop_regions = []
    for c in drop_cues:
        drop_regions.append((c.start_time, c.start_time + drop_length_secs))

    # Build transition boundaries: each boundary is (time, phrase_type_after)
    # sorted by time. Transitions alternate: other→drop at start, drop→other at end.
    boundaries: list[tuple[float, str]] = []
    for start, end in drop_regions:
        boundaries.append((start, "drop"))
        boundaries.append((end, "other"))
    boundaries.sort(key=lambda x: x[0])

    labels: list[dict | None] = []
    for t in downbeat_times:
        # Determine current phrase at time t
        current_phrase = "other"
        for start, end in drop_regions:
            if start <= t < end:
                current_phrase = "drop"
                break

        # Find next transition boundary after t
        next_boundary_time = None
        next_phrase = None
        for b_time, b_phrase in boundaries:
            if b_time > t:
                # Only count if it's actually a transition (different from current)
                if b_phrase != current_phrase:
                    next_boundary_time = b_time
                    next_phrase = b_phrase
                    break

        if next_phrase is None:
            if current_phrase == "other":
                # No upcoming transition — label as other→other so the model
                # learns what non-buildup "other" sounds like.
                labels.append({
                    "downbeat_time": t,
                    "current_phrase": "other",
                    "next_phrase": "other",
                    "beats_until": 999,  # large placeholder (log1p(999)≈6.9 vs real max ~4.2)
                })
            else:
                # In a drop with no next boundary (past last drop end) — skip
                labels.append(None)
            continue

        beats_until = (next_boundary_time - t) / beat_duration
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
                cur = lbl["current_phrase"]
                nxt = lbl["next_phrase"]
                # Remap unknown phrases to "other" if it exists
                if cur not in self.phrase_to_idx:
                    if "other" in self.phrase_to_idx:
                        cur = "other"
                    else:
                        continue
                if nxt not in self.phrase_to_idx:
                    if "other" in self.phrase_to_idx:
                        nxt = "other"
                    else:
                        continue
                self._windows.append(windows[i])
                self._current_phrase.append(
                    self.phrase_to_idx[cur]
                )
                self._next_phrase.append(
                    self.phrase_to_idx[nxt]
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
) -> tuple[list[str], list[str], list[str]]:
    """Split labeled tracks into train/val/test by track.

    75% train+val (internally split 80/20), 25% held-out test.
    Returns (train_track_ids, val_track_ids, test_track_ids).
    """
    tracks = load_tracks(tracks_dir)

    # Only tracks with both cue points and preprocessed features
    labeled_ids: list[str] = []
    for t in tracks:
        if t.cue_points and (features_dir / f"{t.track_id}.safetensors").exists():
            labeled_ids.append(t.track_id)

    if len(labeled_ids) <= 2:
        if labeled_ids:
            print(
                f"Warning: Only {len(labeled_ids)} labeled track(s) — "
                "all assigned to training, val/test sets are empty"
            )
        return labeled_ids, [], []

    rng = random.Random(seed)
    rng.shuffle(labeled_ids)

    # 25% held-out test
    test_idx = max(1, int(len(labeled_ids) * 0.75))
    trainval_ids = labeled_ids[:test_idx]
    test_ids = labeled_ids[test_idx:]

    # 80/20 train/val within trainval
    val_idx = max(1, int(len(trainval_ids) * 0.8))
    train_ids = trainval_ids[:val_idx]
    val_ids = trainval_ids[val_idx:]

    return train_ids, val_ids, test_ids
