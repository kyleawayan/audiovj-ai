"""Track data models + downbeat utilities.

Naming retained from the Rekordbox era; the XML parser is gone and these models
are now populated by Raveform and migration importers. Rename pending.
"""

import json
from pathlib import Path

from pydantic import BaseModel


class TempoEntry(BaseModel):
    start_time: float
    bpm: float
    time_signature: str = "4/4"
    beat_position: int = 1


class CuePoint(BaseModel):
    start_time: float
    hotcue: int = -1  # -1 = not a Rekordbox hot cue (Raveform-style)
    phrase_type: str


class Track(BaseModel):
    track_id: str
    name: str
    artist: str = ""
    bpm: float
    location: str = ""
    filename: str = ""
    audio_path: str | None = None
    tempo_entries: list[TempoEntry] = []
    cue_points: list[CuePoint] = []
    downbeats: list[float] | None = None  # explicit per-track downbeats (Raveform)
    fold: int | None = None  # Raveform's 8-fold cross-validation index


def build_downbeat_times(
    track: Track, total_duration: float | None = None
) -> list[float]:
    """Return downbeat timestamps. Prefers track.downbeats when set; otherwise
    derives them from the first TEMPO entry (constant-BPM approximation)."""
    if track.downbeats is not None:
        return list(track.downbeats)

    if not track.tempo_entries:
        return []

    first = track.tempo_entries[0]
    beat_duration = 60.0 / first.bpm
    beats_to_downbeat = (4 - first.beat_position + 1) % 4
    first_downbeat = first.start_time + beats_to_downbeat * beat_duration

    if total_duration is None:
        total_duration = 600.0

    bar_duration = 4 * beat_duration
    downbeats: list[float] = []
    t = first_downbeat
    while t < total_duration:
        downbeats.append(t)
        t += bar_duration

    return downbeats


def save_tracks(tracks: list[Track], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        path = output_dir / f"{track.track_id}.json"
        path.write_text(track.model_dump_json(indent=2))


def load_tracks(tracks_dir: Path) -> list[Track]:
    tracks: list[Track] = []
    for path in sorted(tracks_dir.glob("*.json")):
        data = json.loads(path.read_text())
        tracks.append(Track.model_validate(data))
    return tracks
