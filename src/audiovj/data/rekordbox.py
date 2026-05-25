"""Track / cue-point data models and persistence.

Originally housed a Rekordbox XML parser; that path was retired when allin1's
offline analysis took over labeling. The data models below are still shared by
the rest of the pipeline (preprocess, dataset, training).
"""

import json
from pathlib import Path

from pydantic import BaseModel


class TempoEntry(BaseModel):
    start_time: float
    bpm: float
    time_signature: str
    beat_position: int


class CuePoint(BaseModel):
    start_time: float
    hotcue: int = -1  # legacy field; -1 means "not a Rekordbox hot cue"
    phrase_type: str


class Track(BaseModel):
    track_id: str
    name: str
    artist: str
    bpm: float
    location: str
    filename: str
    audio_path: str | None = None
    tempo_entries: list[TempoEntry] = []
    cue_points: list[CuePoint] = []
    downbeats: list[float] | None = None


def build_downbeat_times(
    track: Track, total_duration: float | None = None
) -> list[float]:
    """Return downbeat (beat 1) timestamps for the track.

    Prefers `track.downbeats` if set (allin1 output). Otherwise reconstructs
    from the first TEMPO entry assuming constant BPM.
    """
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
