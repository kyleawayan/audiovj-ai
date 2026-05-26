"""Import tracks from the Raveform EDM dataset.

Raveform ships:
  <RAVEFORM_DIR>/structures/segments.json          - per-track annotations
  <RAVEFORM_DIR>/structures/beats/<KEY>.beat.csv   - per-track beats (time, downbeat, section)

Audio is NOT shipped; the user must place WAVs at <AUDIO_DIR>/<KEY>.wav before
import. WAV-only on purpose: compressed codecs (MP3, M4A, etc.) decode with
small (~20-40ms) timing offsets that throw off beat alignment.
Tracks whose audio is missing are skipped silently.
"""

import csv
import json
from pathlib import Path

from audiovj.config import PHRASE_TYPES
from audiovj.data.rekordbox import CuePoint, Track

AUDIO_EXTENSIONS = (".wav",)


def _load_segments(raveform_dir: Path) -> list[dict]:
    path = raveform_dir / "structures" / "segments.json"
    return json.loads(path.read_text())


def _load_downbeats(raveform_dir: Path, key: str) -> list[float]:
    """Return downbeat timestamps (time where downbeat == 1) for one track."""
    csv_path = raveform_dir / "structures" / "beats" / f"{key}.beat.csv"
    if not csv_path.exists():
        return []
    downbeats: list[float] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["downbeat"] == "1":
                downbeats.append(float(row["time"]))
    return downbeats


def _find_audio_for_key(audio_dir: Path, key: str) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        p = audio_dir / f"{key}{ext}"
        if p.exists():
            return p
    return None


def _cue_points_from_sections(sections: list[dict], track_key: str) -> list[CuePoint]:
    cues: list[CuePoint] = []
    for s in sections:
        label = s["name"]
        if label not in PHRASE_TYPES:
            print(f"  warn: unknown segment label '{label}' in {track_key}, skipping")
            continue
        cues.append(
            CuePoint(
                start_time=float(s["start"]),
                hotcue=-1,
                phrase_type=label,
            )
        )
    cues.sort(key=lambda c: c.start_time)
    return cues


def import_raveform(
    raveform_dir: Path,
    audio_dir: Path,
    limit: int | None = None,
) -> tuple[list[Track], int, int]:
    """Build Tracks from Raveform metadata + locally-available audio.

    Returns (tracks, skipped_missing_audio, skipped_no_cues).
    """
    raveform_dir = raveform_dir.resolve()
    audio_dir = audio_dir.resolve()

    entries = _load_segments(raveform_dir)
    if limit is not None:
        entries = entries[:limit]

    tracks: list[Track] = []
    skipped_missing_audio = 0
    skipped_no_cues = 0

    for entry in entries:
        key = entry["key"]
        audio_path = _find_audio_for_key(audio_dir, key)
        if audio_path is None:
            skipped_missing_audio += 1
            continue

        cue_points = _cue_points_from_sections(entry["sections"], key)
        if not cue_points:
            skipped_no_cues += 1
            continue

        downbeats = _load_downbeats(raveform_dir, key)

        tracks.append(
            Track(
                track_id=key,
                name=entry.get("title", key),
                artist="",
                bpm=float(entry["average_bpm"]),
                location=str(audio_path),
                filename=audio_path.name,
                audio_path=str(audio_path),
                tempo_entries=[],
                cue_points=cue_points,
                downbeats=downbeats,
                fold=entry.get("fold"),
            )
        )

    return tracks, skipped_missing_audio, skipped_no_cues
