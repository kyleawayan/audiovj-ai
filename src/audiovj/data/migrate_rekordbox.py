"""Migrate pre-allin1 Rekordbox-format Track JSONs into the new Raveform vocab.

Archived JSONs were hand-labeled by the user in Rekordbox. Only `drop` markers
are kept (everything else is too sparse / generic for supervision). The output
is a validation set: a way to check whether the Raveform-trained model agrees
with the user's own ear on the user's own music.

Audio path translation is optional: pass `audio_path_from` / `audio_path_to`
if the archived JSONs reference a different OS's filesystem (e.g. Mac paths
that need rewriting for WSL).
"""

import json
from pathlib import Path

from audiovj.data.rekordbox import CuePoint, Track

KEEP_LABELS = {"drop"}  # vocab in PHRASE_TYPES we accept from old labels
SKIP_TRACK_NAMES = {"live_my_life"}  # programmatic test data, not user-labeled


def _resolve_audio_path(
    raw: str | None,
    audio_path_from: str | None,
    audio_path_to: str | None,
) -> str | None:
    if not raw:
        return None
    if audio_path_from and audio_path_to and raw.startswith(audio_path_from):
        return audio_path_to + raw[len(audio_path_from):]
    return raw


def migrate_track(
    raw: dict,
    audio_path_from: str | None = None,
    audio_path_to: str | None = None,
) -> Track | None:
    """Convert one archived Rekordbox Track dict → new-format Track.
    Returns None if the track is skipped (test data, no usable cues)."""
    if raw.get("name") in SKIP_TRACK_NAMES:
        return None

    filtered_cues: list[CuePoint] = []
    for c in raw.get("cue_points", []):
        if c.get("phrase_type") not in KEEP_LABELS:
            continue
        filtered_cues.append(
            CuePoint(
                start_time=float(c["start_time"]),
                hotcue=int(c.get("hotcue", -1)),
                phrase_type=c["phrase_type"],
            )
        )

    if not filtered_cues:
        return None

    audio_path = _resolve_audio_path(raw.get("audio_path"), audio_path_from, audio_path_to)

    return Track(
        track_id=raw["track_id"],
        name=raw.get("name", ""),
        artist=raw.get("artist", ""),
        bpm=float(raw.get("bpm", 0)),
        location=audio_path or "",
        filename=raw.get("filename", ""),
        audio_path=audio_path,
        tempo_entries=raw.get("tempo_entries", []),
        cue_points=filtered_cues,
        downbeats=None,
        fold=None,
    )


def migrate_folder(
    source_dir: Path,
    target_dir: Path,
    audio_path_from: str | None = None,
    audio_path_to: str | None = None,
) -> tuple[int, int, int]:
    """Read every *.json in source_dir, write migrated Tracks to target_dir.

    Returns (kept, skipped, audio_missing).
    """
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped = audio_missing = 0

    for src in sorted(source_dir.glob("*.json")):
        raw = json.loads(src.read_text())
        track = migrate_track(raw, audio_path_from, audio_path_to)
        if track is None:
            skipped += 1
            continue

        if track.audio_path and not Path(track.audio_path).exists():
            audio_missing += 1
            print(f"  warn: audio missing for '{track.name}': {track.audio_path}")

        out = target_dir / f"{track.track_id}.json"
        out.write_text(track.model_dump_json(indent=2))
        kept += 1

    return kept, skipped, audio_missing
