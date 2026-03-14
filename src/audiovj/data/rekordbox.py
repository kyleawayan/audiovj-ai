"""Rekordbox XML parser and audio file matching."""

import json
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

from pydantic import BaseModel

from audiovj.config import color_to_phrase


class TempoEntry(BaseModel):
    start_time: float  # seconds (Inizio attribute)
    bpm: float
    time_signature: str  # e.g. "4/4" (Metro attribute)
    beat_position: int  # 1-4 (Battito attribute)


class CuePoint(BaseModel):
    start_time: float  # seconds (Start attribute)
    color: tuple[int, int, int]  # (R, G, B)
    phrase_type: str  # resolved from color


class Track(BaseModel):
    track_id: str
    name: str
    artist: str
    bpm: float
    location: str  # original URL-decoded path from XML
    filename: str  # extracted filename for matching
    audio_path: str | None = None  # resolved local path after matching
    tempo_entries: list[TempoEntry]
    cue_points: list[CuePoint]  # sorted by start_time


def parse_rekordbox_xml(
    xml_path: Path, playlist: str = "audiovj"
) -> list[Track]:
    """Parse Rekordbox XML, returning only tracks from the specified playlist
    that have local file paths."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Build a lookup of TrackID -> TRACK element from COLLECTION
    collection: dict[str, ET.Element] = {}
    for track_el in root.findall(".//COLLECTION/TRACK"):
        tid = track_el.get("TrackID", "")
        if tid:
            collection[tid] = track_el

    # Find the playlist and get its track keys
    playlist_keys: list[str] = []
    for node in root.findall(".//PLAYLISTS//NODE"):
        if node.get("Name") == playlist and node.get("Type") == "1":
            for track_ref in node.findall("TRACK"):
                key = track_ref.get("Key", "")
                if key:
                    playlist_keys.append(key)
            break

    if not playlist_keys:
        print(f"Warning: Playlist '{playlist}' not found or empty")
        return []

    tracks: list[Track] = []
    skipped_streaming = 0

    for key in playlist_keys:
        track_el = collection.get(key)
        if track_el is None:
            continue

        location_raw = track_el.get("Location", "")

        # Filter out streaming tracks
        if not location_raw.startswith("file://localhost/"):
            skipped_streaming += 1
            continue

        # URL-decode and extract local path
        # file://localhost/Users/kyle/... -> /Users/kyle/...
        local_path = unquote(location_raw.removeprefix("file://localhost"))
        filename = Path(local_path).name

        # Parse tempo entries
        tempo_entries: list[TempoEntry] = []
        for tempo_el in track_el.findall("TEMPO"):
            tempo_entries.append(
                TempoEntry(
                    start_time=float(tempo_el.get("Inizio", "0")),
                    bpm=float(tempo_el.get("Bpm", "0")),
                    time_signature=tempo_el.get("Metro", "4/4"),
                    beat_position=int(tempo_el.get("Battito", "1")),
                )
            )

        # Parse cue points — only keep those with recognized vocabulary colors
        cue_points: list[CuePoint] = []
        for pm_el in track_el.findall("POSITION_MARK"):
            r_str = pm_el.get("Red")
            g_str = pm_el.get("Green")
            b_str = pm_el.get("Blue")

            # Skip entries without color attributes
            if r_str is None or g_str is None or b_str is None:
                continue

            r, g, b = int(r_str), int(g_str), int(b_str)
            phrase = color_to_phrase(r, g, b)

            if phrase is None:
                print(
                    f"Warning: Unrecognized cue color ({r}, {g}, {b}) "
                    f"in track '{track_el.get('Name', '')}', skipping"
                )
                continue

            cue_points.append(
                CuePoint(
                    start_time=float(pm_el.get("Start", "0")),
                    color=(r, g, b),
                    phrase_type=phrase,
                )
            )

        cue_points.sort(key=lambda c: c.start_time)

        tracks.append(
            Track(
                track_id=track_el.get("TrackID", ""),
                name=track_el.get("Name", ""),
                artist=track_el.get("Artist", ""),
                bpm=float(track_el.get("AverageBpm", "0")),
                location=local_path,
                filename=filename,
                tempo_entries=tempo_entries,
                cue_points=cue_points,
            )
        )

    if skipped_streaming > 0:
        print(f"Skipped {skipped_streaming} streaming track(s)")

    return tracks


def match_audio_files(
    tracks: list[Track], audio_folder: Path
) -> tuple[list[Track], int]:
    """Match tracks to audio files by filename. Returns (matched_tracks, unmatched_count)."""
    # Build filename -> path lookup (recursive scan)
    # Normalize to NFC for consistent matching (macOS uses NFD for filenames)
    extensions = {".wav", ".flac", ".mp3", ".m4a", ".aif", ".aiff"}
    file_lookup: dict[str, Path] = {}
    for f in audio_folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in extensions:
            normalized_name = unicodedata.normalize("NFC", f.name)
            if normalized_name in file_lookup:
                print(f"Warning: Duplicate filename '{f.name}', using {f}")
            file_lookup[normalized_name] = f

    matched: list[Track] = []
    unmatched = 0

    for track in tracks:
        lookup_name = unicodedata.normalize("NFC", track.filename)
        path = file_lookup.get(lookup_name)
        if path is not None:
            track.audio_path = str(path)
            matched.append(track)
        else:
            unmatched += 1
            print(f"  No audio file found for: {track.filename}")

    return matched, unmatched


def build_downbeat_times(
    track: Track, total_duration: float | None = None
) -> list[float]:
    """Generate downbeat (beat 1) timestamps from the track's beat grid.

    Uses the first TEMPO entry to determine the starting phase and BPM.
    For V1, assumes constant BPM (uses first entry only).
    """
    if not track.tempo_entries:
        return []

    first = track.tempo_entries[0]
    beat_duration = 60.0 / first.bpm

    # Find the first downbeat: count forward from the TEMPO entry's
    # beat_position to reach beat 1 of the next bar.
    # Battito=1 means we're already on a downbeat.
    # Battito=3 means we need 2 more beats to reach the next beat 1.
    beats_to_downbeat = (4 - first.beat_position + 1) % 4
    first_downbeat = first.start_time + beats_to_downbeat * beat_duration

    # Use track duration from BPM * total_time, or estimate from last cue/tempo
    if total_duration is None:
        # Estimate: use a generous upper bound
        total_duration = 600.0  # 10 minutes max

    bar_duration = 4 * beat_duration
    downbeats: list[float] = []
    t = first_downbeat
    while t < total_duration:
        downbeats.append(t)
        t += bar_duration

    return downbeats


def save_tracks(tracks: list[Track], output_dir: Path) -> None:
    """Save each track as a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        path = output_dir / f"{track.track_id}.json"
        path.write_text(track.model_dump_json(indent=2))


def load_tracks(tracks_dir: Path) -> list[Track]:
    """Load all track JSON files from a directory."""
    tracks: list[Track] = []
    for path in sorted(tracks_dir.glob("*.json")):
        data = json.loads(path.read_text())
        tracks.append(Track.model_validate(data))
    return tracks
