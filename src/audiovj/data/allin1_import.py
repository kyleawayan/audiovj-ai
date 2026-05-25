"""Import tracks from offline `allin1` structure analyses.

`allin1.analyze()` writes one JSON per audio file with bpm/beats/downbeats/segments.
This module wraps that step over a folder and converts the JSON output into
`Track` objects compatible with the rest of the pipeline.
"""

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

from audiovj.config import PHRASE_TYPES
from audiovj.data.rekordbox import CuePoint, Track

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aif", ".aiff", ".ogg", ".aac"}

# allin1 segments we ignore: sub-second markers, not musical sections.
SKIP_LABELS = {"start", "end"}


def find_audio_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    return sorted(files)


def _convert_to_wav(src: Path, dst: Path) -> bool:
    """Decode `src` to a stereo PCM WAV at `dst`. Returns True on success."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "2", str(dst)],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.decode('utf-8', errors='replace')[:200]}", flush=True)
        return False
    return True


def analyze_folder(
    audio_folder: Path,
    struct_dir: Path,
    force: bool = False,
) -> tuple[int, int, int]:
    """Run `allin1.analyze` on every audio file under `audio_folder`.

    Non-WAV inputs are first decoded to a temporary stereo PCM WAV — the allin1
    author observed 20–40ms beat-tracking drift on MP3/M4A inputs (per README),
    which is meaningful at the 70ms beat-tolerance level. Writes per-track JSONs
    to `struct_dir`. Returns (processed, skipped, failed).
    """
    import allin1

    audio_folder = audio_folder.resolve()
    struct_dir = struct_dir.resolve()
    struct_dir.mkdir(parents=True, exist_ok=True)

    files = find_audio_files(audio_folder)
    if not files:
        print(f"No audio files found under {audio_folder}")
        return 0, 0, 0

    wav_tmp_root = Path(tempfile.gettempdir()) / "audiovj-allin1"
    wav_tmp_root.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0
    for i, path in enumerate(files, 1):
        json_out = struct_dir / f"{path.stem}.json"
        if json_out.exists() and not force:
            print(f"  [{i}/{len(files)}] {path.name} — skipped (cached)", flush=True)
            skipped += 1
            continue

        # allin1's README recommends WAV input; the package's MP3 decoder can drift
        # beat timing by 20–40 ms. Pre-convert with ffmpeg, analyze, then discard.
        if path.suffix.lower() == ".wav":
            wav_path = path
            tmp_wav = None
        else:
            tmp_wav = wav_tmp_root / f"{path.stem}.wav"
            print(f"  [{i}/{len(files)}] {path.name} — converting to WAV...", flush=True)
            if not _convert_to_wav(path, tmp_wav):
                print(f"  [{i}/{len(files)}] {path.name} — FAILED (ffmpeg)", flush=True)
                failed += 1
                continue
            wav_path = tmp_wav

        t0 = time.time()
        try:
            allin1.analyze(
                str(wav_path),
                out_dir=str(struct_dir),
                keep_byproducts=False,
            )
            elapsed = time.time() - t0
            print(f"  [{i}/{len(files)}] {path.name} — done ({elapsed:.1f}s)", flush=True)
            processed += 1
        except Exception as e:
            print(f"  [{i}/{len(files)}] {path.name} — FAILED: {e}", flush=True)
            failed += 1
        finally:
            if tmp_wav is not None and tmp_wav.exists():
                tmp_wav.unlink()

    return processed, skipped, failed


def _dedupe_consecutive(cues: list[CuePoint]) -> list[CuePoint]:
    """Keep only the earliest of each run of same-phrase cues."""
    out: list[CuePoint] = []
    last_phrase: str | None = None
    for c in cues:
        if c.phrase_type == last_phrase:
            continue
        out.append(c)
        last_phrase = c.phrase_type
    return out


def parse_allin1_json(json_path: Path, audio_path: Path) -> Track:
    """Convert one allin1 analysis JSON into a Track."""
    data = json.loads(json_path.read_text())

    cue_points: list[CuePoint] = []
    for seg in data.get("segments", []):
        label = seg.get("label", "")
        if label in SKIP_LABELS:
            continue
        if label not in PHRASE_TYPES:
            print(f"  warn: unknown allin1 label '{label}' in {json_path.name}, skipping cue")
            continue
        cue_points.append(
            CuePoint(
                start_time=float(seg["start"]),
                hotcue=-1,
                phrase_type=label,
            )
        )
    cue_points.sort(key=lambda c: c.start_time)
    cue_points = _dedupe_consecutive(cue_points)

    track_id = hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()[:12]

    return Track(
        track_id=track_id,
        name=audio_path.stem,
        artist="",
        bpm=float(data.get("bpm", 0)),
        location=str(audio_path),
        filename=audio_path.name,
        audio_path=str(audio_path),
        tempo_entries=[],
        cue_points=cue_points,
        downbeats=list(data.get("downbeats", [])),
    )


def import_folder(audio_folder: Path, struct_dir: Path) -> list[Track]:
    """Pair audio files with their allin1 JSONs and produce Tracks.

    Audio files without a matching JSON are skipped (run analyze_folder first).
    """
    audio_folder = audio_folder.resolve()
    struct_dir = struct_dir.resolve()

    files = find_audio_files(audio_folder)
    tracks: list[Track] = []
    seen_stems: dict[str, Path] = {}
    missing = 0

    for path in files:
        stem = path.stem
        if stem in seen_stems:
            # Disambiguate duplicate stems by hashing the relative path.
            suffix = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:6]
            print(f"  warn: duplicate stem '{stem}' (also at {seen_stems[stem]}); tagging this one with suffix {suffix}")
            # We can't rename the JSON the analyzer wrote; assume the first wins
            # and the duplicate is silently dropped. Worth revisiting if it bites.
            continue
        seen_stems[stem] = path

        json_path = struct_dir / f"{stem}.json"
        if not json_path.exists():
            missing += 1
            continue

        try:
            tracks.append(parse_allin1_json(json_path, path))
        except Exception as e:
            print(f"  failed to parse {json_path.name}: {e}")

    if missing:
        print(f"  {missing} audio file(s) had no matching allin1 JSON")
    return tracks
