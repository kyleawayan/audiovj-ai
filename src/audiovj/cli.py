from pathlib import Path

import typer

from audiovj.config import FEATURES_DIR, TRACKS_DIR
from audiovj.data.rekordbox import (
    build_downbeat_times,
    load_tracks,
    match_audio_files,
    parse_rekordbox_xml,
    save_tracks,
    Track,
)

app = typer.Typer(name="audiovj", help="DJ phrase detection data pipeline")


@app.callback()
def main() -> None:
    pass


@app.command()
def import_rekordbox(
    xml_path: Path = typer.Argument(help="Path to Rekordbox XML export"),
    audio_folder: Path = typer.Argument(
        help="Folder containing audio files to match against"
    ),
    playlist: str = typer.Option(
        "audiovj", help="Rekordbox playlist name to import from"
    ),
) -> None:
    """Import tracks from a Rekordbox XML export."""
    if not xml_path.exists():
        typer.echo(f"Error: XML file not found: {xml_path}")
        raise typer.Exit(1)
    if not audio_folder.is_dir():
        typer.echo(f"Error: Audio folder not found: {audio_folder}")
        raise typer.Exit(1)

    typer.echo(f"Parsing Rekordbox XML: {xml_path}")
    tracks = parse_rekordbox_xml(xml_path, playlist=playlist)
    typer.echo(f"Found {len(tracks)} local track(s) in playlist '{playlist}'")

    if not tracks:
        raise typer.Exit(0)

    typer.echo(f"Matching audio files in: {audio_folder}")
    matched, unmatched = match_audio_files(tracks, audio_folder)

    labeled = sum(1 for t in matched if t.cue_points)

    typer.echo(f"\nSummary:")
    typer.echo(f"  Matched to audio files: {len(matched)}")
    typer.echo(f"  Unmatched (no audio):   {unmatched}")
    typer.echo(f"  With phrase labels:     {labeled}")

    if matched:
        save_tracks(matched, TRACKS_DIR)
        typer.echo(f"\nSaved {len(matched)} track(s) to {TRACKS_DIR}/")


@app.command()
def preprocess() -> None:
    """Extract mel-spectrograms and generate training labels for all imported tracks."""
    from audiovj.data.features import preprocess_track

    tracks = load_tracks(TRACKS_DIR)
    if not tracks:
        typer.echo("No imported tracks found. Run import-rekordbox first.")
        raise typer.Exit(1)

    typer.echo(f"Preprocessing {len(tracks)} track(s)...")

    total_windows = 0
    for i, track in enumerate(tracks, 1):
        if track.audio_path is None:
            typer.echo(f"  [{i}/{len(tracks)}] {track.name} — skipped (no audio)")
            continue

        typer.echo(f"  [{i}/{len(tracks)}] {track.name}...", nl=False)
        n = preprocess_track(track, FEATURES_DIR)
        total_windows += n
        typer.echo(f" {n} windows")

    labeled = sum(1 for t in tracks if t.cue_points)
    typer.echo(f"\nDone: {total_windows} total windows from {len(tracks)} track(s)")
    typer.echo(f"Tracks with phrase labels: {labeled}")


@app.command()
def inspect(
    track_id: str = typer.Argument(help="Track ID to inspect"),
) -> None:
    """Inspect a track's metadata, beat grid, cue points, and labels."""
    from safetensors.torch import load_file

    from audiovj.data.dataset import generate_labels

    track_path = TRACKS_DIR / f"{track_id}.json"
    if not track_path.exists():
        typer.echo(f"Error: Track not found: {track_id}")
        typer.echo(f"Available tracks:")
        for p in sorted(TRACKS_DIR.glob("*.json")):
            t = Track.model_validate_json(p.read_text())
            typer.echo(f"  {t.track_id}  {t.artist} - {t.name}")
        raise typer.Exit(1)

    track = Track.model_validate_json(track_path.read_text())

    typer.echo(f"Track: {track.artist} - {track.name}")
    typer.echo(f"  ID:       {track.track_id}")
    typer.echo(f"  BPM:      {track.bpm}")
    typer.echo(f"  Audio:    {track.audio_path or 'not matched'}")
    typer.echo(f"  Filename: {track.filename}")

    # Beat grid
    typer.echo(f"\nBeat Grid ({len(track.tempo_entries)} entries):")
    for te in track.tempo_entries[:5]:
        typer.echo(
            f"  {te.start_time:.3f}s  BPM={te.bpm}  "
            f"{te.time_signature}  beat={te.beat_position}"
        )
    if len(track.tempo_entries) > 5:
        typer.echo(f"  ... and {len(track.tempo_entries) - 5} more")

    # Cue points
    if track.cue_points:
        typer.echo(f"\nCue Points ({len(track.cue_points)}):")
        for cp in track.cue_points:
            mins = int(cp.start_time // 60)
            secs = cp.start_time % 60
            typer.echo(
                f"  {mins}:{secs:05.2f}  {cp.phrase_type:<12}  "
                f"RGB=({cp.color[0]}, {cp.color[1]}, {cp.color[2]})"
            )
    else:
        typer.echo("\nNo cue points (unlabeled track)")

    # Preprocessed features
    features_path = FEATURES_DIR / f"{track_id}.safetensors"
    if features_path.exists():
        data = load_file(str(features_path))
        windows = data["windows"]
        typer.echo(f"\nPreprocessed Features:")
        typer.echo(f"  Windows: {windows.shape[0]}")
        typer.echo(f"  Shape:   {list(windows.shape)}")
    else:
        typer.echo("\nNo preprocessed features (run preprocess)")

    # Labels
    if track.cue_points:
        if not track.audio_path or not Path(track.audio_path).exists():
            typer.echo("\nCannot generate labels — audio file not found")
            raise typer.Exit(1)

        from audiovj.data.features import load_audio

        _, duration = load_audio(Path(track.audio_path))
        downbeats = build_downbeat_times(track, total_duration=duration)
        labels = generate_labels(track, downbeats)
        if labels:
            typer.echo(f"\nLabels (first 10 of {len(labels)}):")
            for lbl in labels[:10]:
                t = lbl["downbeat_time"]
                mins = int(t // 60)
                secs = t % 60
                typer.echo(
                    f"  {mins}:{secs:05.2f}  "
                    f"current={lbl['current_phrase']:<12}  "
                    f"next={lbl['next_phrase']:<12}  "
                    f"beats_until={lbl['beats_until']:.0f}"
                )
