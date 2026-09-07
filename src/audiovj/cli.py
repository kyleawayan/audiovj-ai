from pathlib import Path

import typer

from audiovj.config import (
    FEATURES_DIR,
    MODELS_DIR,
    PHRASE_TYPES,
    TRACKS_DIR,
    TRACKS_VALIDATION_DIR,
)
from audiovj.data.rekordbox import (
    Track,
    build_downbeat_times,
    load_tracks,
    save_tracks,
)

DEFAULT_RAVEFORM_DIR = Path("data/raveform")
DEFAULT_AUDIO_DIR = Path("data/audio")

app = typer.Typer(name="audiovj", help="DJ phrase detection data pipeline")


@app.callback()
def main() -> None:
    pass


@app.command()
def raveform_import(
    raveform_dir: Path = typer.Option(DEFAULT_RAVEFORM_DIR, help="Root of Raveform dataset"),
    audio_dir: Path = typer.Option(DEFAULT_AUDIO_DIR, help="Where audio files live"),
    limit: int = typer.Option(None, help="Cap number of tracks (smoke runs)"),
) -> None:
    """Build Track records from Raveform metadata + locally-available audio."""
    from audiovj.data.raveform_import import import_raveform

    tracks, missing, no_cues = import_raveform(raveform_dir, audio_dir, limit=limit)
    typer.echo(
        f"Imported {len(tracks)} tracks  "
        f"(skipped {missing} missing audio, {no_cues} without cues)"
    )

    if not tracks:
        raise typer.Exit(1)

    save_tracks(tracks, TRACKS_DIR)
    typer.echo(f"Saved to {TRACKS_DIR}/")


@app.command()
def migrate_rekordbox_labels(
    source_dir: Path = typer.Argument(help="Old Track JSON dir from `experiment/binary-drop-detection` branch"),
    target_dir: Path = typer.Option(TRACKS_VALIDATION_DIR, help="Where validation Tracks go"),
    audio_path_from: str = typer.Option("", help="If audio_path strings need prefix substitution: replace this..."),
    audio_path_to: str = typer.Option("", help="...with this (e.g. --from /old/prefix --to /new/prefix)"),
) -> None:
    """Migrate archived Rekordbox-format Tracks into a Raveform-vocab validation set."""
    from audiovj.data.migrate_rekordbox import migrate_folder

    if not source_dir.exists():
        typer.echo(f"Error: source dir not found: {source_dir}")
        raise typer.Exit(1)

    kept, skipped, audio_missing = migrate_folder(
        source_dir, target_dir,
        audio_path_from=audio_path_from or None,
        audio_path_to=audio_path_to or None,
    )
    typer.echo(
        f"Migrated {kept} tracks  (skipped {skipped} with no usable cues)  "
        f"({audio_missing} had missing audio locally)"
    )
    typer.echo(f"Saved to {target_dir}/")


@app.command()
def preprocess() -> None:
    """Extract mel-spectrograms and generate training labels for all imported tracks."""
    from audiovj.data.features import preprocess_track

    tracks = load_tracks(TRACKS_DIR)
    if not tracks:
        typer.echo("No imported tracks found. Run raveform-import first.")
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
        typer.echo("Available tracks:")
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

    # Cue points
    if track.cue_points:
        typer.echo(f"\nCue Points ({len(track.cue_points)}):")
        for cp in track.cue_points:
            mins = int(cp.start_time // 60)
            secs = cp.start_time % 60
            typer.echo(
                f"  {mins}:{secs:05.2f}  {cp.phrase_type:<12}  hotcue={cp.hotcue}"
            )
    else:
        typer.echo("\nNo cue points (unlabeled track)")

    # Preprocessed features
    features_path = FEATURES_DIR / f"{track_id}.safetensors"
    if features_path.exists():
        data = load_file(str(features_path))
        windows = data["windows"]
        typer.echo("\nPreprocessed Features:")
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
        all_labels = generate_labels(track, downbeats)
        labels = [lbl for lbl in all_labels if lbl is not None]
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


@app.command()
def train(
    epochs: int = typer.Option(50, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Batch size"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
    grad_clip: float = typer.Option(1.0, help="Gradient norm clip"),
    lr_patience: int = typer.Option(5, help="ReduceLROnPlateau patience (epochs)"),
    lr_factor: float = typer.Option(0.5, help="ReduceLROnPlateau decay factor"),
    class_weight_cap: float = typer.Option(5.0, help="Cap on inverse-freq class weights"),
    weight_power: float = typer.Option(
        1.0, help="Class-weight exponent: 1=inverse-freq, 0.5=sqrt (gentler on majority), 0=uniform"
    ),
    f1_save_threshold: float = typer.Option(
        0.0, help="Minimum macro-F1 before checkpoint saves (0 = always keep best)"
    ),
    num_workers: int = typer.Option(4, help="DataLoader worker processes (RAM-bound)"),
    prefetch_factor: int = typer.Option(4, help="Batches prefetched per worker"),
    log_interval: int = typer.Option(200, help="Log in-epoch progress every N batches"),
    balance: str = typer.Option(
        "sampler", help="Class balancing (pick ONE): sampler | loss | none"
    ),
    dropout: float = typer.Option(0.3, help="Dropout (LSTM inter-layer + pre-head)"),
    weight_decay: float = typer.Option(1e-4, help="Adam weight decay (L2 regularization)"),
) -> None:
    """Train the phrase predictor model."""
    from audiovj.training import train_model

    train_model(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_clip=grad_clip,
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        class_weight_cap=class_weight_cap,
        weight_power=weight_power,
        f1_save_threshold=f1_save_threshold,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        log_interval=log_interval,
        balance=balance,
        dropout=dropout,
        weight_decay=weight_decay,
    )


@app.command()
def evaluate(
    checkpoint: str = typer.Option(
        None, help="Path to model checkpoint (default: data/models/phrase_predictor.safetensors)"
    ),
) -> None:
    """Evaluate the trained model on the validation split."""
    from audiovj.evaluate import evaluate_model

    metrics = evaluate_model(checkpoint=checkpoint)

    if "error" in metrics:
        typer.echo(f"Error: {metrics['error']}")
        raise typer.Exit(1)

    typer.echo(f"Evaluation ({metrics['total_samples']} samples):")
    typer.echo(f"  Next phrase accuracy:    {metrics['next_phrase_accuracy']:.1f}%")
    typer.echo(f"  Current phrase accuracy: {metrics['current_phrase_accuracy']:.1f}%")
    typer.echo(f"  Beats-until MAE (all):   {metrics['beats_until_mae']:.2f}")
    typer.echo(
        f"  Beats-until MAE (trans): {metrics['beats_until_mae_transition']:.2f}"
        f"  ({metrics['transition_samples']} transition samples)"
    )
    typer.echo(f"  Flip-flop rate:          {metrics['flip_flop_rate']:.1f}%")
    typer.echo(f"  Macro-F1 (load-bearing): {metrics['macro_f1_load_bearing']:.1f}%")

    typer.echo("\nPer-class (current phrase)      acc    prec    rec     F1")
    prec = metrics["per_class_precision"]
    rec = metrics["per_class_recall"]
    f1 = metrics["per_class_f1"]
    for phrase, acc in metrics["per_class_accuracy"].items():
        typer.echo(
            f"  {phrase:<12} {acc:6.1f}% {prec[phrase]:6.1f}% "
            f"{rec[phrase]:6.1f}% {f1[phrase]:6.1f}%"
        )


@app.command()
def evaluate_seq(
    checkpoint: str = typer.Option(None, help="Seq checkpoint (default data/models/seq_unified.safetensors)"),
    onset_threshold: float = typer.Option(0.30, help="LB onset threshold (locked operating point)"),
    fold: int = typer.Option(None, help="Restrict to one Raveform fold (e.g. 7 for held-out certification)"),
    limit: int = typer.Option(None, help="Cap number of tracks"),
) -> None:
    """Offline twin of run-live: stateful seq inference + onset cueing on labeled
    tracks. Reports the locked operating point (LB transition recall etc.)."""
    from audiovj.evaluate import evaluate_seq as _evaluate_seq

    r = _evaluate_seq(checkpoint=checkpoint, onset_threshold=onset_threshold, fold=fold, limit=limit)
    if "error" in r:
        typer.echo(f"Error: {r['error']}")
        raise typer.Exit(1)
    typer.echo(f"Seq pipeline (onset@{onset_threshold}) on {r['n_tracks']} tracks"
               + (f" (fold {r['fold']})" if r["fold"] is not None else "") + ":")
    typer.echo(f"  LB transition recall : {r['lb_transition_recall']:.1f}%")
    typer.echo(f"  fire precision       : {r['fire_precision']:.1f}%")
    typer.echo(f"  matched latency      : {r['matched_latency_beats']:.1f} beats  (fires {r['fires']})")
    typer.echo("  per-class recall     : "
               + "  ".join(f"{k} {v:.0f}%" for k, v in r["per_class_recall"].items()))
    typer.echo(f"  drop events          : {r['drop_start_events']} starts / {r['drop_end_events']} ends")


@app.command()
def evaluate_pipeline(
    checkpoint: str = typer.Option(None, help="Path to model checkpoint"),
    correction_threshold: float = typer.Option(0.5, help="Min confidence for phrase correction"),
    transition_beats: float = typer.Option(4.0, help="Beats-until threshold for transition"),
    anticipate_beats: float = typer.Option(8.0, help="Beats-until threshold for anticipation"),
    latch_after: int = typer.Option(2, help="Consecutive agreements before latching a countdown"),
    sticky_beats: float = typer.Option(32.0, help="Sticky-hold window after a transition (beats)"),
    warmup_beats: float = typer.Option(16.0, help="Warmup window before SM decisions engage (beats)"),
    limit: int = typer.Option(
        None, help="Cap number of tracks (quick eval; default all). Each track re-decodes its WAV."
    ),
) -> None:
    """Evaluate model + State Manager on labeled tracks (e2e pipeline simulation)."""
    from audiovj.evaluate import evaluate_pipeline as _evaluate_pipeline

    results = _evaluate_pipeline(
        checkpoint=checkpoint,
        correction_threshold=correction_threshold,
        transition_beats=transition_beats,
        anticipate_beats=anticipate_beats,
        latch_after=latch_after,
        sticky_beats=sticky_beats,
        warmup_beats=warmup_beats,
        limit=limit,
    )

    if not results or "error" in results[0]:
        typer.echo(f"Error: {results[0].get('error', 'No results')}")
        raise typer.Exit(1)

    agg_raw = 0.0
    agg_sm = 0.0
    agg_transitions = 0
    agg_actual = 0
    agg_corrections = 0
    agg_downbeats = 0
    agg_detected = 0.0
    agg_timing_errors: list[float] = []
    agg_matched: list[tuple[float, int]] = []
    agg_cd_n = 0
    agg_cd_mae = 0.0
    agg_cd_corr = 0.0
    agg_cd_mono = 0.0

    for r in results:
        typer.echo(f"\n{r['name']}")
        typer.echo(f"  Raw model accuracy:    {r['raw_accuracy']:5.1f}%")
        typer.echo(f"  State Manager accuracy:{r['sm_accuracy']:5.1f}%")
        typer.echo(
            f"  Changes: {r['transitions_fired']} transitions + {r['corrections']} corrections, "
            f"{r['actual_transitions']} actual boundaries "
            f"({r['transition_recall']:.0f}% recall, {r['transition_precision']:.0f}% precision)"
        )
        if r["detected_boundaries"] > 0:
            typer.echo(
                f"  Matched latency: {r['matched_latency']:.1f} beats "
                f"(real cueing precision on {r['detected_boundaries']} detected boundaries)"
            )
        if r["transitions_fired"] > 0 or r["corrections"] > 0:
            typer.echo(f"  Fire->boundary (inflated): {r['mean_timing_error']:.1f} beats mean")
        if r["countdown_samples"] > 0:
            typer.echo(
                f"  Countdown: MAE {r['countdown_mae']:.1f} beats, "
                f"corr {r['countdown_corr']:+.2f}, "
                f"monotonicity {r['countdown_monotonicity'] * 100:.0f}% "
                f"({r['countdown_samples']} samples)"
            )

        agg_raw += r["raw_accuracy"] * r["labeled_downbeats"]
        agg_sm += r["sm_accuracy"] * r["labeled_downbeats"]
        agg_downbeats += r["labeled_downbeats"]
        agg_transitions += r["transitions_fired"]
        agg_actual += r["actual_transitions"]
        agg_corrections += r["corrections"]
        agg_detected += r["transition_recall"] / 100 * r["actual_transitions"]
        if r["transitions_fired"] > 0 or r["corrections"] > 0:
            agg_timing_errors.append(r["mean_timing_error"])
        if r["detected_boundaries"] > 0:
            agg_matched.append((r["matched_latency"], r["detected_boundaries"]))
        if r["countdown_samples"] > 0:
            n = r["countdown_samples"]
            agg_cd_n += n
            agg_cd_mae += r["countdown_mae"] * n
            agg_cd_corr += r["countdown_corr"] * n
            agg_cd_mono += r["countdown_monotonicity"] * n

    typer.echo(f"\n{'─' * 50}")
    typer.echo(f"Aggregate ({len(results)} tracks, {agg_downbeats} downbeats):")
    typer.echo(
        f"  Raw accuracy: {agg_raw / max(agg_downbeats, 1):.1f}%  →  "
        f"SM accuracy: {agg_sm / max(agg_downbeats, 1):.1f}%"
    )
    typer.echo(
        f"  Changes: {agg_transitions} transitions + {agg_corrections} corrections, "
        f"{agg_actual} actual boundaries "
        f"({agg_detected / max(agg_actual, 1) * 100:.0f}% recall)"
    )
    typer.echo(f"  Correction rate: {agg_corrections / max(agg_downbeats, 1):.2f}/downbeat")
    if agg_matched:
        wn = sum(n for _, n in agg_matched)
        ml = sum(v * n for v, n in agg_matched) / max(wn, 1)
        typer.echo(f"  Matched latency: {ml:.1f} beats (real cueing precision, {wn} detected boundaries)")
    if agg_timing_errors:
        typer.echo(f"  Fire->boundary (inflated by mid-phrase fires): {sum(agg_timing_errors) / len(agg_timing_errors):.1f} beats")
    if agg_cd_n > 0:
        typer.echo(
            f"  Countdown quality: MAE {agg_cd_mae / agg_cd_n:.1f} beats, "
            f"corr {agg_cd_corr / agg_cd_n:+.2f}, "
            f"monotonicity {agg_cd_mono / agg_cd_n * 100:.0f}% "
            f"({agg_cd_n} samples)"
        )


@app.command(name="validate-on-old-binary-drop-detection-see-experiment-binary-drop-detection-branch")
def validate_on_old_binary_drop_detection_see_experiment_binary_drop_detection_branch() -> None:
    """Pointer: this lived in the pre-Raveform manual-drop-label era. See branch `experiment/binary-drop-detection`."""
    typer.echo("See branch: experiment/binary-drop-detection")


@app.command()
def predict_folder(
    folder: Path = typer.Argument(help="Folder of audio files (recursive)"),
    out_dir: Path = typer.Option(Path("data/predictions"), help="Where to write per-track JSON"),
    checkpoint: Path = typer.Option(None, help="Path to model checkpoint"),
    correction_threshold: float = typer.Option(0.5, help="Min confidence for SM phrase correction"),
    transition_beats: float = typer.Option(4.0, help="SM transition beats threshold"),
    anticipate_beats: float = typer.Option(8.0, help="SM anticipation beats threshold"),
    latch_after: int = typer.Option(2, help="Consecutive agreements before latching a countdown"),
    sticky_beats: float = typer.Option(32.0, help="Sticky-hold window after a transition (beats)"),
    warmup_beats: float = typer.Option(16.0, help="Warmup window before SM decisions engage (beats)"),
    force: bool = typer.Option(False, help="Re-predict files that already have output"),
) -> None:
    """Run model + State Manager on every audio file in a folder; dump predictions to JSON."""
    from audiovj.predict_folder import predict_folder as _predict_folder

    if not folder.is_dir():
        typer.echo(f"Error: not a directory: {folder}")
        raise typer.Exit(1)

    processed, skipped, failed = _predict_folder(
        folder=folder,
        out_dir=out_dir,
        checkpoint=checkpoint,
        correction_threshold=correction_threshold,
        transition_beats=transition_beats,
        anticipate_beats=anticipate_beats,
        latch_after=latch_after,
        sticky_beats=sticky_beats,
        warmup_beats=warmup_beats,
        skip_existing=not force,
    )
    typer.echo(f"\nDone: {processed} processed, {skipped} skipped, {failed} failed")
    typer.echo(f"Predictions written to: {out_dir}/")
    if failed:
        raise typer.Exit(1)


@app.command()
def predict_file(
    track_id: str = typer.Argument(help="Track ID to run predictions on"),
    checkpoint: str = typer.Option(None, help="Path to model checkpoint"),
) -> None:
    """Run phrase predictions on a track, emulating real-time left-to-right processing."""
    import torch
    from safetensors.torch import load_file

    from audiovj.config import FIXED_FRAMES
    from audiovj.data.features import (
        extract_mel_spectrogram,
        load_audio,
        slice_beat_windows,
    )
    from audiovj.model import PhrasePredictor

    track_path = TRACKS_DIR / f"{track_id}.json"
    if not track_path.exists():
        typer.echo(f"Error: Track not found: {track_id}")
        raise typer.Exit(1)

    track = Track.model_validate_json(track_path.read_text())
    if not track.audio_path or not Path(track.audio_path).exists():
        typer.echo("Error: Audio file not found")
        raise typer.Exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt_path = checkpoint or str(MODELS_DIR / "phrase_predictor.safetensors")
    if not Path(ckpt_path).exists():
        typer.echo(f"Error: Checkpoint not found: {ckpt_path}")
        raise typer.Exit(1)

    model = PhrasePredictor()
    state = load_file(ckpt_path)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    waveform, duration = load_audio(Path(track.audio_path))
    mel_spec = extract_mel_spectrogram(waveform)
    downbeats = build_downbeat_times(track, total_duration=duration)

    typer.echo(f"Track: {track.artist} - {track.name}")
    typer.echo(f"BPM: {track.bpm}  Downbeats: {len(downbeats)}")
    typer.echo()

    with torch.no_grad():
        for i, t in enumerate(downbeats):
            window, _ = slice_beat_windows(mel_spec, [t], track.bpm)
            if window.shape[0] == 0:
                continue

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
            confidence_next = next_probs[0, next_idx].item()
            confidence_current = current_probs[0, current_idx].item()
            beats_until = torch.expm1(out.beats_until[0, 0]).item()

            mins = int(t // 60)
            secs = t % 60
            typer.echo(
                f"{mins}:{secs:05.2f}  "
                f"current={PHRASE_TYPES[current_idx]:<12} ({confidence_current:.0%})  "
                f"next={PHRASE_TYPES[next_idx]:<12} ({confidence_next:.0%})  "
                f"beats_until={beats_until:.0f}"
            )


@app.command()
def list_devices() -> None:
    """List available audio input devices and their channels."""
    import sounddevice as sd

    devices = sd.query_devices()

    typer.echo("Audio input devices:\n")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] == 0:
            continue
        typer.echo(f"  [{i}] {dev['name']}")
        typer.echo(
            f"       Inputs: {dev['max_input_channels']}  "
            f"Sample rate: {int(dev['default_samplerate'])}Hz"
        )
        n = dev["max_input_channels"]
        if n <= 16:
            ch_list = ", ".join(str(c) for c in range(n))
            typer.echo(f"       Channels: {ch_list}")
        else:
            typer.echo(f"       Channels: 0-{n - 1}")
        typer.echo()

    typer.echo("Usage: audiovj run-live --audio-device <index|name> --audio-channels <ch,ch>")


@app.command()
def run_live(
    audio_device: str = typer.Option(None, help="Audio input device name or index"),
    audio_channels: str = typer.Option(
        None, help="Input channels to capture, 0-indexed comma-separated (e.g. '6,7'). Max 2."
    ),
    checkpoint: str = typer.Option(None, help="Path to model checkpoint"),
    carabiner_host: str = typer.Option("127.0.0.1", help="Carabiner host"),
    carabiner_port: int = typer.Option(17000, help="Carabiner port"),
    osc_host: str = typer.Option("127.0.0.1", help="OSC destination host"),
    osc_port: int = typer.Option(9000, help="OSC destination port"),
    correction_threshold: float = typer.Option(0.5, help="Min confidence for phrase correction"),
    transition_beats: float = typer.Option(4.0, help="Beats-until threshold for transition"),
    anticipate_beats: float = typer.Option(8.0, help="Beats-until threshold for anticipation cue"),
    latch_after: int = typer.Option(2, help="Consecutive agreements before latching a countdown"),
    sticky_beats: float = typer.Option(32.0, help="Sticky-hold window after a transition (beats)"),
    warmup_beats: float = typer.Option(16.0, help="Warmup window before SM decisions engage (beats)"),
    onset_threshold: float = typer.Option(
        0.30, help="Load-bearing onset threshold for transition cueing (locked operating point)"
    ),
    auto_gain: bool = typer.Option(
        True, "--auto-gain/--no-auto-gain",
        help="Auto-normalize the feed to training level (tracks loud sections, self-corrects mid-set volume changes)."
    ),
    input_gain_db: float = typer.Option(
        0.0, help="Fixed manual gain (dB) trim on top of auto-gain. Use with --no-auto-gain for a fixed level."
    ),
    ma3_host: str = typer.Option(
        None, help="grandMA3 console IP. Set this to drive executors 201-208 from the phrase."
    ),
    ma3_port: int = typer.Option(8000, help="grandMA3 OSC input port"),
    ma3_prefix: str = typer.Option("gma3", help="grandMA3 OSC prefix (Menu>In&Out>OSC)"),
    ma3_on_value: float = typer.Option(
        1.0, help="Fader value for the active phrase (1 if MA3 range 0..1, 100 if 0..100)"
    ),
    ma3_speedmaster: str = typer.Option(
        "3.1", help="SpeedMaster to sync BPM to (e.g. 3.1). Empty to disable BPM sync."
    ),
) -> None:
    """Start real-time phrase detection from live audio."""
    from audiovj.live.pipeline import LivePipeline

    # Default to the production seq model (UnifiedSeqPredictor); the old 8-beat
    # phrase_predictor is not wired into the streaming path.
    ckpt = Path(checkpoint) if checkpoint else MODELS_DIR / "seq_unified.safetensors"
    if not ckpt.exists():
        typer.echo(f"Error: Checkpoint not found: {ckpt}")
        raise typer.Exit(1)

    device = None
    if audio_device is not None:
        try:
            device = int(audio_device)
        except ValueError:
            device = audio_device

    channels = None
    if audio_channels is not None:
        try:
            channels = [int(c.strip()) for c in audio_channels.split(",")]
        except ValueError:
            typer.echo("Error: Invalid channel format. Use comma-separated integers (e.g. '6,7')")
            raise typer.Exit(1)
        if len(channels) > 2:
            typer.echo("Error: Max 2 audio channels supported")
            raise typer.Exit(1)

    pipeline = LivePipeline(
        checkpoint_path=ckpt,
        audio_device=device,
        audio_channels=channels,
        carabiner_host=carabiner_host,
        carabiner_port=carabiner_port,
        osc_host=osc_host,
        osc_port=osc_port,
        correction_threshold=correction_threshold,
        transition_beats=transition_beats,
        anticipate_beats=anticipate_beats,
        latch_after=latch_after,
        sticky_beats=sticky_beats,
        warmup_beats=warmup_beats,
        onset_threshold=onset_threshold,
        input_gain_db=input_gain_db,
        auto_gain=auto_gain,
        ma3_host=ma3_host,
        ma3_port=ma3_port,
        ma3_prefix=ma3_prefix,
        ma3_on_value=ma3_on_value,
        ma3_speedmaster=ma3_speedmaster,
    )
    pipeline.run()
