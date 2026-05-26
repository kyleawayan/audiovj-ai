"""Run phrase predictions on a folder of unlabeled audio for ear-checking.

No labels needed; for each audio file we:
  - estimate BPM + beats via librosa
  - take every 4th beat as a downbeat (4/4 assumption — typical for EDM)
  - run the model + State Manager left-to-right
  - dump predictions to <out>/<filename>.json

This is the "real-world" eval: you point it at a folder the model has never seen
and inspect the predicted phrase sequence + state-manager transitions by ear.
"""

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from audiovj.config import FIXED_FRAMES, MODELS_DIR, PHRASE_TYPES
from audiovj.data.features import extract_mel_spectrogram, load_audio, slice_beat_windows
from audiovj.live.inference import PredictionResult
from audiovj.live.state import PhraseStateManager
from audiovj.model import PhrasePredictor

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aif", ".aiff", ".ogg", ".aac"}


def _find_audio_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in folder.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    return sorted(files)


def _estimate_beats(audio_path: Path) -> tuple[float, list[float]]:
    """Return (bpm, beat_times) via librosa.beat.beat_track."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, beat_times


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_folder(
    folder: Path,
    out_dir: Path,
    checkpoint: Path | None = None,
    correction_threshold: float = 0.7,
    transition_beats: float = 4.0,
    anticipate_beats: float = 8.0,
    skip_existing: bool = True,
) -> tuple[int, int, int]:
    """Run predictions on every audio file under `folder`.

    Returns (processed, skipped, failed).
    """
    folder = folder.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint or (MODELS_DIR / "phrase_predictor.safetensors")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = _get_device()
    print(f"device: {device}")

    model = PhrasePredictor()
    model.load_state_dict(load_file(str(ckpt_path)))
    model.to(device)
    model.eval()

    files = _find_audio_files(folder)
    if not files:
        print(f"no audio files under {folder}")
        return 0, 0, 0

    processed = skipped = failed = 0

    for i, audio_path in enumerate(files, 1):
        out_path = out_dir / f"{audio_path.stem}.json"
        if skip_existing and out_path.exists():
            print(f"  [{i}/{len(files)}] {audio_path.name} — skipped (cached)")
            skipped += 1
            continue

        t0 = time.time()
        try:
            bpm, beats = _estimate_beats(audio_path)
            # Every 4th beat = downbeat (4/4 assumption)
            downbeats = beats[::4]
        except Exception as e:
            print(f"  [{i}/{len(files)}] {audio_path.name} — FAILED beat detection: {e}")
            failed += 1
            continue

        try:
            waveform, _duration = load_audio(audio_path)
            mel_spec = extract_mel_spectrogram(waveform)
        except Exception as e:
            print(f"  [{i}/{len(files)}] {audio_path.name} — FAILED audio load: {e}")
            failed += 1
            continue

        sm = PhraseStateManager(
            correction_threshold=correction_threshold,
            transition_beats=transition_beats,
            anticipate_beats=anticipate_beats,
        )

        predictions: list[dict] = []
        sm_events: list[dict] = []

        with torch.no_grad():
            for t in downbeats:
                window, _ = slice_beat_windows(mel_spec, [t], bpm)
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

                prediction = PredictionResult(
                    current_phrase=PHRASE_TYPES[current_idx],
                    current_confidence=current_probs[0, current_idx].item(),
                    next_phrase=PHRASE_TYPES[next_idx],
                    next_confidence=next_probs[0, next_idx].item(),
                    beats_until=torch.expm1(out.beats_until[0, 0]).item(),
                )

                events = sm.update(prediction)

                predictions.append({
                    "downbeat_time": round(float(t), 3),
                    "raw_current": prediction.current_phrase,
                    "raw_current_conf": round(prediction.current_confidence, 3),
                    "raw_next": prediction.next_phrase,
                    "raw_next_conf": round(prediction.next_confidence, 3),
                    "beats_until": round(prediction.beats_until, 2),
                    "sm_running_phrase": sm.running_phrase,
                })

                for ev in events:
                    sm_events.append({
                        "time": round(float(t), 3),
                        "kind": ev.kind,
                    })

        result = {
            "audio_path": str(audio_path),
            "estimated_bpm": round(bpm, 2),
            "n_downbeats": len(downbeats),
            "predictions": predictions,
            "sm_events": sm_events,
        }
        out_path.write_text(json.dumps(result, indent=2))

        elapsed = time.time() - t0
        print(
            f"  [{i}/{len(files)}] {audio_path.name} — done ({elapsed:.1f}s, "
            f"{len(predictions)} downbeats, {len(sm_events)} sm events)"
        )
        processed += 1

    return processed, skipped, failed
