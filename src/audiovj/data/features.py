"""Audio feature extraction: mel-spectrograms and beat-aligned windowing."""

from pathlib import Path

import torch
import torchaudio
from safetensors.torch import save_file

from audiovj.config import (
    CONTEXT_BEATS,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)
from audiovj.data.rekordbox import Track, build_downbeat_times


def load_audio(
    audio_path: Path, sample_rate: int = SAMPLE_RATE
) -> tuple[torch.Tensor, float]:
    """Load audio, convert to mono, resample. Returns (waveform, duration_sec)."""
    waveform, sr = torchaudio.load(str(audio_path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    duration = waveform.shape[1] / sample_rate
    return waveform, duration


def extract_mel_spectrogram(
    waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """Compute mel-spectrogram from waveform.

    Returns tensor of shape [1, n_mels, time_frames] (log-scale dB).
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    mel_spec = mel_transform(waveform)
    mel_spec = db_transform(mel_spec)

    return mel_spec  # [1, n_mels, time_frames]


def slice_beat_windows(
    mel_spec: torch.Tensor,
    downbeat_times: list[float],
    bpm: float,
    context_beats: int = CONTEXT_BEATS,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> tuple[torch.Tensor, list[int]]:
    """Extract fixed-size windows at each downbeat position.

    Each window covers `context_beats` beats of audio ending at the downbeat.
    Zero-pads if the window extends before the start of the audio.

    Returns (windows_tensor, kept_indices) where windows_tensor has shape
    [num_downbeats, n_mels, frames_per_window] and kept_indices maps each
    window back to its position in the original downbeat_times list.
    """
    frames_per_beat = (60.0 / bpm) * sample_rate / hop_length
    window_frames = int(context_beats * frames_per_beat)
    total_frames = mel_spec.shape[-1]

    windows: list[torch.Tensor] = []
    kept_indices: list[int] = []
    for i, t in enumerate(downbeat_times):
        # Frame index of this downbeat
        center_frame = int(t * sample_rate / hop_length)
        start_frame = center_frame - window_frames
        end_frame = center_frame

        if end_frame <= 0 or start_frame >= total_frames:
            continue

        # Extract window, handling left padding
        if start_frame < 0:
            pad_frames = -start_frame
            chunk = mel_spec[0, :, 0:end_frame]  # [n_mels, available_frames]
            padding = torch.zeros(chunk.shape[0], pad_frames)
            window = torch.cat([padding, chunk], dim=-1)
        else:
            window = mel_spec[0, :, start_frame:end_frame]

        # Ensure exact size (in case of rounding)
        if window.shape[-1] < window_frames:
            pad = torch.zeros(window.shape[0], window_frames - window.shape[-1])
            window = torch.cat([pad, window], dim=-1)
        elif window.shape[-1] > window_frames:
            window = window[:, -window_frames:]

        windows.append(window)
        kept_indices.append(i)

    if not windows:
        return torch.empty(0, mel_spec.shape[1], window_frames), []

    return torch.stack(windows), kept_indices  # [num_windows, n_mels, window_frames]


def preprocess_track(track: Track, output_dir: Path) -> int:
    """Full preprocessing pipeline for a single track.

    Extracts mel-spectrogram, slices into beat-aligned windows,
    saves as .safetensors. Returns number of windows created.
    """
    if track.audio_path is None:
        return 0

    audio_path = Path(track.audio_path)

    # Load audio once, get duration and mel-spectrogram
    waveform, duration = load_audio(audio_path)
    mel_spec = extract_mel_spectrogram(waveform)

    # Build downbeat times
    downbeats = build_downbeat_times(track, total_duration=duration)

    # Slice into windows
    windows, kept_indices = slice_beat_windows(mel_spec, downbeats, track.bpm)

    if windows.shape[0] == 0:
        return 0

    # Save windows as safetensors
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{track.track_id}.safetensors"
    save_file(
        {
            "windows": windows,
            "duration": torch.tensor([duration]),
            "kept_indices": torch.tensor(kept_indices, dtype=torch.long),
        },
        str(output_path),
    )

    return windows.shape[0]
