"""Inference engine: loads model and runs single-window predictions."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audiovj.config import FIXED_FRAMES, N_MELS, PHRASE_TYPES
from audiovj.data.features import extract_mel_spectrogram
from audiovj.model import PhrasePredictor


@dataclass
class PredictionResult:
    current_phrase: str
    current_confidence: float
    next_phrase: str
    next_confidence: float
    beats_until: float
    # Full current-phrase probability vector (len == NUM_PHRASES), in PHRASE_TYPES
    # order. Populated by the seq engine; needed for onset-based cueing (the
    # locked operating point). None for the legacy single-window engine.
    current_probs: tuple[float, ...] | None = None


class InferenceEngine:
    """Loads the trained model and runs single-window predictions."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        from safetensors.torch import load_file

        self._device = device
        self._model = PhrasePredictor()
        state = load_file(str(checkpoint_path))
        self._model.load_state_dict(state)
        self._model.to(device)
        self._model.eval()

        # Warm up MPS (first call compiles Metal shaders)
        dummy = torch.zeros(1, N_MELS, FIXED_FRAMES, device=device)
        with torch.no_grad():
            self._model(dummy)

    def predict(self, audio_samples: np.ndarray, bpm: float) -> PredictionResult:
        """Run inference on a raw audio window.

        Args:
            audio_samples: 1D float32 array of raw audio samples.
            bpm: Current BPM for context (unused by model, reserved for future).

        Returns:
            PredictionResult with phrase classifications and beats_until.
        """
        waveform = torch.from_numpy(audio_samples).unsqueeze(0)  # [1, samples]
        mel_spec = extract_mel_spectrogram(waveform)  # [1, n_mels, frames]

        # Pad to multiple of FIXED_FRAMES for MPS compatibility
        frames = mel_spec.shape[-1]
        pad_to = ((frames + FIXED_FRAMES - 1) // FIXED_FRAMES) * FIXED_FRAMES
        if pad_to > frames:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_to - frames))

        mel_spec = mel_spec.to(self._device)

        with torch.no_grad():
            out = self._model(mel_spec)

        next_probs = torch.softmax(out.next_phrase_logits, dim=-1)
        current_probs = torch.softmax(out.current_phrase_logits, dim=-1)

        next_idx = next_probs.argmax(-1).item()
        current_idx = current_probs.argmax(-1).item()

        return PredictionResult(
            current_phrase=PHRASE_TYPES[current_idx],
            current_confidence=current_probs[0, current_idx].item(),
            next_phrase=PHRASE_TYPES[next_idx],
            next_confidence=next_probs[0, next_idx].item(),
            beats_until=torch.expm1(out.beats_until[0, 0]).item(),
        )


class SeqInferenceEngine:
    """Stateful streaming inference for the UnifiedSeqPredictor (production model).

    Unlike InferenceEngine (single-window, stateless), this carries the
    cross-downbeat LSTM hidden state across calls — which is what makes the seq
    model work. Call ``reset()`` at the start of a track, then ``step_window`` or
    ``predict`` once per downbeat. Because the context LSTM is causal, the
    per-downbeat outputs are identical to an offline full-sequence forward
    (verified: experiments/_steptest.py).
    """

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        from safetensors.torch import load_file

        from audiovj.model import UnifiedSeqPredictor

        self._device = device
        self._model = UnifiedSeqPredictor()
        self._model.load_state_dict(load_file(str(checkpoint_path)))
        self._model.to(device)
        self._model.eval()
        self._state: tuple | None = None

    def reset(self) -> None:
        """Clear the LSTM state — call at track start (or after a long gap)."""
        self._state = None

    def step_window(self, mel_window: torch.Tensor) -> PredictionResult:
        """Advance one downbeat from a precomputed mel window [n_mels, frames]."""
        with torch.no_grad():
            out, self._state = self._model.step(mel_window.to(self._device), self._state)
        cprob = torch.softmax(out.current_phrase_logits[0], dim=-1)
        nprob = torch.softmax(out.next_phrase_logits[0], dim=-1)
        ci = int(cprob.argmax()); ni = int(nprob.argmax())
        return PredictionResult(
            current_phrase=PHRASE_TYPES[ci],
            current_confidence=float(cprob[ci]),
            next_phrase=PHRASE_TYPES[ni],
            next_confidence=float(nprob[ni]),
            beats_until=float(torch.expm1(out.beats_until[0, 0])),
            current_probs=tuple(cprob.tolist()),
        )

    def predict(self, audio_samples: np.ndarray, bpm: float) -> PredictionResult:
        """Advance one downbeat from the trailing CONTEXT_BEATS of live audio."""
        waveform = torch.from_numpy(audio_samples).unsqueeze(0)  # [1, samples]
        mel_spec = extract_mel_spectrogram(waveform)[0]          # [n_mels, frames]
        return self.step_window(mel_spec)
