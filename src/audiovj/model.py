"""Phrase Predictor model: CNN encoder + LSTM backbone with three output heads."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from audiovj.config import (
    ENCODER_CHANNELS,
    FIXED_FRAMES,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    N_MELS,
    NUM_PHRASES,
)


@dataclass
class ModelOutput:
    next_phrase_logits: torch.Tensor  # [batch, num_phrases]
    current_phrase_logits: torch.Tensor  # [batch, num_phrases]
    beats_until: torch.Tensor  # [batch, 1]


class SpectrogramEncoder(nn.Module):
    """CNN front-end that normalizes variable-width mel-spectrograms.

    Treats mel bins as input channels, convolves over the time axis.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        fixed_frames: int = FIXED_FRAMES,
        channels: list[int] = ENCODER_CHANNELS,
    ) -> None:
        super().__init__()
        self.fixed_frames = fixed_frames
        self.pool_time = nn.AdaptiveAvgPool1d(fixed_frames)

        layers: list[nn.Module] = []
        in_ch = n_mels
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.out_channels = channels[-1]
        self.out_seq_len = fixed_frames // (2 ** len(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [batch, n_mels, variable_frames]. Output: [batch, seq_len, channels]."""
        # AdaptiveAvgPool1d on MPS requires the input length be divisible by the
        # output size; live/eval windows are arbitrary widths (~344 frames for an
        # 8-beat window). Compute the pool on CPU in that case (cheap) to stay
        # numerically exact, rather than padding (which would dilute the last bin).
        if x.device.type == "mps" and x.shape[-1] % self.fixed_frames != 0:
            x = self.pool_time(x.cpu()).to(x.device)  # [batch, n_mels, fixed_frames]
        else:
            x = self.pool_time(x)  # [batch, n_mels, fixed_frames]
        x = self.conv(x)  # [batch, out_channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, seq_len, out_channels]
        return x


class PhrasePredictor(nn.Module):
    """Full model: CNN encoder -> LSTM -> three output heads."""

    def __init__(
        self,
        n_mels: int = N_MELS,
        fixed_frames: int = FIXED_FRAMES,
        encoder_channels: list[int] = ENCODER_CHANNELS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        num_phrases: int = NUM_PHRASES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = SpectrogramEncoder(n_mels, fixed_frames, encoder_channels)

        self.lstm = nn.LSTM(
            input_size=self.encoder.out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Regularize the shared representation before the heads.
        self.head_dropout = nn.Dropout(dropout)

        # Forward prediction head
        self.next_phrase_head = nn.Linear(lstm_hidden, num_phrases)
        self.beats_until_head = nn.Linear(lstm_hidden, 1)

        # Current phrase classification head
        self.current_phrase_head = nn.Linear(lstm_hidden, num_phrases)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Input: [batch, n_mels, variable_frames]."""
        encoded = self.encoder(x)  # [batch, seq_len, channels]
        lstm_out, _ = self.lstm(encoded)  # [batch, seq_len, hidden]
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden]
        last_hidden = self.head_dropout(last_hidden)

        return ModelOutput(
            next_phrase_logits=self.next_phrase_head(last_hidden),
            current_phrase_logits=self.current_phrase_head(last_hidden),
            beats_until=self.beats_until_head(last_hidden),
        )


class UnifiedSeqPredictor(nn.Module):
    """Longer-context sequence model: per-downbeat CNN window encoder ->
    *causal* LSTM ACROSS downbeats -> 3 heads (current/next phrase + a dedicated
    beats-until branch).

    This is the production model (KA-233/234 full-scale winner; macro-F1 ~0.65 vs
    the 8-beat PhrasePredictor's ~0.46). Its advantage is the cross-downbeat LSTM
    that accumulates context, so it MUST be run statefully when streaming live:
    feed one downbeat window at a time and carry the LSTM hidden state forward
    (see ``step``). Because the context LSTM is unidirectional/causal, stepping
    one downbeat at a time produces outputs identical to a full-sequence
    ``forward`` — so the offline-certified numbers transfer to live exactly.

    Submodule names match experiments/_unified.py so the trained checkpoint
    (seq_unified_full_v2.safetensors) loads directly.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        fixed_frames: int = FIXED_FRAMES,
        encoder_channels: list[int] = ENCODER_CHANNELS,
        lstm_hidden: int = LSTM_HIDDEN,
        lstm_layers: int = LSTM_LAYERS,
        num_phrases: int = NUM_PHRASES,
        dropout: float = 0.0,
        detach: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SpectrogramEncoder(n_mels, fixed_frames, encoder_channels)
        ch = self.encoder.out_channels
        self.ctx_lstm = nn.LSTM(
            ch, lstm_hidden, lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.next_phrase_head = nn.Linear(lstm_hidden, num_phrases)
        self.current_phrase_head = nn.Linear(lstm_hidden, num_phrases)
        self.beats_branch = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(lstm_hidden, 1),
        )
        self.detach = detach

    def _encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, n_mels, frames] -> per-downbeat embeddings [B, T, ch]."""
        b, t = x.shape[0], x.shape[1]
        enc = self.encoder(x.reshape(b * t, x.shape[2], x.shape[3]))  # [B*T, seq, ch]
        return enc.mean(dim=1).reshape(b, t, -1)  # [B, T, ch]

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Whole-sequence forward. Input: [B, T, n_mels, frames]."""
        win = self._encode_windows(x)
        ctx, _ = self.ctx_lstm(win)
        h = self.head_dropout(ctx)
        beats_in = h.detach() if self.detach else h
        return ModelOutput(
            next_phrase_logits=self.next_phrase_head(h),
            current_phrase_logits=self.current_phrase_head(h),
            beats_until=self.beats_branch(beats_in),
        )

    def step(
        self, window: torch.Tensor, state: tuple | None = None
    ) -> tuple[ModelOutput, tuple]:
        """Stateful single-downbeat step for live streaming.

        ``window``: one downbeat's mel window [n_mels, frames] (or [1, n_mels,
        frames]). ``state``: carried (h, c) from the previous downbeat, or None
        to start a fresh track. Returns (ModelOutput with [1, num_phrases]
        logits, new_state). Carry ``new_state`` into the next call. Equivalent to
        ``forward`` over the sequence so far (causal LSTM).
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)  # [1, n_mels, frames]
        enc = self.encoder(window)            # [1, seq, ch]
        win = enc.mean(dim=1).unsqueeze(1)    # [1, 1, ch]
        ctx, new_state = self.ctx_lstm(win, state)  # [1, 1, hidden]
        h = self.head_dropout(ctx[:, -1, :])  # [1, hidden]
        out = ModelOutput(
            next_phrase_logits=self.next_phrase_head(h),
            current_phrase_logits=self.current_phrase_head(h),
            beats_until=self.beats_branch(h),
        )
        return out, new_state
