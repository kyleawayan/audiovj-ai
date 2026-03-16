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
    ) -> None:
        super().__init__()
        self.encoder = SpectrogramEncoder(n_mels, fixed_frames, encoder_channels)

        self.lstm = nn.LSTM(
            input_size=self.encoder.out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)

        # Forward prediction head
        self.next_phrase_head = nn.Linear(lstm_hidden, num_phrases)
        self.beats_until_head = nn.Linear(lstm_hidden, 1)

        # Current phrase classification head
        self.current_phrase_head = nn.Linear(lstm_hidden, num_phrases)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Input: [batch, n_mels, variable_frames]."""
        encoded = self.encoder(x)  # [batch, seq_len, channels]
        lstm_out, _ = self.lstm(encoded)  # [batch, seq_len, hidden]
        last_hidden = self.dropout(lstm_out[:, -1, :])  # [batch, hidden]

        return ModelOutput(
            next_phrase_logits=self.next_phrase_head(last_hidden),
            current_phrase_logits=self.current_phrase_head(last_hidden),
            beats_until=self.beats_until_head(last_hidden),
        )
