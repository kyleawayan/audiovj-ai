"""Live audio capture with a ring buffer fed by sounddevice."""

import numpy as np
import sounddevice as sd

from audiovj.config import SAMPLE_RATE

BUFFER_SIZE = 256_000  # ~5.8s at 44100Hz, covers 8 beats down to ~95 BPM


class AudioCapture:
    """Ring buffer fed by a sounddevice InputStream callback."""

    def __init__(
        self,
        device: int | str | None = None,
        channels: list[int] | None = None,
        sample_rate: int = SAMPLE_RATE,
        buffer_size: int = BUFFER_SIZE,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._buffer = np.zeros(buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._stream: sd.InputStream | None = None
        self._peak: float = 0.0  # mono peak level for metering

        # Channel selection (0-indexed). None = open mono, take channel 0.
        # e.g. [6, 7] to capture channels 7+8 and average to mono.
        self._channels = channels
        if channels:
            self._stream_channels = max(channels) + 1  # need at least this many
            self._channel_indices = channels
        else:
            self._stream_channels = 1
            self._channel_indices = [0]

        self._channel_peaks: list[float] = [0.0] * len(self._channel_indices)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — averages selected channels to mono, writes to ring buffer."""
        # indata shape: [frames, num_channels]
        selected = indata[:, self._channel_indices]
        mono = selected.mean(axis=1)
        self._peak = max(self._peak * 0.8, float(np.abs(mono).max()))
        for i in range(selected.shape[1]):
            self._channel_peaks[i] = max(
                self._channel_peaks[i] * 0.8, float(np.abs(selected[:, i]).max())
            )
        n = len(mono)
        buf_len = len(self._buffer)
        pos = self._write_pos % buf_len

        if pos + n <= buf_len:
            self._buffer[pos : pos + n] = mono
        else:
            first = buf_len - pos
            self._buffer[pos:] = mono[:first]
            self._buffer[: n - first] = mono[first:]

        self._write_pos += n

    def start(self) -> None:
        """Open the audio stream and begin capturing."""
        self._stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=self._stream_channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def peak(self) -> float:
        """Current mono peak level (0.0 to 1.0) with decay."""
        return self._peak

    @property
    def channel_peaks(self) -> list[float]:
        """Per-channel peak levels (0.0 to 1.0) with decay."""
        return list(self._channel_peaks)

    def read_last_n_samples(self, n: int) -> np.ndarray:
        """Read the most recent n samples from the ring buffer.

        Returns a 1D float32 array. Zero-pads if fewer than n samples
        have been captured so far.
        """
        buf_len = len(self._buffer)
        n = min(n, buf_len)
        pos = self._write_pos % buf_len

        if pos >= n:
            return self._buffer[pos - n : pos].copy()

        # Wrap-around read
        result = np.empty(n, dtype=np.float32)
        tail = n - pos
        result[:tail] = self._buffer[buf_len - tail :]
        result[tail:] = self._buffer[:pos]
        return result
