"""Session recorder: capture everything needed to reconstruct a set offline.

The point of recording is that every offline question becomes rerunnable from
ONE live set instead of needing a new one: sweeping p(drop) at sub-beat window
positions, re-simulating the AGC with different constants, retuning thresholds,
measuring true signed cue latency against human drop marks.

Three files per session:
  manifest.json  run config, checkpoint digest, device, negotiated sample rate,
                 measured input latency, Link peer count -- everything needed to
                 know what produced the data
  audio.wav      float32 mono, PRE-AGC, at the capture sample rate. Pre-AGC on
                 purpose: gain is a pure function of the signal, so recording
                 the raw feed lets any AGC design be re-simulated offline, while
                 recording post-AGC would bake one design in permanently.
  events.jsonl   one record per downbeat (full 10-class prob vector, events,
                 applied gain, ring-buffer position) plus marks and resets

Alignment is by SAMPLE, not wall clock. Every record carries ``audio_pos``
(total samples captured since the stream opened); subtract the manifest's
``audio_start_pos`` to get the sample offset into audio.wav. Wall-clock
alignment would only be good to the input latency, which is the same size as
the effect being measured.

Audio is written on a dedicated thread fed by a queue. Writing to a file from
inside the PortAudio callback can block and cause dropouts, which during a live
set means glitched audio into the model AND a glitched recording.
"""

import json
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np


class _Float32WavWriter:
    """Streaming WAV writer for 32-bit IEEE float mono.

    Hand-rolled because soundfile/scipy are not dependencies and stdlib ``wave``
    only does integer PCM. Sizes are patched on close; an interrupted session
    leaves a file with zeroed sizes, which ``repair()`` can fix from the byte
    count on disk.
    """

    def __init__(self, path: Path, sample_rate: int) -> None:
        self._path = path
        self._sr = sample_rate
        self._n = 0
        self._f = path.open("wb")
        self._write_header(0)

    def _write_header(self, n_frames: int) -> None:
        data_bytes = n_frames * 4
        # fmt chunk is 18 bytes (not 16) and a 'fact' chunk is present because
        # WAVE_FORMAT_IEEE_FLOAT (3) requires both for strict readers.
        riff_size = 4 + (8 + 18) + (8 + 4) + (8 + data_bytes)
        self._f.seek(0)
        self._f.write(b"RIFF")
        self._f.write(struct.pack("<I", riff_size))
        self._f.write(b"WAVE")
        self._f.write(b"fmt ")
        self._f.write(struct.pack("<I", 18))
        self._f.write(struct.pack("<HHIIHH", 3, 1, self._sr, self._sr * 4, 4, 32))
        self._f.write(struct.pack("<H", 0))          # cbSize
        self._f.write(b"fact")
        self._f.write(struct.pack("<I", 4))
        self._f.write(struct.pack("<I", n_frames))
        self._f.write(b"data")
        self._f.write(struct.pack("<I", data_bytes))

    def write(self, chunk: np.ndarray) -> None:
        buf = np.asarray(chunk, dtype="<f4").tobytes()
        self._f.seek(0, 2)
        self._f.write(buf)
        self._n += len(chunk)

    @property
    def frames(self) -> int:
        return self._n

    def close(self) -> None:
        if self._f.closed:
            return
        self._write_header(self._n)
        self._f.flush()
        self._f.close()


class SessionRecorder:
    """Owns one session directory. Thread-safe for log(); audio via a queue."""

    def __init__(self, session_dir: Path, sample_rate: int,
                 record_audio: bool = True, queue_chunks: int = 512) -> None:
        self.dir = session_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sr = sample_rate
        self._record_audio = record_audio
        self._events = (self.dir / "events.jsonl").open("a")
        self._lock = threading.Lock()
        self._wav: _Float32WavWriter | None = None
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.dropped_chunks = 0
        self._manifest: dict = {}
        if record_audio:
            self._wav = _Float32WavWriter(self.dir / "audio.wav", sample_rate)
            self._q = queue.Queue(maxsize=queue_chunks)

    # -- audio ---------------------------------------------------------------
    def audio_sink(self, mono: np.ndarray) -> None:
        """Called from the PortAudio callback. Must not block."""
        if self._q is None:
            return
        try:
            self._q.put_nowait(mono.copy())
        except queue.Full:
            # Better to lose a chunk than to stall the audio thread and glitch
            # the live feed. Counted so the manifest can report it honestly.
            self.dropped_chunks += 1

    def _writer_loop(self) -> None:
        while self._running or (self._q is not None and not self._q.empty()):
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._wav is not None:
                self._wav.write(chunk)

    # -- lifecycle -----------------------------------------------------------
    def start(self, manifest: dict) -> None:
        self._manifest = dict(manifest)
        self._write_manifest()
        if self._q is not None:
            self._running = True
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()

    def _write_manifest(self) -> None:
        (self.dir / "manifest.json").write_text(
            json.dumps(self._manifest, indent=2, default=str)
        )

    def update_manifest(self, extra: dict) -> None:
        """Merge in values only knowable after the stream opens (negotiated
        sample rate, real input latency, Link peers, dropped chunks)."""
        self._manifest.update(extra)
        self._write_manifest()

    def log(self, rec: dict) -> None:
        with self._lock:
            try:
                self._events.write(json.dumps(rec, default=str) + "\n")
                self._events.flush()
            except Exception:
                pass

    def close(self, summary: dict) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        frames = self._wav.frames if self._wav else 0
        summary = dict(summary)
        summary.update({
            "kind": "session_end",
            "audio_frames": frames,
            "audio_seconds": frames / self._sr if frames else 0.0,
            "dropped_audio_chunks": self.dropped_chunks,
        })
        self.log(summary)
        if self._wav is not None:
            self._wav.close()
        self.update_manifest({
            "audio_frames": frames,
            "audio_seconds": frames / self._sr if frames else 0.0,
            "dropped_audio_chunks": self.dropped_chunks,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        with self._lock:
            self._events.close()

    @staticmethod
    def new_dir(root: Path) -> Path:
        return Path(root) / time.strftime("%Y%m%d-%H%M%S")
