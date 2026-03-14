"""Carabiner TCP client: connects to Ableton Link bridge, detects downbeats."""

import re
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

POLL_INTERVAL = 0.02  # 50Hz polling

STATUS_RE = re.compile(
    r"status\s*\{\s*"
    r":peers\s+(\d+)\s+"
    r":bpm\s+([\d.]+)\s+"
    r":start\s+(\d+)\s+"
    r":beat\s+([\d.]+)"
)


@dataclass
class DownbeatEvent:
    time: float  # wall-clock time.monotonic() when detected
    bpm: float
    beat_number: float


class CarabinerClient:
    """TCP client for Carabiner (Ableton Link bridge).

    Connects to Carabiner, polls status at ~50Hz, and fires a callback
    when a downbeat (phase crossing 0 in quantum=4) is detected.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_downbeat: Callable[[DownbeatEvent], None],
    ) -> None:
        self._host = host
        self._port = port
        self._on_downbeat = on_downbeat
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._bpm = 120.0
        self._beat_phase = 0.0  # 0.0-3.99, current position within 4-beat bar
        self._recv_buffer = b""

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def beat_phase(self) -> float:
        """Current position within a 4-beat bar (0.0 to <4.0)."""
        return self._beat_phase

    def start(self) -> None:
        """Connect to Carabiner (retrying until available) and start polling."""
        if self._running:
            return
        self._connect_with_retry()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _connect_with_retry(self) -> None:
        """Keep trying to connect until Carabiner is available."""
        printed_waiting = False
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((self._host, self._port))
                sock.settimeout(0.1)  # non-blocking reads in poll loop
                self._sock = sock
                # Request initial status
                self._send("status\n")
                msgs = self._read_messages()
                for msg in msgs:
                    self._parse_status(msg)
                print(f"Connected to Carabiner at {self._host}:{self._port} (BPM: {self._bpm:.1f})")
                return
            except (ConnectionRefusedError, OSError):
                sock.close()
                if not printed_waiting:
                    print(f"Waiting for Carabiner on {self._host}:{self._port}...")
                    printed_waiting = True
                time.sleep(2.0)

    def _send(self, msg: str) -> None:
        """Send a message to Carabiner."""
        if self._sock is not None:
            self._sock.sendall(msg.encode())

    def _read_messages(self) -> list[str]:
        """Read available data and return complete newline-delimited messages."""
        if self._sock is None:
            return []
        try:
            data = self._sock.recv(4096)
            if not data:
                return []
            self._recv_buffer += data
        except (socket.timeout, BlockingIOError):
            pass

        messages = []
        while b"\n" in self._recv_buffer:
            line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
            decoded = line.decode("utf-8", errors="replace").strip()
            if decoded:
                messages.append(decoded)

        # Handle case where message doesn't end with newline
        # (some Carabiner versions may not send trailing newline)
        if self._recv_buffer and b"\n" not in self._recv_buffer:
            decoded = self._recv_buffer.decode("utf-8", errors="replace").strip()
            if decoded and STATUS_RE.search(decoded):
                messages.append(decoded)
                self._recv_buffer = b""

        return messages

    def _parse_status(self, msg: str) -> float | None:
        """Parse a status message. Returns beat number or None."""
        m = STATUS_RE.search(msg)
        if m is None:
            return None
        self._bpm = float(m.group(2))
        return float(m.group(4))

    def _poll_loop(self) -> None:
        """Polling thread: request status, detect downbeats."""
        prev_phase: float | None = None

        while self._running:
            self._send("status\n")
            msgs = self._read_messages()

            for msg in msgs:
                beat = self._parse_status(msg)
                if beat is None:
                    continue

                phase = beat % 4.0
                self._beat_phase = phase

                if prev_phase is not None and prev_phase > 3.0 and phase < 1.0:
                    self._on_downbeat(DownbeatEvent(
                        time=time.monotonic(),
                        bpm=self._bpm,
                        beat_number=beat,
                    ))

                prev_phase = phase

            time.sleep(POLL_INTERVAL)
