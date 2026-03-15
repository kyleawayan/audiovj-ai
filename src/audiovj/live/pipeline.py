"""Live pipeline: wires audio capture, Carabiner, inference, state, and OSC."""

import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import torch

from audiovj.config import CONTEXT_BEATS, SAMPLE_RATE
from audiovj.live.audio import AudioCapture
from audiovj.live.carabiner import CarabinerClient, DownbeatEvent
from audiovj.live.inference import InferenceEngine
from audiovj.live.osc import OSCEmitter
from audiovj.live.state import PhraseStateManager

BEAT_ON = "\u25cf"   # ●
BEAT_OFF = "\u25cb"  # ○


def _meter_bar(peak: float, width: int = 20) -> str:
    """Render an audio meter bar with pipe characters."""
    db = 20 * math.log10(max(peak, 1e-10))
    db = max(db, -60.0)
    filled = int((db + 60) / 60 * width)
    filled = max(0, min(filled, width))
    bar = "\u25a0" * filled + " " * (width - filled)
    return f"[{bar}] {db:+5.1f}dB"


def _beat_dots(phase: float) -> str:
    """Render 4 beat indicators showing current position in bar."""
    current_beat = int(phase) % 4
    return " ".join(
        BEAT_ON if i == current_beat else BEAT_OFF for i in range(4)
    )


def _setup_scroll_region() -> None:
    """Reserve bottom 2 lines for status bar by setting scroll region."""
    rows = os.get_terminal_size().lines
    # Set scroll region to rows 1 through rows-2 (leaves bottom 2 free)
    sys.stdout.write(f"\x1b[1;{rows - 2}r")
    # Move cursor into the scroll region
    sys.stdout.write(f"\x1b[{rows - 2};0H")
    sys.stdout.flush()


def _teardown_scroll_region() -> None:
    """Reset scroll region to full terminal and clear status bar."""
    rows = os.get_terminal_size().lines
    # Reset scroll region
    sys.stdout.write("\x1b[r")
    # Clear status bar lines
    sys.stdout.write(f"\x1b[{rows - 1};0H\x1b[2K\x1b[{rows};0H\x1b[2K")
    # Move cursor to bottom of scroll area
    sys.stdout.write(f"\x1b[{rows - 2};0H")
    sys.stdout.flush()


class LivePipeline:
    """Orchestrates all live components and runs the main inference loop."""

    def __init__(
        self,
        checkpoint_path: Path,
        audio_device: int | str | None = None,
        audio_channels: list[int] | None = None,
        carabiner_host: str = "127.0.0.1",
        carabiner_port: int = 17000,
        osc_host: str = "127.0.0.1",
        osc_port: int = 9000,
        correction_threshold: float = 0.7,
        transition_beats: float = 4.0,
        anticipate_beats: float = 8.0,
    ) -> None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading model from {checkpoint_path} on {device}...")
        self._engine = InferenceEngine(checkpoint_path, device)

        self._audio = AudioCapture(
            device=audio_device, channels=audio_channels, sample_rate=SAMPLE_RATE
        )
        self._osc = OSCEmitter(osc_host, osc_port)
        self._state = PhraseStateManager(
            correction_threshold=correction_threshold,
            transition_beats=transition_beats,
            anticipate_beats=anticipate_beats,
        )

        self._downbeat_queue: queue.Queue[DownbeatEvent] = queue.Queue()
        self._carabiner = CarabinerClient(
            host=carabiner_host,
            port=carabiner_port,
            on_downbeat=self._downbeat_queue.put,
        )
        self._status_running = False
        self._display_phrase = ""
        self._display_next = ""
        self._countdown_at_downbeat: float | None = None
        self._countdown_phrase_display: str = ""

    def _draw_status(self) -> None:
        """Draw the status bar on the reserved bottom 2 lines."""
        phase = self._carabiner.beat_phase
        bpm = self._carabiner.bpm
        channel_peaks = self._audio.channel_peaks
        beats = _beat_dots(phase)

        cols = os.get_terminal_size().columns
        rows = os.get_terminal_size().lines

        phrase_info = ""
        if self._display_phrase:
            phrase_info = f"  Current: {self._display_phrase}"
            if self._countdown_at_downbeat is not None and self._countdown_phrase_display:
                interpolated = max(0.0, self._countdown_at_downbeat - phase)
                phrase_info += f"   Next: {self._countdown_phrase_display} in {interpolated:.0f} beats"
            elif self._display_next:
                phrase_info += f"   Next: {self._display_next}"

        if len(channel_peaks) == 2:
            line1 = f"  L {_meter_bar(channel_peaks[0])}  {beats}  {bpm:6.1f} BPM"
            line2 = f"  R {_meter_bar(channel_peaks[1])}{phrase_info}"
        else:
            peak = channel_peaks[0] if channel_peaks else 0.0
            line1 = f"  {_meter_bar(peak)}  {beats}  {bpm:6.1f} BPM"
            line2 = phrase_info.strip() if phrase_info else ""

        # Save cursor, draw on reserved lines, restore cursor
        sys.stdout.write(
            f"\x1b7"
            f"\x1b[{rows - 1};0H\x1b[2K{line1[:cols]}"
            f"\x1b[{rows};0H\x1b[2K{line2[:cols]}"
            f"\x1b8"
        )
        sys.stdout.flush()

    def _status_loop(self) -> None:
        """Background thread: updates the status bar at ~20Hz."""
        while self._status_running:
            self._draw_status()
            time.sleep(0.05)

    def run(self) -> None:
        """Start all components and run until KeyboardInterrupt."""
        print("Starting audio capture...")
        self._audio.start()

        print("Connecting to Carabiner...")
        self._carabiner.start()

        self._osc.send_status("running")
        print("Live pipeline running. Press Ctrl+C to stop.\n")

        # Set up scroll region and start status bar
        _setup_scroll_region()
        self._status_running = True
        status_thread = threading.Thread(target=self._status_loop, daemon=True)
        status_thread.start()

        try:
            while True:
                evt = self._downbeat_queue.get()

                # Extract audio window: last CONTEXT_BEATS beats
                beat_duration = 60.0 / evt.bpm
                samples_needed = int(CONTEXT_BEATS * beat_duration * SAMPLE_RATE)
                audio_window = self._audio.read_last_n_samples(samples_needed)

                prediction = self._engine.predict(audio_window, evt.bpm)
                events = self._state.update(prediction)

                for event in events:
                    self._osc.send_event(event)
                self._osc.send_beat(evt.bpm)

                # Update status bar phrase info
                self._display_phrase = self._state.running_phrase
                countdown = self._state.countdown
                if countdown:
                    self._countdown_phrase_display = countdown[0]
                    self._countdown_at_downbeat = countdown[1]
                    self._display_next = f"{countdown[0]} in {countdown[1]:.0f} beats"
                else:
                    self._countdown_phrase_display = ""
                    self._countdown_at_downbeat = None
                    self._display_next = ""

                # Console output
                state_indicator = ""
                for event in events:
                    if event.kind == "transition":
                        state_indicator = f"  >>> TRANSITION to {event.phrase}"
                    elif event.kind == "correction":
                        state_indicator = f"  !!! CORRECTION to {event.phrase}"
                    elif event.kind == "anticipate":
                        state_indicator = f"  ... {event.phrase} in ~{event.beats_until:.0f} beats"

                print(
                    f"[{self._state.running_phrase:<12}] "
                    f"current={prediction.current_phrase:<12} ({prediction.current_confidence:.0%})  "
                    f"next={prediction.next_phrase:<12} ({prediction.next_confidence:.0%})  "
                    f"beats_until={prediction.beats_until:.0f}"
                    f"{state_indicator}"
                )

        except KeyboardInterrupt:
            pass
        finally:
            self._status_running = False
            self._osc.send_status("stopped")
            self._carabiner.stop()
            self._audio.stop()
            _teardown_scroll_region()
            print("Stopped.")
