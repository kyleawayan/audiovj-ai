"""Live pipeline: wires audio capture, Carabiner, inference, state, and OSC."""

import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from audiovj.config import CONTEXT_BEATS, PHRASE_TYPES, SAMPLE_RATE
from audiovj.live.audio import AudioCapture
from audiovj.live.carabiner import CarabinerClient, DownbeatEvent
from audiovj.live.cue import OnsetCueTracker
from audiovj.live.inference import SeqInferenceEngine
from audiovj.live.osc import OSCEmitter
from audiovj.live.state import PhraseStateManager

BEAT_ON = "\u25cf"   # ●
BEAT_OFF = "\u25cb"  # ○

_DROP_IDX = PHRASE_TYPES.index("drop")
_BUILDUP_IDX = PHRASE_TYPES.index("buildup")


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


def _term_size() -> os.terminal_size | None:
    """Terminal size, or None when stdout is not a TTY.

    Piping (``run-live | tee set.log``) makes stdout a pipe, and
    ``os.get_terminal_size()`` then raises OSError. Unguarded that kills the
    status thread on exactly the runs you most want to capture.
    """
    try:
        return os.get_terminal_size()
    except OSError:
        return None


def _setup_scroll_region() -> None:
    """Reserve bottom 2 lines for status bar by setting scroll region."""
    size = _term_size()
    if size is None:
        return
    rows = size.lines
    # Set scroll region to rows 1 through rows-2 (leaves bottom 2 free)
    sys.stdout.write(f"\x1b[1;{rows - 2}r")
    # Move cursor into the scroll region
    sys.stdout.write(f"\x1b[{rows - 2};0H")
    sys.stdout.flush()


def _teardown_scroll_region() -> None:
    """Reset scroll region to full terminal and clear status bar."""
    size = _term_size()
    if size is None:
        return
    rows = size.lines
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
        correction_threshold: float = 0.5,
        transition_beats: float = 4.0,
        anticipate_beats: float = 8.0,
        latch_after: int = 2,
        sticky_beats: float = 32.0,
        warmup_beats: float = 16.0,
        onset_threshold: float = 0.30,
        input_gain_db: float = 0.0,
        auto_gain: bool = True,
        ma3_host: str | None = None,
        ma3_port: int = 8000,
        ma3_prefix: str = "gma3",
        ma3_on_value: float = 1.0,
        ma3_speedmaster: str = "3.1",
        midi_port: str = "DDJ-GRV6",
        midi_note: int = 61,
        midi_mark_note: int | None = None,
        force_drop_beats: int = 32,
        drop_confirm: int = 1,
        drop_release: int = 2,
        drop_light_threshold: float = 0.0,
        drop_light_hold: int = 2,
        event_log: Path | None = None,
        reset_state_beats: int = 0,
    ) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Loading model from {checkpoint_path} on {device}...")
        # Stateful streaming seq model: carries cross-downbeat LSTM context.
        self._engine = SeqInferenceEngine(checkpoint_path, device)
        self._engine.reset()
        # Level matching: training audio is full-scale WAV; a line/loopback feed is
        # often much quieter, and mel is absolute dB, so a quiet feed collapses
        # predictions to the quiet classes (intro/altintro). Two stages:
        #  - auto_gain: a slow peak-follower AGC that tracks the loud (drop) level
        #    over ~tens of seconds and normalizes it toward full scale, so a
        #    mid-set volume/limiter change self-corrects. Slow enough to preserve
        #    within-track dynamics (a drop stays louder than a breakdown).
        #  - input_gain: a fixed manual trim applied on top (dB).
        self._input_gain = 10.0 ** (input_gain_db / 20.0)
        self._auto_gain = auto_gain
        self._agc_level = 0.0        # slow peak follower (linear, 0..1)
        self._agc_target = 0.9       # aim the loud level near full scale
        self._agc_release = 0.95     # per-downbeat decay (~40s memory at ~2s/db)
        self._agc_max = 60.0         # cap boost at ~35 dB (avoid amplifying silence)
        self._agc_floor = 0.003      # below this peak = silence, hold gain at 1
        self._agc_gain = 1.0         # last applied auto gain (for the status bar)

        self._audio = AudioCapture(
            device=audio_device, channels=audio_channels, sample_rate=SAMPLE_RATE
        )
        self._osc = OSCEmitter(osc_host, osc_port)
        # Optional grandMA3 bridge: current phrase -> executor 201-208 (one lit).
        self._ma3 = None
        if ma3_host:
            from audiovj.live.grandma3 import GrandMA3PhraseBridge
            self._ma3 = GrandMA3PhraseBridge(
                ma3_host, ma3_port, ma3_prefix, ma3_on_value, ma3_speedmaster
            )
            bpm_note = f", BPM->Master {ma3_speedmaster}" if ma3_speedmaster else ""
            print(f"grandMA3 -> {ma3_host}:{ma3_port} /{ma3_prefix}/Fader20x{bpm_note}")
        # Onset cueing = the locked operating point for transitions / drop state.
        self._cue = OnsetCueTracker(
            onset_threshold=onset_threshold,
            drop_confirm=drop_confirm,
            drop_release=drop_release,
        )
        # Optional: light the drop executor on p(drop) crossing a threshold rather
        # than on winning a 10-class argmax. The argmax is the strictest possible
        # rule and is what makes the lights a full bar late; the model was trained
        # to emit "drop" at the boundary downbeat from buildup evidence alone
        # (dataset.py:31-37 vs features.py:84-86), so a probability that is high
        # but second-place is exactly the anticipation being thrown away.
        # 0.0 disables — default keeps the previous argmax behaviour.
        self._drop_light_thr = drop_light_threshold
        self._drop_light_hold = drop_light_hold
        self._drop_light_remaining = 0
        # Periodic LSTM reset. State is otherwise carried for the whole session:
        # a 3h set is ~5,700 consecutive steps, while training sequences were
        # ~150-200 and always started from h=0. 0 disables.
        self._reset_state_beats = reset_state_beats
        self._downbeats_since_reset = 0
        # State Manager kept ONLY for its (good) mechanical countdown / drop-incoming
        # anticipation; its consensus-transition core spams for this model so we do
        # not emit its transition/correction events.
        self._state = PhraseStateManager(
            correction_threshold=correction_threshold,
            transition_beats=transition_beats,
            anticipate_beats=anticipate_beats,
            latch_after=latch_after,
            sticky_beats=sticky_beats,
            warmup_beats=warmup_beats,
        )

        # Manual drop-arm from a MIDI pad: press -> force "drop" starting at the
        # next downbeat for force_drop_beats, then resume model output.
        self._force_armed = False       # set by the MIDI thread on a pad press
        self._force_remaining = 0       # downbeats left in the forced window
        self._force_downbeats = max(1, force_drop_beats // 4)  # 4/4 bars
        # Label marks from a SECOND pad. This one only timestamps — it must not
        # force the phrase, or every marked bar reports zero latency by
        # construction and the marks are useless as ground truth.
        self._marks: list[dict] = []
        self._midi = None
        if midi_port:
            from audiovj.live.midi import MidiNoteListener
            handlers = {midi_note: self._arm_drop}
            if midi_mark_note is not None and midi_mark_note != midi_note:
                handlers[midi_mark_note] = self._mark_drop
            self._midi = MidiNoteListener(
                handlers=handlers, port_match=midi_port
            )

        self._downbeat_queue: queue.Queue[DownbeatEvent] = queue.Queue()
        self._carabiner = CarabinerClient(
            host=carabiner_host,
            port=carabiner_port,
            on_downbeat=self._downbeat_queue.put,
        )
        self._event_log_path = event_log
        self._event_log = None
        self._status_running = False
        self._display_phrase = ""
        self._display_next = ""
        self._countdown_at_downbeat: float | None = None
        self._countdown_phrase_display: str = ""
        self._sm_debug: str = ""

    def _arm_drop(self) -> None:
        """MIDI callback (background thread): arm a forced drop at the next downbeat."""
        self._force_armed = True

    def _mark_drop(self) -> None:
        """MIDI callback (background thread): timestamp a drop WITHOUT forcing it.

        Records the bar phase at press time. That phase is the whole point: it
        says whether you press *reacting* to a drop you already hear (phase near
        0, just after the downbeat) or *anticipating* one (phase near 3, late in
        the previous bar). Those two need opposite fixes, and nothing in the rig
        currently distinguishes them.
        """
        mark = {
            "t": time.monotonic(),
            "beat_phase": self._carabiner.beat_phase,
            "bpm": self._carabiner.bpm,
            "audio_pos": self._audio.write_pos,
        }
        self._marks.append(mark)
        self._log_event({"kind": "mark", **mark})

    def _log_event(self, rec: dict) -> None:
        """Append one JSON line to the session log, if enabled."""
        if self._event_log is None:
            return
        try:
            self._event_log.write(json.dumps(rec) + "\n")
            self._event_log.flush()
        except Exception:
            pass

    def _draw_status(self) -> None:
        """Draw the status bar on the reserved bottom 2 lines."""
        phase = self._carabiner.beat_phase
        bpm = self._carabiner.bpm
        channel_peaks = self._audio.channel_peaks
        beats = _beat_dots(phase)

        size = _term_size()
        if size is None:
            return  # piped output: the per-downbeat print lines carry everything
        cols, rows = size.columns, size.lines

        phrase_info = ""
        if self._display_phrase:
            phrase_info = f"  Current: {self._display_phrase}"
            if self._countdown_at_downbeat is not None and self._countdown_phrase_display:
                interpolated = max(0.0, self._countdown_at_downbeat - phase)
                phrase_info += f"   Next: {self._countdown_phrase_display} in {interpolated:.0f} beats"
            elif self._display_next:
                phrase_info += f"   Next: {self._display_next}"
        if self._sm_debug:
            phrase_info += f"   [SM {self._sm_debug}]"

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
        if self._event_log_path is not None:
            self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._event_log = self._event_log_path.open("a")
            print(f"Event log -> {self._event_log_path}")

        print("Starting audio capture...")
        self._audio.start()

        print("Connecting to Carabiner...")
        self._carabiner.start()

        if self._midi is not None and not self._midi.start():
            print(f"MIDI: no input port matched '{self._midi._port_match}' — manual drop-arm disabled")
        self._osc.send_status("running")
        self._log_event({
            "kind": "session_start",
            "t": time.monotonic(),
            "input_latency_s": self._audio.stream_latency,
            "link_peers": self._carabiner.peers,
            "bpm": self._carabiner.bpm,
            "drop_light_threshold": self._drop_light_thr,
        })
        print("Live pipeline running. Press Ctrl+C to stop.\n")

        # Set up scroll region and start status bar
        _setup_scroll_region()
        self._status_running = True
        status_thread = threading.Thread(target=self._status_loop, daemon=True)
        status_thread.start()

        try:
            while True:
                try:
                    # Bounded so a dead Carabiner surfaces as a message rather
                    # than a silent freeze that looks exactly like extreme lateness.
                    evt = self._downbeat_queue.get(timeout=5.0)
                except queue.Empty:
                    if not self._carabiner.alive:
                        print("No beat grid — Carabiner connection is down. Waiting...")
                    continue

                # Extract audio window: last CONTEXT_BEATS beats
                beat_duration = 60.0 / evt.bpm
                samples_needed = int(CONTEXT_BEATS * beat_duration * SAMPLE_RATE)
                audio_window = self._audio.read_last_n_samples(samples_needed)
                if self._auto_gain:
                    peak = float(np.abs(audio_window).max())
                    # instant attack, slow release -> follows the loud level
                    self._agc_level = max(peak, self._agc_level * self._agc_release)
                    if self._agc_level > self._agc_floor:
                        self._agc_gain = min(self._agc_target / self._agc_level, self._agc_max)
                    else:
                        self._agc_gain = 1.0
                    audio_window = audio_window * self._agc_gain
                if self._input_gain != 1.0:
                    audio_window = audio_window * self._input_gain

                # Periodic LSTM state reset (opt-in). Carrying one hidden state
                # for a whole set puts the model thousands of steps beyond
                # anything it was trained on.
                self._downbeats_since_reset += 1
                if (
                    self._reset_state_beats
                    and self._downbeats_since_reset * 4 >= self._reset_state_beats
                ):
                    self._engine.reset()
                    self._cue.reset()
                    self._downbeats_since_reset = 0
                    self._log_event({"kind": "state_reset", "t": evt.time})

                audio_pos = self._audio.write_pos
                prediction = self._engine.predict(audio_window, evt.bpm)
                probs = prediction.current_probs or ()
                p_drop = probs[_DROP_IDX] if probs else 0.0
                p_buildup = probs[_BUILDUP_IDX] if probs else 0.0

                # Manual drop-arm: a pad press arms the NEXT downbeat (this one),
                # then we hold "drop" for the configured window before handing
                # control back to the model.
                if self._force_armed:
                    self._force_armed = False
                    self._force_remaining = self._force_downbeats
                forced_drop = self._force_remaining > 0
                if forced_drop:
                    self._force_remaining -= 1
                # Drop lighting: either the raw 10-class argmax (default, matches
                # previous behaviour) or a p(drop) threshold with a hold, which
                # can fire a bar earlier when the drop is running second behind
                # buildup. The hold stops it chattering on a marginal probability.
                thresholded_drop = False
                if self._drop_light_thr > 0.0:
                    if p_drop >= self._drop_light_thr:
                        self._drop_light_remaining = self._drop_light_hold
                    elif self._drop_light_remaining > 0:
                        self._drop_light_remaining -= 1
                    thresholded_drop = self._drop_light_remaining > 0

                if forced_drop:
                    effective_phrase = "drop"
                elif thresholded_drop:
                    effective_phrase = "drop"
                else:
                    effective_phrase = prediction.current_phrase

                # Onset cueing owns transitions / drop_start / drop_end / buildup.
                cue_events = self._cue.update(prediction)
                # SM owns only the countdown / drop-incoming anticipation here.
                sm_events = self._state.update(prediction)

                events = list(cue_events)
                for event in cue_events:
                    self._osc.send_event(event)
                for event in sm_events:
                    if event.kind in ("anticipate", "phrase"):
                        self._osc.send_event(event)
                        events.append(event)
                self._osc.send_beat(evt.bpm)

                self._log_event({
                    "kind": "downbeat",
                    "t": evt.time,
                    "beat": evt.beat_number,
                    "bpm": evt.bpm,
                    "irregular": evt.irregular,
                    "audio_pos": audio_pos,
                    "agc_gain": self._agc_gain,
                    "current": prediction.current_phrase,
                    "current_conf": prediction.current_confidence,
                    "next": prediction.next_phrase,
                    "beats_until": prediction.beats_until,
                    "probs": list(probs),
                    "effective_phrase": effective_phrase,
                    "forced": forced_drop,
                    "events": [
                        {"kind": e.kind, "phrase": e.phrase, "conf": e.confidence}
                        for e in events
                    ],
                })

                # grandMA3: light the executor for the current phrase + sync BPM.
                # During a manual drop-arm the forced phrase wins over the model.
                if self._ma3 is not None:
                    self._ma3.set_phrase(effective_phrase)
                    self._ma3.set_bpm(evt.bpm)

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
                self._sm_debug = self._state.debug_str

                # Console output
                state_indicator = ""
                for event in events:
                    if event.kind == "drop_start":
                        state_indicator = "  >>> DROP START"
                    elif event.kind == "drop_end":
                        state_indicator = "  <<< DROP END"
                    elif event.kind == "buildup":
                        state_indicator = "  ^^^ BUILDUP"
                    elif event.kind == "transition":
                        state_indicator = f"  >>> {event.phrase.upper()}"
                    elif event.kind == "anticipate":
                        state_indicator = f"  ... {event.phrase} in ~{event.beats_until:.0f} beats"
                if forced_drop:
                    held = (self._force_remaining + 1) * 4
                    state_indicator = f"  !!! MANUAL DROP ({held} beats left)"

                # p(drop) is the number that decides whether the lateness is the
                # model failing to anticipate or the argmax hiding an anticipation
                # it already made. Print it every downbeat.
                grid_flag = " !GRID" if evt.irregular else ""
                print(
                    f"[{self._state.running_phrase:<12}] "
                    f"current={prediction.current_phrase:<12} ({prediction.current_confidence:.0%})  "
                    f"pDrop={p_drop:.2f} pBuild={p_buildup:.2f}  "
                    f"next={prediction.next_phrase:<12} ({prediction.next_confidence:.0%})  "
                    f"beats_until={prediction.beats_until:.0f}  "
                    f"gain={20.0 * math.log10(max(self._agc_gain, 1e-9)):+.0f}dB"
                    f"{grid_flag}{state_indicator}"
                )

        except KeyboardInterrupt:
            pass
        finally:
            self._status_running = False
            if self._ma3 is not None:
                self._ma3.all_off()
            self._osc.send_status("stopped")
            if self._midi is not None:
                self._midi.stop()
            self._carabiner.stop()
            self._audio.stop()
            self._log_event({"kind": "session_end", "t": time.monotonic(),
                             "marks": len(self._marks)})
            if self._event_log is not None:
                self._event_log.close()
                self._event_log = None
            _teardown_scroll_region()
            if self._marks:
                phases = [m["beat_phase"] for m in self._marks]
                avg = sum(phases) / len(phases)
                print(f"{len(self._marks)} drop marks, mean bar phase {avg:.2f} "
                      f"(near 0 = you press after the downbeat / reacting; "
                      f"near 3 = before it / anticipating)")
            print("Stopped.")
