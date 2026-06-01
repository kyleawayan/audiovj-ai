"""Phrase State Manager: tracks running phrase and emits events.

Ported forward from the KA-167 binary-drop-detection branch (consensus
latching, sticky hold, mechanical countdown, corrections-cancel-holds) and
extended for the 10-class Raveform vocab with a warmup period and richer
debug state for the live status bar.
"""

from dataclasses import dataclass

from audiovj.live.inference import PredictionResult


@dataclass
class StateEvent:
    kind: str  # "phrase" | "transition" | "correction" | "anticipate"
    phrase: str
    from_phrase: str | None = None
    confidence: float = 0.0
    beats_until: float | None = None


class PhraseStateManager:
    """Maintains running_phrase and decides what events to emit each downbeat.

    Countdown uses a consensus-based approach: after the model consistently
    predicts a transition for ``latch_after`` consecutive downbeats, the SM
    latches a countdown seeded from the model's ``beats_until`` (clamped to a
    reasonable range). The countdown then decrements *mechanically* by 4 beats
    per downbeat — the model's ``beats_until`` regression head is architecturally
    unreliable (KA-167 / KA-233), so once latched we never consult it again for
    the countdown itself.

    After a transition fires, the new phrase is "sticky" for ``sticky_beats``
    beats: corrections away from it are suppressed unless the model disagrees
    with high confidence for ``sticky_override_needed`` consecutive downbeats.

    A ``warmup_beats`` window at track start keeps ``running_phrase`` tracking the
    raw current-phrase prediction but suppresses the transition / correction /
    countdown machinery, so cold-start garbage predictions can't fire spurious
    transitions before the model has real audio context.

    ``update`` must be called exactly once per downbeat (one 4-beat bar); the
    mechanical countdown and sticky decrements both assume that cadence.
    """

    def __init__(
        self,
        correction_threshold: float = 0.5,
        transition_beats: float = 4.0,
        anticipate_beats: float = 8.0,
        latch_after: int = 2,
        sticky_beats: float = 32.0,
        sticky_override_needed: int = 3,
        warmup_beats: float = 16.0,
    ) -> None:
        self._correction_threshold = correction_threshold
        self._transition_beats = transition_beats
        self._anticipate_beats = anticipate_beats
        self._latch_after = latch_after
        self._sticky_beats = sticky_beats
        self._sticky_override_needed = sticky_override_needed
        self._warmup_beats = warmup_beats

        self._running_phrase: str | None = None
        self._last_anticipation_sent: str | None = None
        self._countdown: float | None = None
        self._countdown_phrase: str | None = None
        self._agree_count: int = 0
        self._agree_phrase: str | None = None
        self._agree_beats_sum: float = 0.0
        self._disagree_count: int = 0
        self._sticky_remaining: float = 0.0
        self._sticky_override_count: int = 0
        self._beats_elapsed: float = 0.0

    @property
    def running_phrase(self) -> str | None:
        return self._running_phrase

    @property
    def countdown(self) -> tuple[str, float] | None:
        """Returns (phrase, beats_remaining) if a countdown is active."""
        if self._countdown is not None and self._countdown_phrase is not None:
            return (self._countdown_phrase, self._countdown)
        return None

    @property
    def sticky_remaining(self) -> float:
        return self._sticky_remaining

    @property
    def warmup_remaining(self) -> float:
        """Beats of warmup left before the decision machinery engages."""
        return max(self._warmup_beats - self._beats_elapsed, 0.0)

    @property
    def debug_state(self) -> dict:
        """Machine-readable SM internals for instrumentation / status bar."""
        return {
            "running": self._running_phrase,
            "countdown": self._countdown,
            "countdown_phrase": self._countdown_phrase,
            "agree": self._agree_count,
            "latch_after": self._latch_after,
            "disagree": self._disagree_count,
            "sticky": self._sticky_remaining,
            "override": self._sticky_override_count,
            "override_needed": self._sticky_override_needed,
            "warmup_remaining": self.warmup_remaining,
        }

    @property
    def debug_str(self) -> str:
        """Compact one-line SM debug string for the live status bar."""
        parts: list[str] = []
        if self.warmup_remaining > 0:
            parts.append(f"warmup {self.warmup_remaining:.0f}")
        if self._countdown is not None:
            parts.append(f"cd {self._countdown:.0f}→{self._countdown_phrase}")
        if self._agree_count:
            parts.append(f"agree {self._agree_count}/{self._latch_after}")
        if self._disagree_count:
            parts.append(f"dis {self._disagree_count}/3")
        if self._sticky_remaining > 0:
            parts.append(f"hold {self._sticky_remaining:.0f}")
        if self._sticky_override_count:
            parts.append(f"ovr {self._sticky_override_count}/{self._sticky_override_needed}")
        return " ".join(parts)

    def _reset_countdown(self) -> None:
        self._countdown = None
        self._countdown_phrase = None
        self._agree_count = 0
        self._agree_phrase = None
        self._agree_beats_sum = 0.0
        self._disagree_count = 0

    def update(self, prediction: PredictionResult) -> list[StateEvent]:
        """Process a prediction and return events to emit."""
        events: list[StateEvent] = []

        self._beats_elapsed += 4.0  # one downbeat == one 4-beat bar
        in_warmup = self._beats_elapsed <= self._warmup_beats

        # Initialize on first downbeat
        if self._running_phrase is None:
            self._running_phrase = prediction.current_phrase

        # Tick down sticky hold
        if self._sticky_remaining > 0:
            self._sticky_remaining = max(self._sticky_remaining - 4.0, 0.0)

        # Always emit current phrase classification
        events.append(StateEvent(
            kind="phrase",
            phrase=prediction.current_phrase,
            confidence=prediction.current_confidence,
        ))

        # During warmup, follow the raw current prediction but suppress the
        # transition / correction / countdown machinery (cold-start predictions
        # are unreliable and would otherwise fire garbage transitions).
        if in_warmup:
            self._running_phrase = prediction.current_phrase
            return events

        # Correction: current-phrase head disagrees with running state.
        # During sticky hold, require N consecutive high-confidence disagreements
        # before breaking the hold; outside the hold, accept immediately.
        if (
            prediction.current_phrase != self._running_phrase
            and prediction.current_confidence > self._correction_threshold
        ):
            if self._sticky_remaining > 0:
                self._sticky_override_count += 1
                if self._sticky_override_count >= self._sticky_override_needed:
                    old = self._running_phrase
                    self._running_phrase = prediction.current_phrase
                    self._last_anticipation_sent = None
                    self._sticky_remaining = 0.0
                    self._sticky_override_count = 0
                    self._reset_countdown()
                    events.append(StateEvent(
                        kind="correction",
                        phrase=prediction.current_phrase,
                        from_phrase=old,
                        confidence=prediction.current_confidence,
                    ))
            else:
                old = self._running_phrase
                self._running_phrase = prediction.current_phrase
                self._last_anticipation_sent = None
                self._sticky_override_count = 0
                self._reset_countdown()
                events.append(StateEvent(
                    kind="correction",
                    phrase=prediction.current_phrase,
                    from_phrase=old,
                    confidence=prediction.current_confidence,
                ))
        else:
            self._sticky_override_count = 0

        # Countdown management
        if self._countdown is None:
            # Track consecutive agreement before latching
            if (
                prediction.next_phrase != self._running_phrase
                and prediction.next_confidence > self._correction_threshold
            ):
                # Only count toward consensus when model predicts a near transition
                if prediction.beats_until <= 32.0:
                    if prediction.next_phrase == self._agree_phrase:
                        self._agree_count += 1
                        self._agree_beats_sum += max(prediction.beats_until, 0.0)
                    else:
                        self._agree_phrase = prediction.next_phrase
                        self._agree_count = 1
                        self._agree_beats_sum = max(prediction.beats_until, 0.0)

                    # Latch after enough consecutive agreements
                    if self._agree_count >= self._latch_after:
                        avg_pred = self._agree_beats_sum / self._agree_count
                        # Clamp to a reasonable range; the regression head is only
                        # used here to seed the countdown, never to drive it.
                        latch_value = max(avg_pred, self._transition_beats + 4.0)
                        latch_value = min(latch_value, 32.0)
                        self._countdown = latch_value
                        self._countdown_phrase = self._agree_phrase
                        self._agree_count = 0
                        self._agree_phrase = None
                        self._agree_beats_sum = 0.0
                else:
                    # Model predicts far away — reset consensus
                    self._agree_count = 0
                    self._agree_phrase = None
                    self._agree_beats_sum = 0.0
            else:
                self._agree_count = 0
                self._agree_phrase = None
                self._agree_beats_sum = 0.0
        else:
            # Pure mechanical countdown: decrement by 4 (one bar per downbeat)
            self._countdown = max(self._countdown - 4.0, 0.0)

            # Re-latch if model consistently disagrees with latched phrase
            if (
                prediction.next_phrase != self._countdown_phrase
                and prediction.next_confidence > self._correction_threshold
            ):
                self._disagree_count += 1
                if self._disagree_count >= 3:
                    if prediction.next_phrase == self._running_phrase:
                        # Model says we're staying — cancel countdown
                        self._reset_countdown()
                    else:
                        # Model says different transition — re-latch
                        self._countdown = min(max(prediction.beats_until, 8.0), 48.0)
                        self._countdown_phrase = prediction.next_phrase
                        self._disagree_count = 0
            else:
                self._disagree_count = 0

        # Transition: countdown reached threshold
        if (
            self._countdown is not None
            and self._countdown <= self._transition_beats
            and self._countdown_phrase is not None
            and self._countdown_phrase != self._running_phrase
        ):
            old = self._running_phrase
            self._running_phrase = self._countdown_phrase
            self._last_anticipation_sent = None
            self._sticky_remaining = self._sticky_beats
            events.append(StateEvent(
                kind="transition",
                phrase=self._countdown_phrase,
                from_phrase=old,
                confidence=prediction.next_confidence,
            ))
            self._reset_countdown()

        # Anticipate: forward cue when approaching transition
        if (
            self._countdown is not None
            and self._countdown <= self._anticipate_beats
            and self._countdown > self._transition_beats
            and self._countdown_phrase != self._last_anticipation_sent
        ):
            self._last_anticipation_sent = self._countdown_phrase
            events.append(StateEvent(
                kind="anticipate",
                phrase=self._countdown_phrase or "",
                confidence=prediction.next_confidence,
                beats_until=self._countdown,
            ))

        return events
