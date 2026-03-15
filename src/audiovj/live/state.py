"""Phrase State Manager: tracks running phrase and emits events."""

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
    """Maintains running_phrase and decides what events to emit each downbeat."""

    def __init__(
        self,
        correction_threshold: float = 0.7,
        transition_beats: float = 4.0,
        anticipate_beats: float = 8.0,
    ) -> None:
        self._correction_threshold = correction_threshold
        self._transition_beats = transition_beats
        self._anticipate_beats = anticipate_beats
        self._running_phrase: str | None = None
        self._last_anticipation_sent: str | None = None
        self._countdown: float | None = None
        self._countdown_phrase: str | None = None
        self._disagree_count: int = 0

    @property
    def running_phrase(self) -> str | None:
        return self._running_phrase

    @property
    def countdown(self) -> tuple[str, float] | None:
        """Returns (phrase, beats_remaining) if a countdown is active."""
        if self._countdown is not None and self._countdown_phrase is not None:
            return (self._countdown_phrase, self._countdown)
        return None

    def _reset_countdown(self) -> None:
        self._countdown = None
        self._countdown_phrase = None
        self._disagree_count = 0

    def update(self, prediction: PredictionResult) -> list[StateEvent]:
        """Process a prediction and return events to emit."""
        events: list[StateEvent] = []

        # Initialize on first downbeat
        if self._running_phrase is None:
            self._running_phrase = prediction.current_phrase

        # Always emit current phrase classification
        events.append(StateEvent(
            kind="phrase",
            phrase=prediction.current_phrase,
            confidence=prediction.current_confidence,
        ))

        # Correction: current-phrase head disagrees with running state
        if (
            prediction.current_phrase != self._running_phrase
            and prediction.current_confidence > self._correction_threshold
        ):
            old = self._running_phrase
            self._running_phrase = prediction.current_phrase
            self._last_anticipation_sent = None
            self._reset_countdown()
            events.append(StateEvent(
                kind="correction",
                phrase=prediction.current_phrase,
                from_phrase=old,
                confidence=prediction.current_confidence,
            ))

        # Countdown management
        if self._countdown is None:
            # Latch if model predicts a different phrase with confidence
            if (
                prediction.next_phrase != self._running_phrase
                and prediction.next_confidence > self._correction_threshold
                and prediction.beats_until > self._transition_beats
            ):
                self._countdown = prediction.beats_until
                self._countdown_phrase = prediction.next_phrase
                self._disagree_count = 0
        else:
            # Decrement by 4 (one bar per downbeat)
            self._countdown = max(self._countdown - 4.0, 0.0)

            # Re-latch if model consistently disagrees with latched phrase
            if (
                prediction.next_phrase != self._countdown_phrase
                and prediction.next_confidence > self._correction_threshold
            ):
                self._disagree_count += 1
                if self._disagree_count >= 2:
                    self._countdown = prediction.beats_until
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
