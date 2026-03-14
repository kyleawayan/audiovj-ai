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

    @property
    def running_phrase(self) -> str | None:
        return self._running_phrase

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
            events.append(StateEvent(
                kind="correction",
                phrase=prediction.current_phrase,
                from_phrase=old,
                confidence=prediction.current_confidence,
            ))

        # Transition: beats_until countdown reached
        if (
            prediction.beats_until <= self._transition_beats
            and prediction.next_phrase != self._running_phrase
            and prediction.next_confidence > self._correction_threshold
        ):
            old = self._running_phrase
            self._running_phrase = prediction.next_phrase
            self._last_anticipation_sent = None
            events.append(StateEvent(
                kind="transition",
                phrase=prediction.next_phrase,
                from_phrase=old,
                confidence=prediction.next_confidence,
            ))

        # Anticipate: forward cue when approaching transition
        if (
            prediction.beats_until <= self._anticipate_beats
            and prediction.beats_until > self._transition_beats
            and prediction.next_phrase != self._last_anticipation_sent
        ):
            self._last_anticipation_sent = prediction.next_phrase
            events.append(StateEvent(
                kind="anticipate",
                phrase=prediction.next_phrase,
                confidence=prediction.next_confidence,
                beats_until=prediction.beats_until,
            ))

        return events
