"""Onset-based cue tracker — the locked operating point for transitions.

The State Manager's consensus/correction core spams for the seq model (fires a
third of all downbeats at ~18% precision). The certified operating point instead
cues transitions on the ONSET of a load-bearing phrase: a rising edge of its
current-phrase probability past a threshold (default 0.30). This component runs
once per downbeat on the seq engine's PredictionResult (which carries the full
current-phrase prob vector) and emits:

  - "transition"  : a load-bearing phrase's prob just crossed the threshold
                    (phrase=drop is the tight drop-start cue)
  - "drop_start"  : debounced in-drop state turned ON  (confirmed, for visual hold)
  - "drop_end"    : debounced in-drop state turned OFF
  - "buildup"     : buildup's prob just crossed the threshold

Causal + content-only — no phrase-grid / track-position assumptions (see memory:
raveform-live-no-phrase-grid). The mechanical countdown / drop-incoming
anticipation stays with PhraseStateManager (its countdown is good: MAE ~3 beats,
monotone). Run both alongside each other in the live pipeline.
"""

from audiovj.config import PHRASE_TYPES
from audiovj.live.inference import PredictionResult
from audiovj.live.state import StateEvent

# Classes we cue on. ``breakdown`` is included so a drop->breakdown boundary
# emits a transition at all — without it the ONLY drop-exit signal is the
# debounced drop_end, which costs an extra bar.
_LB = ("intro", "buildup", "drop", "breakdown", "outro")
_LB_IDX = [PHRASE_TYPES.index(p) for p in _LB]
_BUILDUP = PHRASE_TYPES.index("buildup")
_DROP = PHRASE_TYPES.index("drop")


class OnsetCueTracker:
    """Per-downbeat onset cueing + debounced drop state. Call once per downbeat."""

    def __init__(
        self,
        onset_threshold: float = 0.30,
        drop_confirm: int = 1,
        drop_release: int = 2,
    ) -> None:
        """``drop_confirm`` downbeats of "drop" turn the state ON; ``drop_release``
        non-drop downbeats turn it OFF.

        These are deliberately ASYMMETRIC. A single shared value costs a full bar
        on entry (the event the rig cares most about), while dropping it to 1 on
        BOTH edges means one wobbly downbeat mid-drop emits drop_end and then
        drop_start a bar later — a visible flap. Fast in, slow out.
        """
        self._thr = onset_threshold
        self._confirm = drop_confirm
        self._release = drop_release
        self._prev: tuple[float, ...] | None = None
        self._in_drop = False
        self._drop_in = 0
        self._drop_out = 0

    @property
    def in_drop(self) -> bool:
        return self._in_drop

    def reset(self) -> None:
        self._prev = None
        self._in_drop = False
        self._drop_in = 0
        self._drop_out = 0

    def update(self, pred: PredictionResult) -> list[StateEvent]:
        events: list[StateEvent] = []
        probs = pred.current_probs
        if probs is None:
            # Onset cueing requires the full prob vector (seq engine). No-op otherwise.
            return events

        # --- onset transitions: rising edge of any load-bearing class past thr ---
        if self._prev is not None:
            best_c, best_v = -1, -1.0
            for c in _LB_IDX:
                if probs[c] >= self._thr and self._prev[c] < self._thr and probs[c] > best_v:
                    best_v, best_c = probs[c], c
            # A drop crossing outranks a simultaneous crossing of any other class:
            # adding `breakdown` to _LB must never let it mask the drop cue.
            if (
                best_c >= 0
                and best_c != _DROP
                and probs[_DROP] >= self._thr
                and self._prev[_DROP] < self._thr
            ):
                best_v, best_c = probs[_DROP], _DROP
            if best_c >= 0:
                events.append(StateEvent(
                    kind="transition", phrase=PHRASE_TYPES[best_c], confidence=best_v,
                ))
                if best_c == _BUILDUP:
                    events.append(StateEvent(kind="buildup", phrase="buildup", confidence=best_v))

        # --- debounced in-drop state -> drop_start / drop_end ---
        if pred.current_phrase == "drop":
            self._drop_in += 1; self._drop_out = 0
        else:
            self._drop_out += 1; self._drop_in = 0
        if not self._in_drop and self._drop_in >= self._confirm:
            self._in_drop = True
            events.append(StateEvent(kind="drop_start", phrase="drop",
                                     confidence=pred.current_confidence))
        elif self._in_drop and self._drop_out >= self._release:
            self._in_drop = False
            events.append(StateEvent(kind="drop_end", phrase="drop",
                                     confidence=pred.current_confidence))

        self._prev = probs
        return events
