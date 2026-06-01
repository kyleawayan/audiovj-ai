"""Wiring smoke (no audio hardware): imports + OSC new kinds + cue tracker."""
import importlib

mods = ["audiovj.model", "audiovj.live.inference", "audiovj.live.cue",
        "audiovj.live.osc", "audiovj.live.state", "audiovj.cli"]
for m in mods:
    try:
        importlib.import_module(m); print(f"import OK: {m}")
    except Exception as e:
        print(f"IMPORT FAIL: {m}: {type(e).__name__}: {e}")
try:
    import audiovj.live.pipeline  # noqa
    print("import OK: audiovj.live.pipeline")
except Exception as e:
    print(f"pipeline import note: {type(e).__name__}: {e}")

from audiovj.config import PHRASE_TYPES
from audiovj.live.cue import OnsetCueTracker
from audiovj.live.inference import PredictionResult
from audiovj.live.osc import OSCEmitter
from audiovj.live.state import StateEvent

osc = OSCEmitter("127.0.0.1", 9999)
for k in ("drop_start", "drop_end", "buildup", "transition", "anticipate", "phrase"):
    osc.send_event(StateEvent(kind=k, phrase="drop", from_phrase="buildup",
                              confidence=0.8, beats_until=8.0))
print("OSC send_event OK for all kinds incl. drop_start/drop_end/buildup")


def probs(cls, p=0.6):
    v = [(1 - p) / (len(PHRASE_TYPES) - 1)] * len(PHRASE_TYPES)
    v[PHRASE_TYPES.index(cls)] = p
    return tuple(v)


trk = OnsetCueTracker(onset_threshold=0.30)
seq = ["intro", "buildup", "drop", "drop", "drop", "breakdown", "breakdown"]
for c in seq:
    pr = PredictionResult(c, 0.6, c, 0.6, 8.0, current_probs=probs(c))
    print(f"  {c:10s} -> {[e.kind for e in trk.update(pr)]}")
