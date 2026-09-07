"""grandMA3 OSC bridge — light one executor (201-208) for the current phrase.

Address format (grandMA3 remote OSC):  /<prefix>/Fader<exec>  <value>
(no Page segment -> targets the current page).
The prefix and the fader value range are set in MA3 (Menu > In & Out > OSC).
If your fader range is 0..1 use on_value=1; if 0..100 use on_value=100.
"""

from pythonosc.udp_client import SimpleUDPClient

# model phrase -> MA3 executor number (10-class vocab folds two rare variants)
PHRASE_TO_EXEC = {
    "intro": 201, "altintro": 201,
    "buildup": 202,
    "drop": 203,
    "breakdown": 204,
    "bridge": 205,
    "cooldown": 206,
    "outro": 207, "altoutro": 207,
    "end": 208,
}
EXECUTORS = [201, 202, 203, 204, 205, 206, 207, 208]


class GrandMA3PhraseBridge:
    """Lights exactly one executor for the current phrase, clears the rest."""

    def __init__(self, host: str, port: int = 8000,
                 prefix: str = "gma3", on_value: float = 1.0,
                 speedmaster: str = "3.1") -> None:
        self._osc = SimpleUDPClient(host, port)
        self._prefix = prefix.strip("/")
        self._on = on_value
        self._speedmaster = speedmaster  # "" disables BPM sync
        self._active_exec: int | None = None
        self._last_bpm: float | None = None

    def _fader(self, executor: int, value: float) -> None:
        # No Page segment -> MA3 targets the current page.
        self._osc.send_message(f"/{self._prefix}/Fader{executor}", float(value))

    def set_bpm(self, bpm: float) -> None:
        """Sync a SpeedMaster's tempo to bpm via the MA3 command line. De-duped to 0.1 BPM.

        Sends /<prefix>/cmd "Master <sm> BPM <bpm>" (needs OSC 'Receive Command' on in MA3).
        """
        if not self._speedmaster:
            return
        r = round(bpm, 1)
        if r == self._last_bpm:
            return
        self._osc.send_message(f"/{self._prefix}/cmd", f"Master {self._speedmaster} BPM {r}")
        self._last_bpm = r

    def set_phrase(self, phrase: str) -> None:
        """Set the phrase's executor to on_value and all others to 0. De-duped."""
        target = PHRASE_TO_EXEC.get(phrase)
        if target is None or target == self._active_exec:
            return
        for ex in EXECUTORS:
            self._fader(ex, self._on if ex == target else 0.0)
        self._active_exec = target

    def all_off(self) -> None:
        for ex in EXECUTORS:
            self._fader(ex, 0.0)
        self._active_exec = None
