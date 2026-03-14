"""OSC output emitter using python-osc."""

from pythonosc.udp_client import SimpleUDPClient

from audiovj.live.state import StateEvent


class OSCEmitter:
    """Sends resolved phrase events as OSC messages."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 9000, prefix: str = "/audiovj"
    ) -> None:
        self._client = SimpleUDPClient(host, port)
        self._prefix = prefix

    def send_event(self, event: StateEvent) -> None:
        """Send a StateEvent as an OSC message."""
        addr = f"{self._prefix}/{event.kind}"

        if event.kind == "phrase":
            self._client.send_message(addr, [event.phrase, event.confidence])

        elif event.kind in ("transition", "correction"):
            self._client.send_message(
                addr, [event.from_phrase or "", event.phrase, event.confidence]
            )

        elif event.kind == "anticipate":
            self._client.send_message(
                addr, [event.phrase, event.beats_until or 0.0, event.confidence]
            )

    def send_beat(self, bpm: float) -> None:
        """Send a beat sync message (every downbeat)."""
        self._client.send_message(f"{self._prefix}/beat", [bpm])

    def send_status(self, status: str) -> None:
        """Send a status message (running, stopped, error)."""
        self._client.send_message(f"{self._prefix}/status", [status])
