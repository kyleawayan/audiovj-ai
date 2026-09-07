"""MIDI input listener — fires a callback on a specific note press.

Used to let a controller pad manually arm a drop.

Uses mido's callback API rather than iterating the port on our own thread: rtmidi
delivers messages on its internal thread, so there is no blocking iterator to
tear down. Closing a port while a thread is blocked iterating it thrashes
CoreMIDI and makes the device appear to disconnect/reconnect.
"""

from collections.abc import Callable

import mido


class MidiNoteListener:
    """Calls ``on_note`` when a matching NOTE_ON (velocity > 0) arrives.

    Matches note ``note`` on any channel in ``channels`` (1-indexed, e.g. 1-4).
    """

    def __init__(
        self,
        on_note: Callable[[], None],
        port_match: str = "DDJ-GRV6",
        note: int = 61,
        channels: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        self._on_note = on_note
        self._port_match = port_match
        self._note = note
        self._channels = {c - 1 for c in channels}  # mido channels are 0-indexed
        self._port = None

    def _resolve_port(self) -> str | None:
        for name in mido.get_input_names():
            if self._port_match.lower() in name.lower():
                return name
        return None

    def _on_message(self, msg) -> None:
        if (
            msg.type == "note_on"
            and msg.velocity > 0
            and msg.note == self._note
            and msg.channel in self._channels
        ):
            self._on_note()

    def start(self) -> bool:
        """Open the port with a callback. Returns False if no port matched."""
        if self._port is not None:
            return True  # already listening
        name = self._resolve_port()
        if name is None:
            return False
        self._port = mido.open_input(name, callback=self._on_message)
        chans = sorted(c + 1 for c in self._channels)
        print(f"MIDI: listening on '{name}' note {self._note} ch {chans}")
        return True

    def stop(self) -> None:
        """Detach the callback and close the port. Safe to call more than once."""
        if self._port is None:
            return
        port, self._port = self._port, None
        try:
            port.callback = None  # stop delivery before closing
        except Exception:
            pass
        port.close()
