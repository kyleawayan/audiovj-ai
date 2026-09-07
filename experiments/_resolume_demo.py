"""Minimal Resolume OSC demo — fire layer/clip changes, no model involved.

Cycles calm -> medium -> drop, connecting a RANDOM clip in each intensity layer.
Run it, watch Resolume react (or watch Resolume's OSC monitor if nothing is loaded).

Resolume setup (once):
  Preferences > OSC > "OSC Input"  -> Enable, note the port (this script: 7000).
  Preferences > Webserver          -> Enable (REST API, port 8080) if you want
                                      code to read how many clips a layer has.

Run:
  uv run python experiments/_resolume_demo.py
  uv run python experiments/_resolume_demo.py --host 127.0.0.1 --osc-port 7000
"""

import argparse
import json
import random
import time
import urllib.request

from pythonosc.udp_client import SimpleUDPClient

# intensity name -> Resolume layer number
LAYERS = {"calm": 1, "medium": 2, "drop": 3}


def clip_count(host: str, rest_port: int, layer: int) -> int | None:
    """Ask Resolume's REST API how many non-empty clips are in a layer.

    Returns None if the webserver isn't reachable (Resolume closed or REST off).
    """
    url = f"http://{host}:{rest_port}/api/v1/composition/layers/{layer}"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            data = json.load(resp)
    except Exception:
        return None
    # A loaded clip has a non-null video source; empty slots have video == None.
    return sum(1 for c in data.get("clips", []) if c.get("video") is not None)


def connect_random_clip(osc: SimpleUDPClient, host: str, rest_port: int,
                        layer: int, fallback_max: int) -> int:
    """Connect a random clip in `layer`. Uses REST count if available, else 1..fallback_max.

    Connecting an empty slot is a harmless no-op in Resolume, so the fallback is safe.
    """
    n = clip_count(host, rest_port, layer)
    hi = n if n else fallback_max
    clip = random.randint(1, max(hi, 1))
    osc.send_message(f"/composition/layers/{layer}/clips/{clip}/connect", 1)
    return clip


def set_layer_opacity(osc: SimpleUDPClient, layer: int, value: float) -> None:
    """0.0..1.0 opacity for a layer (use this to crossfade intensities / ramp a drop)."""
    osc.send_message(f"/composition/layers/{layer}/video/opacity/values", float(value))


def show(osc: SimpleUDPClient, host: str, rest_port: int, intensity: str,
         fallback_max: int) -> None:
    """Make one intensity active: full opacity + random clip; others faded out."""
    target = LAYERS[intensity]
    clip = connect_random_clip(osc, host, rest_port, target, fallback_max)
    for name, layer in LAYERS.items():
        set_layer_opacity(osc, layer, 1.0 if layer == target else 0.0)
    n = clip_count(host, rest_port, target)
    src = f"{n} clips (via REST)" if n else f"unknown count, random 1..{fallback_max}"
    print(f"  {intensity:6s} -> layer {target}, connect clip {clip}  [{src}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--osc-port", type=int, default=7000, help="Resolume OSC input port")
    ap.add_argument("--rest-port", type=int, default=8080, help="Resolume webserver port")
    ap.add_argument("--fallback-max", type=int, default=8,
                    help="clips to random over when REST is unavailable")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--cycles", type=int, default=3)
    args = ap.parse_args()

    osc = SimpleUDPClient(args.host, args.osc_port)
    print(f"OSC -> {args.host}:{args.osc_port}   REST probe -> {args.host}:{args.rest_port}")
    print("Cycling calm -> medium -> drop. Watch Resolume (or its OSC monitor).\n")
    for i in range(args.cycles):
        print(f"cycle {i + 1}/{args.cycles}")
        for intensity in ("calm", "medium", "drop"):
            show(osc, args.host, args.rest_port, intensity, args.fallback_max)
            time.sleep(args.interval)
    print("\ndone.")


if __name__ == "__main__":
    main()
