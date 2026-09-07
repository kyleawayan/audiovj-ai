"""grandMA3 OSC — drive executors 201-208 from the current phrase.

When the model thinks we're in phrase X, its executor goes to the "on" value and
every other phrase's executor goes to 0. One executor lit at a time.

Address format (grandMA3 remote OSC):  /<prefix>/Fader<exec>  <value>
(no Page segment -> MA3 targets the current page).
  - prefix : whatever you set in MA3 (Menu > In & Out > OSC). Default here: "gma3".
  - value  : fader level. MA3's range is set in that same OSC config.
             If your fader range is 0..1  -> use --on-value 1   (default)
             If your fader range is 0..100-> use --on-value 100

Phrase -> executor map (201..208). Build your MA3 executors to match this order,
or edit the dict. The 10-class model vocab folds its two rare variants in:
  201 intro (+altintro)   205 bridge
  202 buildup             206 cooldown
  203 drop                207 outro (+altoutro)
  204 breakdown           208 end

Demo run (cycles through all 8 so you can watch them light up):
  uv run python experiments/_grandma3_phrase_osc.py
  uv run python experiments/_grandma3_phrase_osc.py --host 10.0.0.5 --on-value 100
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audiovj.live.grandma3 import PHRASE_TO_EXEC, GrandMA3PhraseBridge  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="grandMA3 console/onPC IP")
    ap.add_argument("--port", type=int, default=8000, help="MA3 OSC input port")
    ap.add_argument("--prefix", default="gma3", help="MA3 OSC prefix (Menu>In&Out>OSC)")
    ap.add_argument("--on-value", type=float, default=1.0,
                    help="fader value for the active phrase (1 if range 0..1, 100 if 0..100)")
    ap.add_argument("--interval", type=float, default=3.0)
    args = ap.parse_args()

    bridge = GrandMA3PhraseBridge(args.host, args.port, args.prefix, args.on_value)
    phrases = ["intro", "buildup", "drop", "breakdown",
               "bridge", "cooldown", "outro", "end"]
    print(f"OSC -> {args.host}:{args.port}  /{args.prefix.strip('/')}/Fader20x (current page)")
    print(f"on-value={args.on_value}. Cycling phrases; watch executors 201-208.\n")
    try:
        for p in phrases:
            bridge.set_phrase(p)
            print(f"  {p:10s} -> Fader{PHRASE_TO_EXEC[p]} = {args.on_value}  (others 0)")
            time.sleep(args.interval)
    finally:
        bridge.all_off()
        print("\nall executors -> 0. done.")


if __name__ == "__main__":
    main()
