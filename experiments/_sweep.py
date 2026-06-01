"""SM config sweep over the cached predictions (instant; no GPU forward)."""

import itertools

from _loop import build_cache, fmt, score
from audiovj.live.state import PhraseStateManager as SM

cache = build_cache()

grid = {
    "correction_threshold": [0.4, 0.5, 0.6, 0.7],
    "latch_after": [2, 3],
    "sticky_beats": [16.0, 32.0, 48.0],
    "warmup_beats": [0.0, 8.0, 16.0],
}
keys = list(grid)
combos = list(itertools.product(*(grid[k] for k in keys)))
print(f"sweeping {len(combos)} configs over {len(cache)} tracks\n")

rows = []
for combo in combos:
    cfg = dict(zip(keys, combo))
    m = score(cache, lambda cfg=cfg: SM(**cfg))
    rows.append((cfg, m))

# Composite: recall is primary live-cueing signal; penalize timing > 2 bars (8b)
# and SM accuracy that drops more than 3pp below raw.
def obj(m):
    return m["recall"] - 1.5 * max(0.0, m["timing"] - 8.0) - 2.0 * max(0.0, (m["raw"] - m["sm"]) - 3.0)

rows.sort(key=lambda r: obj(r[1]), reverse=True)

print("== top 8 by composite (recall, low timing, SM-acc guardrail) ==")
for cfg, m in rows[:8]:
    tag = f"ct{cfg['correction_threshold']} la{cfg['latch_after']} st{cfg['sticky_beats']:.0f} wu{cfg['warmup_beats']:.0f}"
    print(fmt(tag, m))

print("\n== best recall ==")
for cfg, m in sorted(rows, key=lambda r: r[1]["recall"], reverse=True)[:3]:
    tag = f"ct{cfg['correction_threshold']} la{cfg['latch_after']} st{cfg['sticky_beats']:.0f} wu{cfg['warmup_beats']:.0f}"
    print(fmt(tag, m))

print("\n== best timing ==")
for cfg, m in sorted(rows, key=lambda r: r[1]["timing"])[:3]:
    tag = f"ct{cfg['correction_threshold']} la{cfg['latch_after']} st{cfg['sticky_beats']:.0f} wu{cfg['warmup_beats']:.0f}"
    print(fmt(tag, m))
