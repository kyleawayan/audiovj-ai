"""Sweep State Manager configs over the SEQ model's predictions (val tracks).

The 72-config SM sweep earlier was on the baseline model. The seq model is
smoother, so the SM's chatter-suppression (sticky hold, thresholds, warmup)
likely over-suppresses and caps recall. Find the SM config that maximizes
boundary recall on the seq model, and compare to the direct-detector ceiling.
"""

import itertools

from _arch_eval import seq_cache
from _loop import score
from audiovj.config import FEATURES_DIR, TRACKS_DIR
from audiovj.data.dataset import create_splits
from audiovj.live.state import PhraseStateManager as SM

_, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
cache = seq_cache(val_ids)
print(f"seq val cache: {len(cache)} tracks\n")

grid = {
    "correction_threshold": [0.3, 0.4, 0.5],
    "latch_after": [1, 2],
    "sticky_beats": [0.0, 8.0, 16.0, 32.0],
    "warmup_beats": [0.0, 8.0],
}
keys = list(grid)
rows = []
for combo in itertools.product(*(grid[k] for k in keys)):
    cfg = dict(zip(keys, combo))
    m = score(cache, lambda cfg=cfg: SM(**cfg))
    rows.append((cfg, m))

rows.sort(key=lambda r: r[1]["recall"], reverse=True)
print("== top SM configs by recall (seq model, val) ==")
for cfg, m in rows[:8]:
    tag = f"ct{cfg['correction_threshold']} la{cfg['latch_after']} st{cfg['sticky_beats']:.0f} wu{cfg['warmup_beats']:.0f}"
    print(f"  {tag:<24} recall {m['recall']:4.1f}  prec {m['precision']:4.1f}  "
          f"sm-acc {m['sm']:4.1f} ({m['sm']-m['raw']:+.1f})  matched~ via fires {m['fires']}")

cur = SM()
base_m = score(cache, lambda: cur)
print(f"\n  current SM defaults: recall {base_m['recall']:.1f}  prec {base_m['precision']:.1f}")
print("  direct onset@0.4 detector ceiling (no SM): LB-recall ~63% / all-recall earlier ~50%")
