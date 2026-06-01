"""Anticipation baseline: how well does the seq model's beats_until / next_phrase
predict the run-in to a DROP?

Metrics (val tracks), grouped per maximal run-in (consecutive downbeats whose
true next boundary is a drop):
  - anticipation recall: fraction of drops where the model predicts next==drop
    at least 1 bar (4 beats) before the boundary
  - mean lead: how many beats early the drop is first called (when called)
  - countdown MAE (final 2 bars): |pred_beats_until - true| for true_bu in (0,8]
  - monotonicity: over each run-in, fraction of steps where pred_beats_until
    decreases as the drop approaches
"""

import sys

import torch
from safetensors.torch import load_file

from _arch import CKPT, SeqPhrasePredictor, build_seqs
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits

CKPT = sys.argv[1] if len(sys.argv) > 1 else CKPT

DROP = PHRASE_TYPES.index("drop")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def runs_to_drop(true_next):
    """Yield (start, end) index spans of maximal consecutive true_next==drop."""
    spans, i, n = [], 0, len(true_next)
    while i < n:
        if true_next[i] == DROP:
            j = i
            while j < n and true_next[j] == DROP:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def main():
    _, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    m = SeqPhrasePredictor().to(DEV); m.load_state_dict(load_file(CKPT)); m.eval()

    n_drops = anticipated = 0
    leads, mae_pairs, mono_ok, mono_tot = [], [], 0, 0
    with torch.no_grad():
        for tid in val_ids:
            seqs = build_seqs([tid])
            if not seqs:
                continue
            s = seqs[0]
            o = m(s["windows"].to(DEV).unsqueeze(0))
            pred_bu = torch.expm1(o.beats_until[0, :, 0]).cpu().tolist()
            pred_next = o.next_phrase_logits[0].argmax(-1).cpu().tolist()
            true_next = s["next"].tolist()
            true_bu = s["beats"].tolist()

            for a, b in runs_to_drop(true_next):
                n_drops += 1
                # anticipation: next==drop predicted >=4 beats before boundary
                called = [true_bu[k] for k in range(a, b) if pred_next[k] == DROP and true_bu[k] >= 4]
                if called:
                    anticipated += 1
                    leads.append(max(called))
                # countdown MAE over final 2 bars (true_bu in (0,8])
                for k in range(a, b):
                    if 0 < true_bu[k] <= 8:
                        mae_pairs.append(abs(pred_bu[k] - true_bu[k]))
                # monotonicity over the APPROACH (both downbeats within final 4
                # bars, true_bu <= 16) — pred_bu should fall as the drop nears.
                for k in range(a + 1, b):
                    if true_bu[k] <= 16 and true_bu[k - 1] <= 16:
                        mono_tot += 1
                        if pred_bu[k] <= pred_bu[k - 1]:
                            mono_ok += 1

    print(f"val drops: {n_drops}")
    print(f"GOAL: anticipate >=70% >=1 bar early | countdown MAE <=4 beats | monotone >=90%\n")
    print(f"  anticipation recall (>=1 bar early): {anticipated/max(n_drops,1)*100:.1f}%  ({anticipated}/{n_drops})")
    print(f"  mean lead when called:               {sum(leads)/max(len(leads),1):.1f} beats")
    print(f"  countdown MAE (final 2 bars):        {sum(mae_pairs)/max(len(mae_pairs),1):.1f} beats  (n={len(mae_pairs)})")
    print(f"  monotonicity:                        {mono_ok/max(mono_tot,1)*100:.1f}%  ({mono_ok}/{mono_tot})")


if __name__ == "__main__":
    main()
