"""Task 10: end-to-end anticipation via a monotone-clamped countdown.

The accurate-but-jittery beats_until head (MAE ~2 near the boundary, monotone
only ~61%) is smoothed by a countdown that re-reads the head each downbeat but
never increases and decrements at least one bar/step:

    cd_t = clamp( min(cd_{t-1} - 4, round4(pred_bu_t)), 0, .. )

This keeps the head's near-boundary accuracy while guaranteeing monotonicity.
The countdown is "active" once next==drop is predicted for 2 consecutive
downbeats (consensus). Measures the 3 goal metrics on val.
"""

import sys

import torch
from safetensors.torch import load_file

from _arch import SeqPhrasePredictor, build_seqs
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits

DROP = PHRASE_TYPES.index("drop")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/scratch/data/loop/seq_antic.safetensors"


def runs_to_drop(true_next):
    spans, i, n = [], 0, len(true_next)
    while i < n:
        if true_next[i] == DROP:
            j = i
            while j < n and true_next[j] == DROP:
                j += 1
            spans.append((i, j)); i = j
        else:
            i += 1
    return spans


def main():
    _, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    m = SeqPhrasePredictor().to(DEV); m.load_state_dict(load_file(CKPT)); m.eval()

    n_drops = anticipated = 0
    leads, mae, mono_ok, mono_tot = [], [], 0, 0
    # warning operating points: (label, min_consec, conf)
    crits = [("loose argmax", 1, 0.0), ("conf>0.4", 1, 0.4), ("consensus2 conf>0.5", 2, 0.5)]
    warn_det = {c[0]: 0 for c in crits}
    warn_fire = {c[0]: 0 for c in crits}
    warn_true = {c[0]: 0 for c in crits}
    with torch.no_grad():
        for tid in val_ids:
            seqs = build_seqs([tid])
            if not seqs:
                continue
            s = seqs[0]
            o = m(s["windows"].to(DEV).unsqueeze(0))
            pbu = torch.expm1(o.beats_until[0, :, 0]).cpu().tolist()
            nconf, nidx = torch.softmax(o.next_phrase_logits[0], -1).max(-1)
            pnext = nidx.cpu().tolist(); pconf = nconf.cpu().tolist()
            tnext = s["next"].tolist(); tbu = s["beats"].tolist()

            # continuous monotone-clamped countdown + early-warning flag
            cd = None; consec = 0; disp = [None] * len(tnext); warn = [False] * len(tnext)
            for k in range(len(tnext)):
                drop_now = pnext[k] == DROP and pconf[k] > 0.5
                consec = consec + 1 if drop_now else 0
                warn[k] = consec >= 2  # consensus early-warning ("drop coming")
                r4 = max(round(pbu[k] / 4) * 4, 0)
                if cd is None:
                    if consec >= 2:
                        cd = max(r4, 4)
                else:
                    cd = min(cd - 4, r4) if drop_now else cd - 4
                    if cd <= 0:
                        cd = None
                disp[k] = cd

            # warning recall/precision per criterion
            for label, mc, cf in crits:
                wflag = [False] * len(tnext); cc = 0
                for k in range(len(tnext)):
                    cc = cc + 1 if (pnext[k] == DROP and pconf[k] > cf) else 0
                    wflag[k] = cc >= mc
                    if wflag[k] and not (k and wflag[k - 1]):  # rising edge = one warning event
                        warn_fire[label] += 1
                        if tnext[k] == DROP:
                            warn_true[label] += 1
                for a, b in runs_to_drop(tnext):
                    if any(wflag[k] and tbu[k] >= 4 for k in range(a, b)):
                        warn_det[label] += 1

            for a, b in runs_to_drop(tnext):
                n_drops += 1
                early = [tbu[k] for k in range(a, b) if warn[k] and tbu[k] >= 4]
                if early:
                    anticipated += 1; leads.append(max(early))
                for k in range(a, b):
                    if disp[k] is not None and 0 < tbu[k] <= 8:
                        mae.append(abs(disp[k] - tbu[k]))
                for k in range(a + 1, b):
                    if disp[k] is not None and disp[k - 1] is not None and tbu[k] <= 16 and tbu[k - 1] <= 16:
                        mono_tot += 1
                        if disp[k] <= disp[k - 1]:
                            mono_ok += 1

    print(f"val drops: {n_drops}   ckpt: {CKPT.split('/')[-1]}")
    print("GOAL: anticipate >=70% >=1 bar early | countdown MAE <=4 | monotone >=90%\n")
    print(f"  anticipation recall (>=1 bar early): {anticipated/max(n_drops,1)*100:.1f}%  ({anticipated}/{n_drops})")
    print(f"  mean lead when active:               {sum(leads)/max(len(leads),1):.1f} beats")
    print(f"  countdown MAE (final 2 bars):        {sum(mae)/max(len(mae),1):.2f} beats  (n={len(mae)})")
    print(f"  monotonicity (approach):             {mono_ok/max(mono_tot,1)*100:.1f}%  ({mono_ok}/{mono_tot})")
    print("\n  anticipation-warning operating points (recall / warning-precision):")
    for label, _, _ in crits:
        rec = warn_det[label] / max(n_drops, 1) * 100
        prec = warn_true[label] / max(warn_fire[label], 1) * 100
        print(f"    {label:<22} recall {rec:5.1f}%  warn-prec {prec:5.1f}%  ({warn_fire[label]} warnings)")


if __name__ == "__main__":
    main()
