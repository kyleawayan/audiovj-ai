"""Drop-centric capability check: can we switch visuals on drop START and drop
END, and get >=1 measure (4 beats) of warning before the drop?

Uses the detection seq model. Ground-truth drop edges come from the true
current-phrase sequence; detection from predicted current-phrase flips into/out
of drop. Pre-drop warning from the next-phrase head (next==drop).

Reports, per edge: recall (within 8 beats), matched latency, precision.
"""

import sys

import torch
from safetensors.torch import load_file

from _arch import SeqPhrasePredictor, build_seqs
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits

DROP = PHRASE_TYPES.index("drop")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/scratch/data/loop/seq_predictor.safetensors"
NEAR = 8.0  # beats (2 bars) tolerance


def edges(seq, val):
    """downbeat indices where (current==val) starts / ends (raw, flickery)."""
    starts, ends = [], []
    for k in range(len(seq)):
        is_v = seq[k] == val
        prev = seq[k - 1] == val if k else False
        if is_v and not prev:
            starts.append(k)
        if not is_v and prev:
            ends.append(k)
    return starts, ends


def debounce_state(seq, val, confirm=2):
    """Per-downbeat debounced bool state for `val`: enter/leave only after
    `confirm` consecutive agreeing downbeats (kills argmax flicker)."""
    state = [False] * len(seq)
    in_state = False; c_in = 0; c_out = 0
    for k in range(len(seq)):
        if seq[k] == val:
            c_in += 1; c_out = 0
        else:
            c_out += 1; c_in = 0
        if not in_state and c_in >= confirm:
            in_state = True
        elif in_state and c_out >= confirm:
            in_state = False
        state[k] = in_state
    return state


def state_edges(state):
    starts = [k for k in range(len(state)) if state[k] and not (k and state[k - 1])]
    ends = [k for k in range(len(state)) if not state[k] and k and state[k - 1]]
    return starts, ends


def score_edges(true_idx, pred_idx, times, bd):
    """recall, matched latency, precision for one edge type on one track."""
    det = 0; lat = []
    for ti in true_idx:
        if pred_idx:
            d = min(abs(times[pi] - times[ti]) / bd for pi in pred_idx)
            if d <= NEAR:
                det += 1; lat.append(d)
    near_fires = sum(1 for pi in pred_idx
                     if true_idx and min(abs(times[pi] - times[ti]) / bd for ti in true_idx) <= NEAR)
    return det, len(true_idx), lat, near_fires, len(pred_idx)


BUILDUP = PHRASE_TYPES.index("buildup")


def main():
    _, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    m = SeqPhrasePredictor().to(DEV); m.load_state_dict(load_file(CKPT)); m.eval()

    # per-state frame-level confusion + edge timing
    frame = {"drop": [0, 0, 0, 0], "buildup": [0, 0, 0, 0]}  # tp, fp, fn, tn
    edge = {("drop", "start"): [0, 0, [], 0, 0], ("drop", "end"): [0, 0, [], 0, 0]}
    warn_det = warn_tot = 0; leads = []
    with torch.no_grad():
        for tid in val_ids:
            seqs = build_seqs([tid])
            if not seqs:
                continue
            s = seqs[0]; times = s["times"]; bd = 60.0 / s["bpm"]
            o = m(s["windows"].to(DEV).unsqueeze(0))
            pcur = o.current_phrase_logits[0].argmax(-1).cpu().tolist()
            pnext = o.next_phrase_logits[0].argmax(-1).cpu().tolist()
            tcur = s["current"].tolist(); tbu = s["beats"].tolist(); tnext = s["next"].tolist()

            for name, val in (("drop", DROP), ("buildup", BUILDUP)):
                pstate = debounce_state(pcur, val)
                tstate = [c == val for c in tcur]
                for k in range(len(tcur)):
                    if pstate[k] and tstate[k]:
                        frame[name][0] += 1
                    elif pstate[k] and not tstate[k]:
                        frame[name][1] += 1
                    elif not pstate[k] and tstate[k]:
                        frame[name][2] += 1
                    else:
                        frame[name][3] += 1
                if name == "drop":
                    ts, _ = edges(tcur, DROP); te_starts, te_ends = state_edges([c == DROP for c in tcur])
                    ps, pe = state_edges(pstate)
                    for key, ti, pi in (("start", te_starts, ps), ("end", te_ends, pe)):
                        d, t, lat, nf, f = score_edges(ti, pi, times, bd)
                        e = edge[("drop", key)]
                        e[0] += d; e[1] += t; e[2] += lat; e[3] += nf; e[4] += f

            i = 0
            while i < len(tnext):
                if tnext[i] == DROP:
                    j = i
                    while j < len(tnext) and tnext[j] == DROP:
                        j += 1
                    warn_tot += 1
                    called = [tbu[k] for k in range(i, j) if pnext[k] == DROP and tbu[k] >= 4]
                    if called:
                        warn_det += 1; leads.append(max(called))
                    i = j
                else:
                    i += 1

    print(f"detection model: {CKPT.split('/')[-1]}   (debounced state, {NEAR:.0f}-beat edge tolerance)\n")
    print("  FRAME-LEVEL 'are we in X right now' (per downbeat):")
    for name in ("drop", "buildup"):
        tp, fp, fn, tn = frame[name]
        acc = (tp + tn) / max(tp + fp + fn + tn, 1) * 100
        prec = tp / max(tp + fp, 1) * 100; rec = tp / max(tp + fn, 1) * 100
        spec = tn / max(tn + fp, 1) * 100
        print(f"    {name:<8} accuracy {acc:4.1f}%  in-state precision {prec:4.1f}%  recall {rec:4.1f}%  "
              f"not-{name} specificity {spec:4.1f}%")
    print("\n  DROP EDGE TIMING (visual switch points):")
    for key in ("start", "end"):
        d, t, lat, nf, f = edge[("drop", key)]
        print(f"    drop {key.upper():<5}: recall {d/max(t,1)*100:4.1f}% ({d}/{t})  "
              f"latency {sum(lat)/max(len(lat),1):.1f}b  precision {nf/max(f,1)*100:4.1f}% ({f} fires)")
    print(f"\n  PRE-DROP WARNING (>=1 measure / 4 beats early): "
          f"recall {warn_det/max(warn_tot,1)*100:4.1f}% ({warn_det}/{warn_tot})  mean lead {sum(leads)/max(len(leads),1):.0f}b")


if __name__ == "__main__":
    main()
