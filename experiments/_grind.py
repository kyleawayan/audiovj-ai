"""Grind: lift drop_incoming / drop_start precision via musical-context gating,
without new data. A drop is almost always preceded by a buildup and never in
the intro, so gate the warning on recent context.

Measures recall (>=1 bar early for the warning; within 8 beats for starts) vs
precision for several gating strategies on the seq detection model (val).
"""

import torch
from safetensors.torch import load_file

from _arch import SeqPhrasePredictor, build_seqs
from _drop import debounce_state
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits

DROP = PHRASE_TYPES.index("drop")
BUILDUP = PHRASE_TYPES.index("buildup")
INTRO = PHRASE_TYPES.index("intro")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = "/mnt/scratch/data/loop/seq_predictor.safetensors"


def load_preds():
    _, val_ids = create_splits(TRACKS_DIR, FEATURES_DIR)
    m = SeqPhrasePredictor().to(DEV); m.load_state_dict(load_file(CKPT)); m.eval()
    out = []
    with torch.no_grad():
        for tid in val_ids:
            seqs = build_seqs([tid])
            if not seqs:
                continue
            s = seqs[0]
            o = m(s["windows"].to(DEV).unsqueeze(0))
            pcur = o.current_phrase_logits[0].argmax(-1).cpu().tolist()
            nconf, nidx = torch.softmax(o.next_phrase_logits[0], -1).max(-1)
            out.append({
                "pcur": pcur,
                "pnext": nidx.cpu().tolist(), "pconf": nconf.cpu().tolist(),
                "tnext": s["next"].tolist(), "tbu": s["beats"].tolist(),
                "bd": 60.0 / s["bpm"], "times": s["times"],
                "build": debounce_state(pcur, BUILDUP),
            })
    return out


def warn_eval(preds, gate):
    """gate(p, k) -> bool: may a warning fire at downbeat k? Measure warning
    recall (>=1 bar before a drop) and precision (warnings landing in a true
    run-in to a drop)."""
    n_drops = det = fires = true_fires = 0
    for p in preds:
        tnext = p["tnext"]; tbu = p["tbu"]
        warn = [False] * len(tnext); consec = 0
        for k in range(len(tnext)):
            hot = p["pnext"][k] == DROP and p["pconf"][k] > 0.5
            consec = consec + 1 if hot else 0
            warn[k] = consec >= 2 and gate(p, k)
            if warn[k] and not (k and warn[k - 1]):
                fires += 1
                if tnext[k] == DROP:
                    true_fires += 1
        i = 0
        while i < len(tnext):
            if tnext[i] == DROP:
                j = i
                while j < len(tnext) and tnext[j] == DROP:
                    j += 1
                n_drops += 1
                if any(warn[k] and tbu[k] >= 4 for k in range(i, j)):
                    det += 1
                i = j
            else:
                i += 1
    return det / max(n_drops, 1) * 100, true_fires / max(fires, 1) * 100, fires


def recent(state, k, window):
    return any(state[max(0, k - window):k + 1])


preds = load_preds()
gates = [
    ("no gate", lambda p, k: True),
    ("not in intro", lambda p, k: p["pcur"][k] != INTRO),
    ("buildup within 16db", lambda p, k: recent(p["build"], k, 16)),
    ("buildup within 8db", lambda p, k: recent(p["build"], k, 8)),
    ("in buildup now", lambda p, k: p["build"][k]),
]
print("drop_incoming gating (val):  recall / precision / #warnings\n")
for name, g in gates:
    rec, prec, fires = warn_eval(preds, g)
    print(f"  {name:<22} recall {rec:5.1f}%  precision {prec:5.1f}%  ({fires} warnings)")
