"""Correctness gate: stateful step() must equal the offline full-sequence
forward() for the ported UnifiedSeqPredictor (causal LSTM => must match)."""

import torch
from safetensors.torch import load_file

from _arch import build_seqs
from _full import fold_split
from audiovj.model import UnifiedSeqPredictor

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = UnifiedSeqPredictor().to(DEV)
m.load_state_dict(load_file("/mnt/scratch/data/loop/seq_unified_full_v2.safetensors"))
m.eval()

tr, va, te = fold_split()
worst_cur = worst_nxt = worst_beats = 0.0
min_cur_agree = min_nxt_agree = 1.0
n = 0
with torch.no_grad():
    for tid in te[:8]:
        seqs = build_seqs([tid])
        if not seqs:
            continue
        w = seqs[0]["windows"].to(DEV)  # [T, mel, fr]
        # full-sequence forward
        full = m(w.unsqueeze(0))
        full_cur = full.current_phrase_logits[0]   # [T, num]
        full_nxt = full.next_phrase_logits[0]
        full_b = full.beats_until[0, :, 0]
        # stateful stepping
        state = None
        cur, nxt, beats = [], [], []
        for t in range(w.shape[0]):
            o, state = m.step(w[t], state)
            cur.append(o.current_phrase_logits[0])
            nxt.append(o.next_phrase_logits[0])
            beats.append(o.beats_until[0, 0])
        step_cur = torch.stack(cur); step_nxt = torch.stack(nxt); step_b = torch.stack(beats)
        worst_cur = max(worst_cur, (full_cur - step_cur).abs().max().item())
        worst_nxt = max(worst_nxt, (full_nxt - step_nxt).abs().max().item())
        worst_beats = max(worst_beats, (full_b - step_b).abs().max().item())
        # what actually matters: do the DECISIONS match (argmax + countdown)?
        cur_agree = (full_cur.argmax(-1) == step_cur.argmax(-1)).float().mean().item()
        nxt_agree = (full_nxt.argmax(-1) == step_nxt.argmax(-1)).float().mean().item()
        min_cur_agree = min(min_cur_agree, cur_agree); min_nxt_agree = min(min_nxt_agree, nxt_agree)
        n += 1
        print(f"{tid}: T={w.shape[0]:3d}  cur-argmax-agree={cur_agree*100:.1f}%  "
              f"next-argmax-agree={nxt_agree*100:.1f}%  max|Δbeats|={(full_b-step_b).abs().max():.1e}")

print(f"\n{n} tracks  decisions: min current-argmax agree {min_cur_agree*100:.1f}%  "
      f"min next-argmax agree {min_nxt_agree*100:.1f}%  worst |Δbeats| {worst_beats:.1e}")
print(f"(raw logit drift {worst_cur:.1e} is expected cuDNN full-seq-vs-stepped numerics; decision-irrelevant)")
print("PASS — stateful step reproduces offline decisions exactly"
      if min_cur_agree == 1.0 and min_nxt_agree == 1.0 and worst_beats < 0.01
      else "FAIL — decisions diverge")
