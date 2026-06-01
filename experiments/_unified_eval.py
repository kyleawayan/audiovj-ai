"""Verify the unified model on the headline VJ metrics (val)."""
import torch
from safetensors.torch import load_file

from _arch import build_seqs
from _drop import debounce_state, state_edges, edges, score_edges
from _unified import UnifiedSeq
from audiovj.config import FEATURES_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.dataset import create_splits

DROP = PHRASE_TYPES.index("drop"); BUILD = PHRASE_TYPES.index("buildup")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val = create_splits(TRACKS_DIR, FEATURES_DIR)
m = UnifiedSeq().to(DEV); m.load_state_dict(load_file("/mnt/scratch/data/loop/seq_unified.safetensors")); m.eval()

dtp = dfp = dfn = dtn = 0
st = [0, 0, [], 0, 0]
warn_det = warn_tot = 0; mae = []; mono_ok = mono_tot = 0
with torch.no_grad():
    for tid in val:
        seqs = build_seqs([tid])
        if not seqs:
            continue
        s = seqs[0]; times = s["times"]; bd = 60.0 / s["bpm"]
        o = m(s["windows"].to(DEV).unsqueeze(0))
        pcur = o.current_phrase_logits[0].argmax(-1).cpu().tolist()
        pbu = torch.expm1(o.beats_until[0, :, 0]).cpu().tolist()
        nconf, nidx = torch.softmax(o.next_phrase_logits[0], -1).max(-1)
        pnext = nidx.cpu().tolist(); pconf = nconf.cpu().tolist()
        tcur = s["current"].tolist(); tnext = s["next"].tolist(); tbu = s["beats"].tolist()

        ds = debounce_state(pcur, DROP)
        ts = [c == DROP for c in tcur]
        for k in range(len(tcur)):
            dtp += ds[k] and ts[k]; dfp += ds[k] and not ts[k]
            dfn += (not ds[k]) and ts[k]; dtn += (not ds[k]) and (not ts[k])
        tss, tse = state_edges(ts); pss, pse = state_edges(ds)
        d, t, lat, nf, f = score_edges(tss, pss, times, bd)
        st[0] += d; st[1] += t; st[2] += lat; st[3] += nf; st[4] += f

        # clamped countdown + warning
        cd = None; consec = 0
        for k in range(len(times)):
            hot = pnext[k] == DROP and pconf[k] > 0.5
            consec = consec + 1 if hot else 0
            r4 = max(round(pbu[k] / 4) * 4, 0)
            cd = (max(r4, 4) if consec >= 2 else None) if cd is None else (min(cd - 4, r4) if hot else cd - 4)
            if cd is not None and cd <= 0:
                cd = None
            # record
            if cd is not None and 0 < tbu[k] <= 8:
                mae.append(abs(cd - tbu[k]))
        i = 0
        while i < len(tnext):
            if tnext[i] == DROP:
                j = i
                while j < len(tnext) and tnext[j] == DROP:
                    j += 1
                warn_tot += 1
                if any(pnext[k] == DROP and pconf[k] > 0.5 and tbu[k] >= 4 for k in range(i, j)):
                    warn_det += 1
                i = j
            else:
                i += 1

acc = (dtp + dtn) / max(dtp + dfp + dfn + dtn, 1) * 100
print("UNIFIED model (val) — VJ metrics:")
print(f"  drop on/off: acc {acc:.1f}%  recall {dtp/max(dtp+dfn,1)*100:.1f}%  precision {dtp/max(dtp+dfp,1)*100:.1f}%")
print(f"  drop START : recall {st[0]/max(st[1],1)*100:.1f}%  latency {sum(st[2])/max(len(st[2]),1):.1f}b  precision {st[3]/max(st[4],1)*100:.1f}%")
print(f"  pre-drop warning (>=1 bar): recall {warn_det/max(warn_tot,1)*100:.1f}%")
print(f"  countdown MAE (final 2 bars, clamped): {sum(mae)/max(len(mae),1):.2f}b  (n={len(mae)})")
