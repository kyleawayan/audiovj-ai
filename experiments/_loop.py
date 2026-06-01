"""Feedback-loop harness for the Raveform phrase predictor (KA-233/KA-234).

The model forward is identical across State Manager configs, so we run the
model ONCE, cache per-downbeat predictions for every track, then sweep SM
configs over the cache near-instantly. Operates on whatever tracks have
precomputed features (the 150-track subset here). GPU via the package CUDA
bootstrap.
"""

import os
import pickle
import sys

import torch
from safetensors.torch import load_file

from audiovj.config import FEATURES_DIR, FIXED_FRAMES, MODELS_DIR, PHRASE_TYPES, TRACKS_DIR
from audiovj.data.rekordbox import load_tracks
from audiovj.evaluate import _countdown_quality, _track_windows
from audiovj.live.inference import PredictionResult
from audiovj.model import PhrasePredictor

CACHE = "/mnt/scratch/data/loop/pred_cache.pkl"
NEAR = 8.0  # beats; a fired change within NEAR of a boundary counts as detected


def build_cache(checkpoint=None):
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhrasePredictor()
    ckpt = checkpoint or str(MODELS_DIR / "phrase_predictor.safetensors")
    model.load_state_dict(load_file(ckpt))
    model.to(device).eval()

    tracks = [t for t in load_tracks(TRACKS_DIR)
              if t.cue_points and (FEATURES_DIR / f"{t.track_id}.safetensors").exists()]
    cache = []
    with torch.no_grad():
        for n, track in enumerate(tracks, 1):
            samples = _track_windows(track, FEATURES_DIR)
            if not samples:
                continue
            rows = []
            for t, lbl, window in samples:
                w = window.unsqueeze(0)
                fr = w.shape[-1]
                pad = ((fr + FIXED_FRAMES - 1) // FIXED_FRAMES) * FIXED_FRAMES
                if pad > fr:
                    w = torch.nn.functional.pad(w, (0, pad - fr))
                out = model(w.to(device))
                npb = torch.softmax(out.next_phrase_logits, -1)
                cpb = torch.softmax(out.current_phrase_logits, -1)
                ni, ci = npb.argmax(-1).item(), cpb.argmax(-1).item()
                pred = PredictionResult(PHRASE_TYPES[ci], cpb[0, ci].item(),
                                        PHRASE_TYPES[ni], npb[0, ni].item(),
                                        torch.expm1(out.beats_until[0, 0]).item())
                rows.append((t, lbl, pred))
            cache.append({
                "bpm": track.bpm,
                "cue_times": [c.start_time for c in track.cue_points],
                "actual_transitions": max(len(track.cue_points) - 1, 0),
                "rows": rows,
            })
            if n % 25 == 0 or n == len(tracks):
                print(f"  cached {n}/{len(tracks)} tracks", flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def score(cache, make_sm):
    """Run an SM (made fresh per track) over the cached predictions."""
    raw_n = sm_n = total = 0
    actual = detected = fires = 0
    timing = []
    cdg, cdp = [], []
    for tc in cache:
        sm = make_sm()
        bd = 60.0 / tc["bpm"]
        cue = tc["cue_times"]
        change_times = []
        for t, lbl, pred in tc["rows"]:
            evs = sm.update(pred)
            total += 1
            raw_n += pred.current_phrase == lbl["current_phrase"]
            sm_n += sm.running_phrase == lbl["current_phrase"]
            cd = sm.countdown
            if cd is not None and lbl.get("beats_until") is not None:
                cdg.append(float(lbl["beats_until"]))
                cdp.append(float(cd[1]))
            for e in evs:
                if e.kind in ("transition", "correction"):
                    change_times.append(t)
                    timing.append(min(abs(ct - t) / bd for ct in cue))
        actual += tc["actual_transitions"]
        fires += len(change_times)
        for ct in cue[1:]:
            if any(abs(ct - ft) / bd <= NEAR for ft in change_times):
                detected += 1
    cq = _countdown_quality(cdg, cdp)
    return {
        "raw": raw_n / max(total, 1) * 100,
        "sm": sm_n / max(total, 1) * 100,
        "recall": detected / max(actual, 1) * 100,
        "precision": sum(1 for e in timing if e <= NEAR) / max(fires, 1) * 100,
        "timing": sum(timing) / max(len(timing), 1),
        "fires": fires,
        "cd_mae": cq["mae"],
        "cd_corr": cq["corr"],
        "cd_mono": cq["monotonicity"] * 100,
    }


def fmt(name, m):
    return (f"{name:<28} raw {m['raw']:4.1f} sm {m['sm']:4.1f} ({m['sm']-m['raw']:+4.1f}) | "
            f"recall {m['recall']:4.1f} prec {m['precision']:4.1f} time {m['timing']:4.1f} | "
            f"cd_mae {m['cd_mae']:4.1f} mono {m['cd_mono']:3.0f}% fires {m['fires']}")


if __name__ == "__main__":
    import importlib.util

    cache = build_cache()
    print(f"\ncache: {len(cache)} tracks, {sum(len(t['rows']) for t in cache)} downbeats\n")

    spec = importlib.util.spec_from_file_location("state_baseline", "/tmp/state_baseline.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    OldSM = base.PhraseStateManager
    from audiovj.live.state import PhraseStateManager as NewSM

    results = []
    results.append(("OLD SM (HEAD, ct=0.7)", score(cache, lambda: OldSM(correction_threshold=0.7))))
    results.append(("NEW SM (defaults)", score(cache, lambda: NewSM())))
    for line in (fmt(n, m) for n, m in results):
        print(line)
