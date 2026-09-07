"""Live-replay a track through the seq model (causal, stateful, no lookahead)
and render a waveform PNG with predicted phrase markers.

Mirrors run-live exactly: SeqInferenceEngine.step_window per downbeat carrying
the LSTM state, + OnsetCueTracker for drop_start/drop_end/buildup/transition.
The model only ever sees up to the current downbeat -- no future context.

Usage: uv run --with matplotlib python experiments/_plotlive.py <track_id> [<track_id> ...]
"""

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audiovj.config import PHRASE_TYPES  # noqa: E402
from audiovj.live.cue import OnsetCueTracker  # noqa: E402
from audiovj.live.inference import SeqInferenceEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "data/models/seq_unified.safetensors"
OUT_DIR = Path("/Users/kyle/SeasonedTech Dropbox/Kyle Awayan/Documents/audiovj artifacts/live_plots")
DROP_IDX = PHRASE_TYPES.index("drop")
BUILD_IDX = PHRASE_TYPES.index("buildup")


def load_envelope(audio_path: str, sr: int = 8000):
    """ffmpeg-decode to mono wav, return (times, amplitude-envelope 0..1)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", str(sr), tf.name],
            check=True, capture_output=True,
        )
        with wave.open(tf.name, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = sr // 20  # 50 ms envelope
    trimmed = x[: len(x) // hop * hop].reshape(-1, hop)
    env = np.abs(trimmed).max(axis=1)
    env = env / (env.max() + 1e-9)
    t = np.arange(len(env)) * hop / sr
    return t, env


def replay(track_id: str):
    track = json.load(open(ROOT / f"data/tracks/{track_id}.json"))
    downbeats = track["downbeats"]
    data = load_file(str(ROOT / f"data/features/{track_id}.safetensors"))
    windows, kept = data["windows"], data["kept_indices"].tolist()

    engine = SeqInferenceEngine(CKPT, torch.device("cpu"))
    engine.reset()
    cue = OnsetCueTracker(onset_threshold=0.30)

    times, in_drop, drop_prob, build_prob = [], [], [], []
    events = {"drop_start": [], "drop_end": [], "buildup": [], "transition_drop": []}
    for k, widx in enumerate(kept):
        t = downbeats[widx]
        pred = engine.step_window(windows[k])
        for ev in cue.update(pred):
            if ev.kind == "drop_start":
                events["drop_start"].append(t)
            elif ev.kind == "drop_end":
                events["drop_end"].append(t)
            elif ev.kind == "buildup":
                events["buildup"].append(t)
            elif ev.kind == "transition" and ev.phrase == "drop":
                events["transition_drop"].append(t)
        times.append(t)
        in_drop.append(cue.in_drop)
        drop_prob.append(pred.current_probs[DROP_IDX])
        build_prob.append(pred.current_probs[BUILD_IDX])

    # ground-truth allin1 spans (chorus ~= drop) for eyeball comparison
    cues = sorted(track["cue_points"], key=lambda c: c["start_time"])
    end_t = downbeats[-1]
    gt = []
    for i, c in enumerate(cues):
        s = c["start_time"]
        e = cues[i + 1]["start_time"] if i + 1 < len(cues) else end_t
        gt.append((c["phrase_type"], s, e))
    return dict(track=track, times=np.array(times), in_drop=np.array(in_drop),
                drop_prob=np.array(drop_prob), build_prob=np.array(build_prob),
                events=events, gt=gt)


def plot(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t_env, env = load_envelope(r["track"]["audio_path"])
    name = r["track"]["name"]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(f"{name}   —   seq model, live/causal (no lookahead)",
                 fontsize=13, fontweight="bold")

    # ground-truth chorus spans (green) behind the waveform
    for ptype, s, e in r["gt"]:
        if ptype == "chorus":
            ax0.axvspan(s, e, color="#2b8a3e", alpha=0.12, zorder=0)
    # predicted in-drop state (red band)
    td, idrop = r["times"], r["in_drop"]
    for i in range(len(td)):
        if idrop[i]:
            s = td[i]
            e = td[i + 1] if i + 1 < len(td) else t_env[-1]
            ax0.axvspan(s, e, color="#c92a2a", alpha=0.16, zorder=1)

    ax0.fill_between(t_env, env, color="#495057", lw=0, zorder=2)
    ax0.set_ylim(0, 1); ax0.set_ylabel("waveform")
    ax0.set_yticks([])

    def vlines(ax, xs, color, ls, lw=1.8):
        for x in xs:
            ax.axvline(x, color=color, ls=ls, lw=lw, zorder=5)

    vlines(ax0, r["events"]["drop_start"], "#c92a2a", "-", 2.2)
    vlines(ax0, r["events"]["drop_end"], "#1c7ed6", "--", 1.8)
    vlines(ax0, r["events"]["buildup"], "#e67700", ":", 1.8)

    legend = [
        Patch(color="#c92a2a", alpha=0.16, label="predicted IN-DROP state"),
        Patch(color="#2b8a3e", alpha=0.12, label='allin1 "chorus" (≈drop) ground truth'),
        plt.Line2D([], [], color="#c92a2a", lw=2.2, label="drop_start (OSC)"),
        plt.Line2D([], [], color="#1c7ed6", lw=1.8, ls="--", label="drop_end (OSC)"),
        plt.Line2D([], [], color="#e67700", lw=1.8, ls=":", label="buildup (OSC)"),
    ]
    ax0.legend(handles=legend, loc="upper right", fontsize=8, ncol=2, framealpha=0.9)

    # probability panel
    ax1.plot(td, r["drop_prob"], color="#c92a2a", lw=1.6, label="P(drop)")
    ax1.plot(td, r["build_prob"], color="#e67700", lw=1.2, alpha=0.8, label="P(buildup)")
    ax1.axhline(0.30, color="#868e96", ls=":", lw=1, label="onset threshold 0.30")
    ax1.set_ylim(0, 1); ax1.set_ylabel("prob"); ax1.set_xlabel("time (s)")
    ax1.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax1.set_xlim(0, t_env[-1])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = sys.argv[1:] or ["5aed254b16ba"]
    for tid in ids:
        r = replay(tid)
        safe = "".join(ch if ch.isalnum() else "_" for ch in r["track"]["name"])[:40]
        out = OUT_DIR / f"{tid}_{safe}.png"
        plot(r, out)
        ev = r["events"]
        print(f"{tid} {r['track']['name'][:34]:34s} "
              f"starts={len(ev['drop_start'])} ends={len(ev['drop_end'])} "
              f"buildups={len(ev['buildup'])} -> {out.name}")
