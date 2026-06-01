# AudioVJ AI

Real-time DJ phrase detection for lighting/visual control.

## Goal

Create a real-time DJ phrase detection system for lighting and visual control during live performances. No pre-processing of tracks or pre-syncing required — just supply the DJ's live audio signal.

## Status

Work in progress. The current model is a **longer-context sequence model**
(`UnifiedSeqPredictor`): a per-downbeat CNN window encoder feeding a *causal* LSTM
**across** downbeats, with three heads (current phrase, next phrase, beats-until).
It runs statefully in `run-live` (one downbeat at a time, carrying the LSTM
state); because the LSTM is causal, the live outputs match offline evaluation.

### Results (high level)

Measured on a held-out Raveform fold (no leakage). Roughly:

| What it does | How well |
|---|---|
| Knows when we're *in* a drop | ~80% |
| Cues the drop start (within ~2 bars) | ~2 of 3 drops, ~1-bar latency |
| Counts down to a drop | ±~1 bar, always smooth (fires for ~60% of drops) |
| Catches load-bearing transitions (≤2 bars) | ~58% |
| Buildup / intro / outro section changes | weaker |

Strong on drops, rough on the subtler transitions — a drop assistant, not an
autopilot. Full breakdown in [`experiments/FINDINGS.md`](experiments/FINDINGS.md).

`run-live` cues transitions on the **onset** of a load-bearing phrase
(rising-edge of its probability past `--onset-threshold`) and emits drop /
buildup / countdown events over OSC. Tempo and downbeats come from Ableton Link /
[Carabiner](https://github.com/Deep-Symmetry/carabiner); a clean beat grid is the
main driver of live quality. You train the model yourself from the Raveform
dataset (below) — no model weights are bundled. Experiment notes and the live
constraints are in [`experiments/FINDINGS.md`](experiments/FINDINGS.md).

## Approach

Training data comes from [**Raveform**](https://mir-aidj.github.io/raveform/), an EDM-specific dataset of 1,423 tracks with human-annotated structure.

### Phrase vocabulary

Raveform's 10 EDM segment labels (all that appear in `segments.json`):

| Label       | Meaning                                  |
|-------------|------------------------------------------|
| `intro`     | Opening, often beatless or sparse        |
| `altintro`  | Alternate intro variant                  |
| `buildup`   | Rising energy / tension                  |
| `drop`      | Full-energy section (the "chorus" of EDM)|
| `breakdown` | Sudden energy drop / atmospheric section |
| `bridge`    | Transitional section (rare)              |
| `cooldown`  | Post-drop release                        |
| `outro`     | Closing section                          |
| `altoutro`  | Alternate outro variant                  |
| `end`       | Final closing marker                     |

## Pipeline

See `audiovj --help` for all commands and per-command options.

### 1. Get the Raveform dataset

Download from the [Raveform site](https://mir-aidj.github.io/raveform/) and extract so the contents land at `data/raveform/`:

```
data/raveform/
├── structures/segments.json
├── structures/beats/<KEY>.beat.csv
├── beats/
├── alignments/
└── ...
```

### 2. Get audio

The Raveform dataset doesn't ship audio. Acquire the audio yourself and place each file at `data/audio/<TRACK_KEY>.wav`, where `<TRACK_KEY>` matches the keys in `data/raveform/structures/segments.json` (e.g. `0019.5WvMql1Ejzs.wav`).

**WAV only.** The importer rejects other formats — compressed codecs (MP3, M4A, etc.) introduce small (~20-40ms) timing offsets during decoding that throw off beat tracking.

The importer skips tracks whose audio is missing, so it's fine to start with a partial corpus.

### 3. Run the pipeline

Import + preprocess features (shared by both models):

```bash
uv run audiovj raveform-import && uv run audiovj preprocess
```

The **sequence model** (the one `run-live` uses) is trained on a clean fold split
in `experiments/_full.py`, then copied to where `run-live` looks for it:

```bash
uv run python experiments/_full.py train 40     # train on folds 0-5, select on fold 6
uv run python experiments/_full.py eval test     # evaluate on held-out fold 7
cp /mnt/scratch/data/loop/seq_unified_full_v2.safetensors data/models/seq_unified.safetensors
```

The original 8-beat model uses the package commands:

```bash
uv run audiovj train && uv run audiovj evaluate && uv run audiovj evaluate-pipeline
```

### Evaluate the live path offline

`evaluate-seq` is the offline twin of `run-live` — it runs the same components
(stateful seq inference + onset cueing) over labeled tracks. Use `--fold 7` to
evaluate only the held-out fold:

```bash
uv run audiovj evaluate-seq --fold 7
```

### Live inference

`run-live` defaults to the seq model (`data/models/seq_unified.safetensors`) and
emits OSC to `/audiovj/...`:

```bash
uv run audiovj list-devices
uv run audiovj run-live --audio-device <index|name> --audio-channels <ch,ch>
# tune cue sensitivity with --onset-threshold (default 0.30; raise to fire less)
```

**OSC events** (address → args): `transition` (onset section change, incl. the
tight drop-start moment) → `[from, to, conf]`; `drop_start` / `drop_end` /
`buildup` → `[phrase, conf]`; `anticipate` (drop-incoming countdown) →
`[phrase, beats_until, conf]`; `phrase` (per-downbeat current) → `[phrase, conf]`;
`beat` → `[bpm]`.
