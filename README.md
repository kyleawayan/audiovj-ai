# AudioVJ AI

Real-time DJ phrase detection for lighting/visual control.

## Goal

Create a real-time DJ phrase detection system for lighting and visual control during live performances. No pre-processing of tracks or pre-syncing required — just supply the DJ's live audio signal.

## 🚧 Work In Progress

- Model architecture is still rough.
- Performance isn't great yet.
- For tempo and phase detection, Ableton Link / [Carabiner](https://github.com/Deep-Symmetry/carabiner) is used in the meantime, until phrase detection is reliable enough. Then a tempo/phase model can come next.

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

```bash
uv run audiovj raveform-import && \
uv run audiovj preprocess && \
uv run audiovj train && \
uv run audiovj evaluate && \
uv run audiovj evaluate-pipeline
```

### Live inference

```bash
uv run audiovj list-devices
uv run audiovj run-live --audio-device <index|name> --audio-channels <ch,ch>
```
