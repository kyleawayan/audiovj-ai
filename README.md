# audiovj-ai

## Pipeline

### 1. Import tracks from Rekordbox

```bash
uv run audiovj import-rekordbox <path-to-library.xml> <path-to-audio-folder>
```

Only imports tracks in the "audiovj" playlist. Override with `--playlist <name>`.

### 2. Preprocess audio

```bash
uv run audiovj preprocess
```

### 3. Inspect a track

```bash
uv run audiovj inspect <track_id>
```

### 4. Train

```bash
uv run audiovj train [--epochs 50] [--batch-size 8] [--lr 1e-3]
```

### 5. Evaluate

```bash
uv run audiovj evaluate
```

### 6. Predict on a track

```bash
uv run audiovj predict-file <track_id>
```
