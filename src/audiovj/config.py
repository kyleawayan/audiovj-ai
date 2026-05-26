from pathlib import Path

# Raveform EDM structure vocabulary (all 10 labels present in segments.json)
PHRASE_TYPES: list[str] = [
    "intro",
    "altintro",
    "buildup",
    "drop",
    "breakdown",
    "bridge",
    "cooldown",
    "outro",
    "altoutro",
    "end",
]

# Audio feature extraction parameters
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CONTEXT_BEATS = 8

# Data paths (relative to project root)
DATA_DIR = Path("data")
TRACKS_DIR = DATA_DIR / "tracks"
TRACKS_VALIDATION_DIR = DATA_DIR / "tracks_validation"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

# Model hyperparameters
FIXED_FRAMES = 128  # AdaptiveAvgPool1d target (normalizes variable BPM window widths)
ENCODER_CHANNELS = [64, 128]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
NUM_PHRASES = len(PHRASE_TYPES)
