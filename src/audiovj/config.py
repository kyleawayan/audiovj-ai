from pathlib import Path

# Phrase vocabulary — sourced directly from allin1's segment labels.
# allin1's `start` and `end` are sub-second markers, not musical sections; we skip them.
PHRASE_TYPES: list[str] = [
    "intro",
    "verse",
    "inst",
    "solo",
    "chorus",
    "break",
    "bridge",
    "outro",
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
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
STRUCT_DIR = DATA_DIR / "struct"

# Model hyperparameters
FIXED_FRAMES = 128
ENCODER_CHANNELS = [64, 128]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
NUM_PHRASES = len(PHRASE_TYPES)
