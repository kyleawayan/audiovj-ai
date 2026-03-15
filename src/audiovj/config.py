from pathlib import Path

# Hot cue letter → phrase type mapping (Rekordbox hot cue slots A-F)
HOTCUE_TO_PHRASE: dict[int, str] = {
    0: "other",      # A (intro)
    1: "other",      # B (verse)
    2: "other",      # C (buildup)
    3: "drop",       # D
    4: "other",      # E (breakdown)
    5: "other",      # F (outro)
}

PHRASE_TYPES = ["other", "drop"]

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

# Model hyperparameters
FIXED_FRAMES = 128  # AdaptiveAvgPool1d target (normalizes variable BPM window widths)
ENCODER_CHANNELS = [64, 128]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
NUM_PHRASES = len(PHRASE_TYPES)


def hotcue_to_phrase(num: int) -> str | None:
    """Map a hot cue number (0=A, 1=B, ...) to a phrase type. Returns None if unmapped."""
    return HOTCUE_TO_PHRASE.get(num)
