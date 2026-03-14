from pathlib import Path

# Color-to-phrase mapping (exact RGB match — Rekordbox uses fixed colors)
COLOR_TO_PHRASE: dict[tuple[int, int, int], str] = {
    (222, 68, 207): "intro",
    (48, 90, 255): "buildup",
    (16, 177, 118): "drop",
    (195, 175, 4): "breakdown",
    (255, 18, 123): "outro",
}

PHRASE_TYPES = list(dict.fromkeys(COLOR_TO_PHRASE.values()))

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


def color_to_phrase(r: int, g: int, b: int) -> str | None:
    """Map an exact RGB color to a phrase type. Returns None if unrecognized."""
    return COLOR_TO_PHRASE.get((r, g, b))
