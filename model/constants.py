"""Shared constants for the guitar transcription model."""

# Audio parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 512          # ~23 ms per frame at 22050 Hz
N_FFT = 2048
N_MELS = 229              # standard for AMT (Onsets and Frames uses 229)

# MIDI / pitch range — standard guitar: E2 (40) to E6 (88)
MIDI_MIN = 40
MIDI_MAX = 88
NUM_PITCHES = MIDI_MAX - MIDI_MIN + 1  # 49

# Training defaults
SEGMENT_DURATION = 5.0     # seconds per training clip
SEGMENT_FRAMES = int(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH)  # ~215 frames
BATCH_SIZE = 16
LEARNING_RATE = 6e-4
NUM_EPOCHS = 50
ONSET_TOLERANCE = 0.05     # 50 ms for evaluation
DEFAULT_THRESHOLD = 0.35   # tuned via diagnose.py sweep

# Paths (relative to project root)
GAPS_DIR = "GAPS"
METADATA_CSV = "GAPS/gaps_metadata_with_splits.csv"
CHECKPOINT_DIR = "checkpoints"
