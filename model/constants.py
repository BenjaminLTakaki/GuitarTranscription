"""Shared constants for the guitar transcription model."""

import librosa as _librosa

# Audio parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 512          # ~23 ms per frame at 22050 Hz

# CQT parameters (replaces Mel spectrogram for better polyphonic resolution)
CQT_FMIN = float(_librosa.note_to_hz('E2'))   # lowest guitar note
CQT_N_BINS = 144          # 24 bins/octave × 6 octaves (E2 → E8 covers harmonics)
CQT_BINS_PER_OCTAVE = 24  # 2× semitone resolution → clean chord patterns

# N_MELS kept as alias so the network & existing code use a single "freq-bins" constant
N_MELS = CQT_N_BINS       # frequency-axis size fed to CNN (was 229 for mel)

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
DEFAULT_THRESHOLD = 0.5    # onset threshold for Schmitt-trigger post-processing

# Schmitt-trigger (dual-threshold) note tracking
ONSET_THRESHOLD = 0.5      # confidence required to START a note
SUSTAIN_THRESHOLD = 0.15   # confidence required to SUSTAIN an already-active note

# Minimum silence gap (frames) — if an onset fires while a note is active on the
# same pitch, the old note is ended and a new one begins. This constant sets the
# minimum number of frames a note must last before it can be re-articulated.
MIN_NOTE_FRAMES = 2

# Paths (relative to project root)
GAPS_DIR = "GAPS"
METADATA_CSV = "GAPS/gaps_metadata_with_splits.csv"
GUITARSET_DIR = "GuitarSet"
CHECKPOINT_DIR = "checkpoints"
