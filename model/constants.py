"""Shared constants for the guitar transcription model."""

from __future__ import annotations

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
NUM_PITCHES = MIDI_MAX - MIDI_MIN + 1  # 49 (kept for GAPS dataset compatibility)

# Tablature / string+fret encoding — for GuitarSet (per-string annotations)
NUM_STRINGS = 6
NUM_FRETS = 21                             # frets 0-20 inclusive
NUM_CLASSES = NUM_STRINGS * NUM_FRETS      # 126

# Standard tuning: MIDI base note for each open string (fret 0)
# String 0 = low E (MIDI 40), ..., String 5 = high e (MIDI 64)
GUITAR_TUNING = (40, 45, 50, 55, 59, 64)

# Training defaults
SEGMENT_DURATION = 5.0     # seconds per training clip
SEGMENT_FRAMES = int(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH)  # ~215 frames
BATCH_SIZE = 16
LEARNING_RATE = 6e-4
NUM_EPOCHS = 50
ONSET_TOLERANCE = 0.05     # 50 ms for evaluation
DEFAULT_THRESHOLD = 0.75   # onset threshold for Schmitt-trigger post-processing

# Schmitt-trigger (dual-threshold) note tracking
ONSET_THRESHOLD = 0.75     # confidence required to START a note
SUSTAIN_THRESHOLD = 0.10   # confidence required to SUSTAIN an already-active note

# Minimum silence gap (frames) — if an onset fires while a note is active on the
# same pitch, the old note is ended and a new one begins. This constant sets the
# minimum number of frames a note must last before it can be re-articulated.
MIN_NOTE_FRAMES = 2

# Paths (relative to project root)
GAPS_DIR = "GAPS"
METADATA_CSV = "GAPS/gaps_metadata_with_splits.csv"
GUITARSET_DIR = "GuitarSet"
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Tablature ↔ MIDI conversion helpers
# ---------------------------------------------------------------------------

def string_fret_to_class(string: int, fret: int) -> int:
    """Convert (string_index, fret_number) to flat class index [0, 125]."""
    return string * NUM_FRETS + fret


def class_to_string_fret(class_idx: int) -> tuple[int, int]:
    """Convert flat class index [0, 125] to (string_index, fret_number)."""
    return class_idx // NUM_FRETS, class_idx % NUM_FRETS


def class_to_midi(class_idx: int) -> int:
    """Convert flat class index [0, 125] to MIDI pitch."""
    string, fret = class_to_string_fret(class_idx)
    return GUITAR_TUNING[string] + fret


def midi_string_to_class(midi_pitch: int, string: int) -> int | None:
    """Convert (midi_pitch, string_index) to class index, or None if fret out of range."""
    fret = midi_pitch - GUITAR_TUNING[string]
    if 0 <= fret < NUM_FRETS:
        return string_fret_to_class(string, fret)
    return None
