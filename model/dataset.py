"""PyTorch Dataset that pairs GAPS audio with frame-level piano-roll labels."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import mido
import numpy as np
import torch
from torch.utils.data import Dataset

from model.constants import (
    GAPS_DIR,
    HOP_LENGTH,
    METADATA_CSV,
    MIDI_MAX,
    MIDI_MIN,
    N_FFT,
    N_MELS,
    NUM_PITCHES,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    SEGMENT_FRAMES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midi_to_note_events(midi_path: str | Path) -> List[Tuple[float, float, int]]:
    """Parse a MIDI file and return a list of (onset_sec, offset_sec, midi_note)."""
    mid = mido.MidiFile(str(midi_path))
    events: List[Tuple[float, float, int]] = []

    for track in mid.tracks:
        active: dict[int, float] = {}       # note -> onset time
        abs_time = 0.0
        tempo = 500_000  # default 120 bpm

        for msg in track:
            # accumulate absolute time in seconds
            abs_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

            if msg.type == "set_tempo":
                tempo = msg.tempo

            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_time
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                onset = active.pop(msg.note, None)
                if onset is not None:
                    events.append((onset, abs_time, msg.note))

        # close any still-open notes
        for note, onset in active.items():
            events.append((onset, abs_time, note))

    return events


def note_events_to_pianoroll(
    events: List[Tuple[float, float, int]],
    num_frames: int,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert note events to a (num_frames, NUM_PITCHES) binary piano-roll."""
    roll = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
    frame_dur = hop_length / sr

    for onset, offset, note in events:
        if note < MIDI_MIN or note > MIDI_MAX:
            continue
        pitch_idx = note - MIDI_MIN
        start_frame = int(round(onset / frame_dur))
        end_frame = int(round(offset / frame_dur))
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, num_frames))
        roll[start_frame:end_frame, pitch_idx] = 1.0

    return roll


def note_events_to_onsets(
    events: List[Tuple[float, float, int]],
    num_frames: int,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Onset-only piano-roll (single-frame impulse at each note onset)."""
    roll = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
    frame_dur = hop_length / sr

    for onset, _offset, note in events:
        if note < MIDI_MIN or note > MIDI_MAX:
            continue
        pitch_idx = note - MIDI_MIN
        frame = int(round(onset / frame_dur))
        if 0 <= frame < num_frames:
            roll[frame, pitch_idx] = 1.0

    return roll


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GAPSDataset(Dataset):
    """Yields (mel_segment, frame_roll, onset_roll) tensors for training.

    During training, a random segment of ``segment_duration`` seconds is
    extracted from each track.  During evaluation the full track is returned
    (no cropping).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        segment_duration: float = SEGMENT_DURATION,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.segment_duration = segment_duration
        self.augment = augment and (split == "train")

        # Read metadata and filter by split
        self.items: List[dict] = []
        csv_path = self.root / METADATA_CSV
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split", "").strip()
                if split == "train" and row_split == "train":
                    self.items.append(row)
                elif split == "val" and row_split == "":
                    # unlabelled split → use as validation
                    self.items.append(row)
                elif split == "test" and row_split == "test":
                    self.items.append(row)

        # Pre-validate that audio + midi files exist
        valid = []
        for item in self.items:
            audio_p = self.root / GAPS_DIR / item["audio_path"]
            midi_p = self.root / GAPS_DIR / item["midi_path"]
            if audio_p.exists() and midi_p.exists():
                valid.append(item)
        self.items = valid

    # ----- cache-friendly: compute mel + labels per item -----

    def _load_audio_mel(self, audio_path: Path) -> np.ndarray:
        """Load audio and return log-mel spectrogram (n_mels, T)."""
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=30.0,
            fmax=SAMPLE_RATE // 2,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        # Normalise to [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        audio_path = self.root / GAPS_DIR / item["audio_path"]
        midi_path = self.root / GAPS_DIR / item["midi_path"]

        # Mel spectrogram: shape (n_mels, T_full)
        mel = self._load_audio_mel(audio_path)
        total_frames = mel.shape[1]

        # MIDI → piano roll + onsets
        events = midi_to_note_events(midi_path)
        frame_roll = note_events_to_pianoroll(events, total_frames)
        onset_roll = note_events_to_onsets(events, total_frames)

        # Crop or pad to fixed segment length during training
        if self.split == "train":
            seg_frames = SEGMENT_FRAMES
            if total_frames > seg_frames:
                start = random.randint(0, total_frames - seg_frames)
            else:
                start = 0
            mel = mel[:, start : start + seg_frames]
            frame_roll = frame_roll[start : start + seg_frames]
            onset_roll = onset_roll[start : start + seg_frames]

            # Pad if shorter
            if mel.shape[1] < seg_frames:
                pad_w = seg_frames - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad_w)))
                frame_roll = np.pad(frame_roll, ((0, pad_w), (0, 0)))
                onset_roll = np.pad(onset_roll, ((0, pad_w), (0, 0)))

            # Apply mel-domain augmentations
            if self.augment:
                mel = self._augment_mel(mel)

        # Convert to tensors
        mel_t = torch.from_numpy(mel)             # (n_mels, T)
        frame_t = torch.from_numpy(frame_roll)    # (T, NUM_PITCHES)
        onset_t = torch.from_numpy(onset_roll)    # (T, NUM_PITCHES)

        return mel_t, frame_t, onset_t

    # ---- Mel-domain augmentation ----

    def _augment_mel(self, mel: np.ndarray) -> np.ndarray:
        """Apply random augmentations to log-mel spectrogram (n_mels, T).

        1. Gain: scale all values by a random factor (simulates volume change)
        2. Frequency masking: zero out 1-3 random mel bands (SpecAugment-lite)
        3. Time masking: zero out a short random time segment
        """
        mel = mel.copy()
        n_mels, T = mel.shape

        # 1. Gain augmentation (±6 dB → scale factor 0.5–2.0 in power)
        #    Since mel is normalised [0,1], scale and re-clip
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            mel = np.clip(mel * gain, 0.0, 1.0)

        # 2. Frequency masking: mask 1-3 contiguous mel bands
        if random.random() < 0.5:
            num_bands = random.randint(1, min(3, n_mels // 10))
            for _ in range(num_bands):
                width = random.randint(1, max(1, n_mels // 15))
                start = random.randint(0, n_mels - width)
                mel[start : start + width, :] = 0.0

        # 3. Time masking: mask a short time segment
        if random.random() < 0.5 and T > 10:
            width = random.randint(1, max(1, T // 10))
            start = random.randint(0, T - width)
            mel[:, start : start + width] = 0.0

        return mel
