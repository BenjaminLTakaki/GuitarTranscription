"""PyTorch Dataset for GuitarSet: pairs mono-mic audio with JAMS annotations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from model.constants import (
    CQT_BINS_PER_OCTAVE,
    CQT_FMIN,
    CQT_N_BINS,
    GUITAR_TUNING,
    HOP_LENGTH,
    NUM_CLASSES,
    NUM_FRETS,
    NUM_STRINGS,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    SEGMENT_FRAMES,
    midi_string_to_class,
)


# ---------------------------------------------------------------------------
# Helpers — JAMS parsing (plain JSON, no ``jams`` library needed)
# ---------------------------------------------------------------------------

def jams_to_tab_events(jams_path: str | Path) -> List[Tuple[float, float, int]]:
    """Extract (onset_sec, offset_sec, class_idx) from a GuitarSet JAMS file.

    GuitarSet stores per-string ``note_midi`` annotations.  The *i*-th
    ``note_midi`` annotation corresponds to string *i* (0 = low E, 5 = high e).
    We convert each note to a tablature class index:
        class_idx = string_index * NUM_FRETS + fret_number
    where fret_number = midi_pitch - GUITAR_TUNING[string_index].

    Notes that fall outside frets 0-20 are silently skipped.
    """
    with open(jams_path, "r", encoding="utf-8") as f:
        jams = json.load(f)

    events: List[Tuple[float, float, int]] = []
    string_idx = 0

    for ann in jams.get("annotations", []):
        ns = ann.get("namespace", "")
        if ns != "note_midi":
            continue
        if string_idx >= NUM_STRINGS:
            break

        for obs in ann.get("data", []):
            onset = obs["time"]
            duration = obs["duration"]
            midi_pitch = int(round(obs["value"]))
            offset = onset + duration

            class_idx = midi_string_to_class(midi_pitch, string_idx)
            if class_idx is None:
                continue
            events.append((onset, offset, class_idx))

        string_idx += 1

    return events


def tab_events_to_roll(
    events: List[Tuple[float, float, int]],
    num_frames: int,
    num_classes: int = NUM_CLASSES,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert tab events to a (num_frames, NUM_CLASSES) binary frame roll."""
    roll = np.zeros((num_frames, num_classes), dtype=np.float32)
    frame_dur = hop_length / sr

    for onset, offset, class_idx in events:
        if class_idx < 0 or class_idx >= num_classes:
            continue
        start_frame = int(round(onset / frame_dur))
        end_frame = int(round(offset / frame_dur))
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, num_frames))
        roll[start_frame:end_frame, class_idx] = 1.0

    return roll


def tab_events_to_onsets(
    events: List[Tuple[float, float, int]],
    num_frames: int,
    num_classes: int = NUM_CLASSES,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Onset-only roll (single-frame impulse at each event onset)."""
    roll = np.zeros((num_frames, num_classes), dtype=np.float32)
    frame_dur = hop_length / sr

    for onset, _offset, class_idx in events:
        if class_idx < 0 or class_idx >= num_classes:
            continue
        frame = int(round(onset / frame_dur))
        if 0 <= frame < num_frames:
            roll[frame, class_idx] = 1.0

    return roll


# ---------------------------------------------------------------------------
# Articulation parsing
# ---------------------------------------------------------------------------

def jams_to_articulation_events(
    jams_path: str | Path,
) -> List[Tuple[float, float, int]]:
    """Extract hammer-on events from the *articulation* JAMS namespace.

    Returns a list of ``(onset_sec, offset_sec, class_idx)`` for every
    hammer-on note.  Class indices use the same encoding as the main
    tablature target (``string * NUM_FRETS + fret``).  If the JAMS file
    has no articulation namespace (e.g. real GuitarSet), an empty list is
    returned.
    """
    with open(jams_path, "r", encoding="utf-8") as f:
        jams = json.load(f)

    events: List[Tuple[float, float, int]] = []

    for ann in jams.get("annotations", []):
        if ann.get("namespace") != "articulation":
            continue
        for obs in ann.get("data", []):
            if obs.get("value") != "hammer_on":
                continue
            # string / midi fields are stored by generate_synthetic.py
            string = obs.get("string")
            midi = obs.get("midi")
            if string is None or midi is None:
                continue
            class_idx = midi_string_to_class(int(round(midi)), int(string))
            if class_idx is None:
                continue
            onset = obs["time"]
            offset = onset + obs["duration"]
            events.append((onset, offset, class_idx))

    return events


def articulation_events_to_roll(
    events: List[Tuple[float, float, int]],
    num_frames: int,
    num_classes: int = NUM_CLASSES,
    hop_length: int = HOP_LENGTH,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Build a (num_frames, NUM_CLASSES) binary roll for hammer-on events.

    The roll is 1 wherever a hammer-on note is active (same layout as the
    main frame roll).  This serves as the supervision target for the
    articulation head.
    """
    roll = np.zeros((num_frames, num_classes), dtype=np.float32)
    frame_dur = hop_length / sr

    for onset, offset, class_idx in events:
        if class_idx < 0 or class_idx >= num_classes:
            continue
        start_frame = int(round(onset / frame_dur))
        end_frame = int(round(offset / frame_dur))
        start_frame = max(0, min(start_frame, num_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, num_frames))
        roll[start_frame:end_frame, class_idx] = 1.0

    return roll


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Standard player-based split used in GuitarSet literature:
#   Train on players 00-03 (4 players, ~240 clips)
#   Validation on player 04 (~60 clips)
#   Test on player 05 (~60 clips)
_SPLIT_PLAYERS = {
    "train": {"00", "01", "02", "03"},
    "val":   {"04"},
    "test":  {"05"},
    "all":   None,  # No filtering — use every track (for synthetic data)
}


class GuitarSetDataset(Dataset):
    """Yields (cqt_segment, frame_roll, onset_roll) tensors for training.

    During training a random segment of ``segment_duration`` seconds is
    extracted from each track.  During evaluation the full track is returned.

    Expected directory layout (created by ``download_guitarset.sh``)::

        GuitarSet/
            annotation/          ← JAMS files
               00_BN1-129-Eb_comp.jams
               …
            audio_mono-mic/      ← mono mic WAV files
               00_BN1-129-Eb_comp_mic.wav
               …
    """

    def __init__(
        self,
        root: str | Path = "GuitarSet",
        split: str = "train",
        segment_duration: float = SEGMENT_DURATION,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.segment_duration = segment_duration
        self.augment = augment and (split == "train")

        ann_dir = self.root / "annotation"
        audio_dir = self.root / "audio_mono-mic"

        if split not in _SPLIT_PLAYERS:
            raise ValueError(f"Unknown split '{split}', expected train/val/test/all")
        allowed_players = _SPLIT_PLAYERS[split]  # None means accept all

        # Discover pairs of (audio, jams) by scanning the annotation dir
        self.items: List[dict] = []
        if not ann_dir.exists():
            print(f"WARNING: annotation dir not found: {ann_dir}")
            return

        for jams_path in sorted(ann_dir.glob("*.jams")):
            if allowed_players is not None:
                player_id = jams_path.stem.split("_")[0]  # e.g. "00"
                if player_id not in allowed_players:
                    continue

            # Corresponding audio: same stem + "_mic.wav"
            audio_path = audio_dir / (jams_path.stem + "_mic.wav")
            if not audio_path.exists():
                continue

            self.items.append({
                "audio_path": str(audio_path),
                "jams_path": str(jams_path),
            })

    # ----- CQT computation -----

    @staticmethod
    def _load_audio_cqt(audio_path: str | Path) -> np.ndarray:
        """Load audio and return log-CQT spectrogram (n_bins, T)."""
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        cqt = np.abs(librosa.cqt(
            y,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            fmin=CQT_FMIN,
            n_bins=CQT_N_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
        ))
        log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)  # (n_bins, T)
        log_cqt = (log_cqt - log_cqt.min()) / (log_cqt.max() - log_cqt.min() + 1e-8)
        return log_cqt.astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # CQT spectrogram → (n_bins, T_full)
        spec = self._load_audio_cqt(item["audio_path"])
        total_frames = spec.shape[1]

        # JAMS → tablature events → frame roll + onset roll
        events = jams_to_tab_events(item["jams_path"])
        frame_roll = tab_events_to_roll(events, total_frames)
        onset_roll = tab_events_to_onsets(events, total_frames)

        # Articulation roll (hammer-on labels) — zeros for datasets without
        # an articulation namespace (e.g. real GuitarSet recordings).
        art_events = jams_to_articulation_events(item["jams_path"])
        art_roll = articulation_events_to_roll(art_events, total_frames)

        # Crop or pad to fixed segment length during training
        if self.split in ("train", "all"):
            seg_frames = SEGMENT_FRAMES
            if total_frames > seg_frames:
                start = random.randint(0, total_frames - seg_frames)
            else:
                start = 0
            spec = spec[:, start : start + seg_frames]
            frame_roll = frame_roll[start : start + seg_frames]
            onset_roll = onset_roll[start : start + seg_frames]
            art_roll = art_roll[start : start + seg_frames]

            # Pad if shorter
            if spec.shape[1] < seg_frames:
                pad_w = seg_frames - spec.shape[1]
                spec = np.pad(spec, ((0, 0), (0, pad_w)))
                frame_roll = np.pad(frame_roll, ((0, pad_w), (0, 0)))
                onset_roll = np.pad(onset_roll, ((0, pad_w), (0, 0)))
                art_roll = np.pad(art_roll, ((0, pad_w), (0, 0)))

            # Apply spectrogram augmentations
            if self.augment:
                spec = self._augment_spec(spec)

        # Convert to tensors
        spec_t = torch.from_numpy(spec)             # (n_bins, T)
        frame_t = torch.from_numpy(frame_roll)      # (T, NUM_CLASSES)
        onset_t = torch.from_numpy(onset_roll)      # (T, NUM_CLASSES)
        art_t = torch.from_numpy(art_roll)           # (T, NUM_CLASSES)

        return spec_t, frame_t, onset_t, art_t

    # ---- Spectrogram augmentation ----

    @staticmethod
    def _augment_spec(spec: np.ndarray) -> np.ndarray:
        """Apply random augmentations to log-CQT spectrogram (n_bins, T).

        1. Gain: scale all values by a random factor (simulates volume change)
        2. Frequency masking: zero out 1-3 random CQT bins (SpecAugment-lite)
        3. Time masking: zero out a short random time segment
        """
        spec = spec.copy()
        n_bins, T = spec.shape

        # 1. Gain augmentation
        if random.random() < 0.5:
            gain = random.uniform(0.7, 1.3)
            spec = np.clip(spec * gain, 0.0, 1.0)

        # 2. Frequency masking
        if random.random() < 0.5:
            num_bands = random.randint(1, min(3, n_bins // 10))
            for _ in range(num_bands):
                width = random.randint(1, max(1, n_bins // 15))
                start = random.randint(0, n_bins - width)
                spec[start : start + width, :] = 0.0

        # 3. Time masking
        if random.random() < 0.5 and T > 10:
            width = random.randint(1, max(1, T // 10))
            start = random.randint(0, T - width)
            spec[:, start : start + width] = 0.0

        return spec
