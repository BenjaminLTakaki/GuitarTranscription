"""PyTorch Dataset for GuitarSet: pairs mono-mic audio with JAMS annotations."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from scipy.ndimage import zoom
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
    "train":    {"00", "01", "02", "03"},
    "trainval": {"00", "01", "02", "03", "04"},  # train + val; use test set for model selection
    "val":      {"04"},
    "test":     {"05"},
    "all":      None,  # No filtering — use every track (for synthetic data)
}

# CQT bins per semitone (CQT_BINS_PER_OCTAVE=24, 12 semitones/octave → 2 bins/semitone)
_BINS_PER_SEMITONE = 2


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
        self.augment = augment and (split in ("train", "trainval", "all"))

        if split not in _SPLIT_PLAYERS:
            raise ValueError(f"Unknown split '{split}', expected train/trainval/val/test/all")
        allowed_players = _SPLIT_PLAYERS[split]  # None means accept all

        ann_dir   = self.root / "annotation"
        audio_dir = self.root / "audio_mono-mic"

        # Support two layouts:
        #   Subdirectory layout:  root/annotation/*.jams + root/audio_mono-mic/*_mic.wav
        #   Flat layout:          root/*.jams            + root/*_mic.wav  (downloaded by benchmark)
        self.items: List[dict] = []

        if ann_dir.exists():
            jams_glob  = sorted(ann_dir.glob("*.jams"))
            audio_base = audio_dir
        else:
            # Flat layout — all files directly in root/
            jams_glob  = sorted(self.root.glob("*.jams"))
            audio_base = self.root

        for jams_path in jams_glob:
            if allowed_players is not None:
                player_id = jams_path.stem.split("_")[0]  # e.g. "00"
                if player_id not in allowed_players:
                    continue

            # Corresponding audio: same stem + "_mic.wav"
            audio_path = audio_base / (jams_path.stem + "_mic.wav")
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
        if self.split in ("train", "trainval", "all"):
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

            # Apply augmentations
            if self.augment:
                # Time-stretch: random ±15% tempo change
                rate = random.uniform(0.85, 1.15)
                if abs(rate - 1.0) > 0.02:
                    spec, frame_roll, onset_roll, art_roll = self._time_stretch_segment(
                        spec, frame_roll, onset_roll, art_roll, rate
                    )

                # Pitch-shift augmentation: random ±3 semitones
                shift = random.choice([-3, -2, -1, 0, 1, 2, 3])
                if shift != 0:
                    spec = self._shift_spec_pitch(spec, shift)
                    frame_roll = self._shift_roll_pitch(frame_roll, shift)
                    onset_roll = self._shift_roll_pitch(onset_roll, shift)
                    art_roll = self._shift_roll_pitch(art_roll, shift)
                spec = self._augment_spec(spec)

        # Convert to tensors
        spec_t = torch.from_numpy(spec)             # (n_bins, T)
        frame_t = torch.from_numpy(frame_roll)      # (T, NUM_CLASSES)
        onset_t = torch.from_numpy(onset_roll)      # (T, NUM_CLASSES)
        art_t = torch.from_numpy(art_roll)           # (T, NUM_CLASSES)

        return spec_t, frame_t, onset_t, art_t

    # ---- Pitch-shift augmentation (CQT domain) ----

    @staticmethod
    def _shift_spec_pitch(spec: np.ndarray, semitones: int) -> np.ndarray:
        """Shift CQT spectrogram by `semitones` half-steps (±2 max).

        Each semitone = _BINS_PER_SEMITONE (2) CQT bins.  Positive = pitch up.
        Vacated bins are filled with zero.
        """
        n_bins = spec.shape[0]
        b = semitones * _BINS_PER_SEMITONE
        shifted = np.zeros_like(spec)
        if b > 0:
            shifted[b:] = spec[:n_bins - b]
        else:
            shifted[:n_bins + b] = spec[-b:]
        return shifted

    @staticmethod
    def _shift_roll_pitch(roll: np.ndarray, semitones: int) -> np.ndarray:
        """Shift tablature roll fret indices by `semitones` (same-string transposition).

        Notes that would fall outside frets [0, NUM_FRETS-1] are dropped.
        roll shape: (T, NUM_CLASSES)
        """
        s = semitones
        F = NUM_FRETS
        new_roll = np.zeros_like(roll)
        for st in range(NUM_STRINGS):
            base = st * F
            if s > 0:
                # fret j → fret j+s; keep frets [0 .. F-s-1]
                new_roll[:, base + s : base + F] = roll[:, base : base + F - s]
            else:  # s < 0
                # fret j → fret j+s; keep frets [-s .. F-1]
                new_roll[:, base : base + F + s] = roll[:, base - s : base + F]
        return new_roll

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

        # 4. Bass boost augmentation — strengthen low CQT bins
        if random.random() < 0.4:
            boost_db = random.uniform(1.5, 4.0)  # dB boost
            boost_factor = 10 ** (boost_db / 20)
            # Apply to bottom ~36 bins (roughly E2-E4 range in 144-bin CQT)
            n_boost = min(36, n_bins)
            # Gradual rolloff so it's not a hard cutoff
            boost_curve = np.linspace(boost_factor, 1.0, n_boost)
            spec[:n_boost, :] *= boost_curve[:, np.newaxis]
            spec = np.clip(spec, 0.0, 1.0)

        # 5. Room reverb simulation — exponentially decaying echo of past frames
        if random.random() < 0.4:
            decay = random.uniform(0.25, 0.55)
            n_echo = random.randint(6, 24)
            echo = np.zeros_like(spec)
            for lag in range(1, n_echo + 1):
                echo[:, lag:] += (decay ** lag) * spec[:, : T - lag]
            spec = np.clip(spec + echo * 0.25, 0.0, 1.0)

        # 6. Gaussian noise — simulates mic hiss and amp noise
        if random.random() < 0.4:
            noise_std = random.uniform(0.01, 0.04)
            spec = np.clip(spec + np.random.randn(*spec.shape).astype(np.float32) * noise_std, 0.0, 1.0)

        return spec

    # ---- Time-stretch augmentation (spectrogram domain) ----

    @staticmethod
    def _time_stretch_segment(
        spec: np.ndarray,
        frame_roll: np.ndarray,
        onset_roll: np.ndarray,
        art_roll: np.ndarray,
        rate: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stretch or compress a segment along the time axis via interpolation.

        rate > 1 = faster playback (fewer frames), rate < 1 = slower (more frames).
        All outputs are cropped/padded back to the original segment length.
        """
        T = spec.shape[1]
        time_factor = 1.0 / rate

        # Stretch spectrogram: (n_bins, T) — interpolate time axis only
        spec_s = zoom(spec, (1.0, time_factor), order=1)

        # Stretch rolls: (T, num_classes) — bilinear for frame/art, nearest for onset
        frame_s = zoom(frame_roll, (time_factor, 1.0), order=1)
        onset_s = zoom(onset_roll, (time_factor, 1.0), order=0)
        art_s = zoom(art_roll, (time_factor, 1.0), order=1)

        # Crop or pad back to T
        T_new = spec_s.shape[1]
        if T_new >= T:
            spec_out = spec_s[:, :T]
            frame_out = frame_s[:T]
            onset_out = onset_s[:T]
            art_out = art_s[:T]
        else:
            pad = T - T_new
            spec_out = np.pad(spec_s, ((0, 0), (0, pad)))
            frame_out = np.pad(frame_s, ((0, pad), (0, 0)))
            onset_out = np.pad(onset_s, ((0, pad), (0, 0)))
            art_out = np.pad(art_s, ((0, pad), (0, 0)))

        # Binarise after interpolation to keep labels clean
        frame_out = (frame_out > 0.5).astype(np.float32)
        onset_out = (onset_out > 0.5).astype(np.float32)
        art_out = (art_out > 0.5).astype(np.float32)

        return spec_out, frame_out, onset_out, art_out
