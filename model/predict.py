#!/usr/bin/env python3
"""Run the trained model on an audio file and write a MIDI."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import mido
import numpy as np
import torch
from scipy.ndimage import median_filter

from model.constants import (
    CHECKPOINT_DIR,
    DEFAULT_THRESHOLD,
    HOP_LENGTH,
    MIDI_MIN,
    N_FFT,
    N_MELS,
    NUM_PITCHES,
    SAMPLE_RATE,
)
from model.network import GuitarTranscriptionModel


def load_mel(audio_path: Path) -> np.ndarray:
    """Load audio and return normalised log-mel spectrogram (n_mels, T)."""
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=30.0, fmax=SAMPLE_RATE // 2,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def pianoroll_to_notes(
    frame_prob: np.ndarray,
    onset_prob: np.ndarray | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_duration_frames: int = 2,
    median_filter_size: tuple[int, int] = (3, 3),
) -> list[dict]:
    """Convert (T, P) sigmoid outputs to note events.

    Improvements over naive thresholding:
    - Median filter on frame probs to suppress spectral bleed
    - Relaxed onset detection (simple above-threshold, no strict local-max)
    - Velocity estimated from mean frame probability per note
    """
    # Median filter suppresses isolated bright pixels (spectral bleed)
    filtered = median_filter(frame_prob, size=median_filter_size)
    binary_frame = (filtered >= threshold).astype(np.int8)
    T, P = binary_frame.shape
    frame_sec = HOP_LENGTH / SAMPLE_RATE
    notes = []

    # Onset threshold is slightly lower to catch more real onsets
    onset_th = max(0.15, threshold * 0.7)

    for p in range(P):
        midi_note = p + MIDI_MIN
        in_note = False
        start = 0
        for t in range(T):
            # Relaxed onset detection: just check if above onset_th
            # and not already at onset level in the previous frame
            is_onset = False
            if onset_prob is not None and onset_prob[t, p] >= onset_th:
                prev_val = onset_prob[t - 1, p] if t > 0 else 0
                if onset_prob[t, p] > prev_val * 1.2:  # 20% rise = new onset
                    is_onset = True

            if binary_frame[t, p]:
                if not in_note:
                    in_note = True
                    start = t
                elif in_note and is_onset:
                    if t - start >= min_duration_frames:
                        vel = _estimate_velocity(frame_prob[start:t, p])
                        notes.append({
                            "midi": midi_note,
                            "start": start * frame_sec,
                            "end": t * frame_sec,
                            "velocity": vel,
                        })
                    start = t
            elif in_note:
                in_note = False
                if t - start >= min_duration_frames:
                    vel = _estimate_velocity(frame_prob[start:t, p])
                    notes.append({
                        "midi": midi_note,
                        "start": start * frame_sec,
                        "end": t * frame_sec,
                        "velocity": vel,
                    })
        if in_note and T - start >= min_duration_frames:
            vel = _estimate_velocity(frame_prob[start:T, p])
            notes.append({
                "midi": midi_note,
                "start": start * frame_sec,
                "end": T * frame_sec,
                "velocity": vel,
            })

    notes.sort(key=lambda n: n["start"])
    return notes


def _estimate_velocity(probs: np.ndarray, lo: int = 40, hi: int = 110) -> int:
    """Map mean frame probability to a MIDI velocity in [lo, hi]."""
    mean_p = float(np.mean(probs))
    return int(np.clip(lo + (hi - lo) * mean_p, lo, hi))


def write_midi(notes: list[dict], output_path: Path, bpm: int = 120):
    """Write note events to a MIDI file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Flatten note_on / note_off and sort by time
    events = []
    for n in notes:
        vel = n.get("velocity", 80)
        events.append(("on", n["start"], n["midi"], vel))
        events.append(("off", n["end"], n["midi"], vel))
    events.sort(key=lambda e: e[1])

    current_tick = 0
    for kind, time_sec, midi_note, velocity in events:
        abs_tick = int(round(mido.second2tick(time_sec, midi_file.ticks_per_beat, tempo)))
        delta = max(0, abs_tick - current_tick)
        if kind == "on":
            track.append(mido.Message("note_on", note=midi_note, velocity=velocity, time=delta))
        else:
            track.append(mido.Message("note_off", note=midi_note, velocity=0, time=delta))
        current_tick = abs_tick

    midi_file.save(str(output_path))


@torch.no_grad()
def predict(audio_path: Path, checkpoint_path: Path, device: torch.device):
    """Run model on full audio and return frame-level sigmoid piano-roll."""
    mel = load_mel(audio_path)                          # (n_mels, T)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # (1, n_mels, T)

    model = GuitarTranscriptionModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frame_logits, onset_logits = model(mel_t)
    frame_prob = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()  # (T, P)
    onset_prob = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()

    return frame_prob, onset_prob


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe guitar audio to MIDI using the trained model"
    )
    parser.add_argument("audio_file", type=Path, help="Input audio (wav/mp3/flac)")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output/transcription_ml.mid"),
        help="Output MIDI path",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(CHECKPOINT_DIR) / "best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio_file}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train first with: python -m model.train"
        )

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    frame_prob, onset_prob = predict(args.audio_file, args.checkpoint, device)

    notes = pianoroll_to_notes(frame_prob, onset_prob, threshold=args.threshold)
    write_midi(notes, args.output)

    unique_pitches = sorted(set(n["midi"] for n in notes))
    print(f"Detected {len(notes)} notes across {len(unique_pitches)} unique pitches")
    print(f"Saved MIDI → {args.output}")


if __name__ == "__main__":
    main()
