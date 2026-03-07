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
    CQT_BINS_PER_OCTAVE,
    CQT_FMIN,
    CQT_N_BINS,
    DEFAULT_THRESHOLD,
    GUITAR_TUNING,
    HOP_LENGTH,
    MIN_NOTE_FRAMES,
    ONSET_THRESHOLD,
    SAMPLE_RATE,
    SUSTAIN_THRESHOLD,
    class_to_string_fret,
)
from model.network import GuitarTranscriptionModel


def load_cqt(audio_path: Path) -> np.ndarray:
    """Load audio and return normalised log-CQT spectrogram (n_bins, T)."""
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    cqt = np.abs(librosa.cqt(
        y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=CQT_FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
    ))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    log_cqt = (log_cqt - log_cqt.min()) / (log_cqt.max() - log_cqt.min() + 1e-8)
    return log_cqt.astype(np.float32)


# Backward-compatible alias so diagnose.py (and any other caller) keeps working
load_mel = load_cqt


def pianoroll_to_notes(
    frame_prob: np.ndarray,
    onset_prob: np.ndarray | None = None,
    art_prob: np.ndarray | None = None,
    onset_threshold: float = ONSET_THRESHOLD,
    sustain_threshold: float = SUSTAIN_THRESHOLD,
    min_duration_frames: int = MIN_NOTE_FRAMES,
    median_filter_size: tuple[int, int] = (1, 1),
    art_threshold: float = 0.5,
) -> list[dict]:
    """Convert (T, P) sigmoid outputs to note events.

    Uses a **Schmitt-trigger** (dual-threshold) strategy:
    * A note begins when ``frame_prob`` exceeds ``onset_threshold``.
    * Once active, the note sustains as long as ``frame_prob`` stays above the
      lower ``sustain_threshold``.
    * If ``onset_prob`` spikes again while a note is already active (onset
      re-articulation), the current note is ended and a new one starts.

    This eliminates the stuttering caused by a single threshold on decaying
    acoustic guitar notes.
    """
    # Median filter suppresses isolated bright pixels (spectral bleed)
    filtered = median_filter(frame_prob, size=median_filter_size)
    T, P = filtered.shape
    frame_sec = HOP_LENGTH / SAMPLE_RATE
    notes: list[dict] = []

    # Onset re-articulation threshold — a spike in onset_prob while a note is
    # already active forces the old note to close and a new one to open.
    onset_reattack_th = onset_threshold * 0.8

    for p in range(P):
        string, fret = class_to_string_fret(p)
        midi_note = GUITAR_TUNING[string] + fret
        in_note = False
        start = 0

        for t in range(T):
            prob = filtered[t, p]

            # --- detect onset re-articulation while note is active -----------
            if in_note and onset_prob is not None:
                is_reattack = (
                    onset_prob[t, p] >= onset_reattack_th
                    and (t == 0 or onset_prob[t, p] > onset_prob[t - 1, p] * 1.2)
                )
                if is_reattack and (t - start) >= min_duration_frames:
                    # close previous note, immediately open new one
                    vel = _estimate_velocity(frame_prob[start:t, p])
                    notes.append({
                        "midi": midi_note,
                        "string": string,
                        "fret": fret,
                        "start": start * frame_sec,
                        "end": t * frame_sec,
                        "velocity": vel,
                    })
                    start = t  # new note starts here
                    continue

            # --- Schmitt trigger: open / sustain / close --------------------
            if not in_note:
                if prob >= onset_threshold:
                    in_note = True
                    start = t
            else:
                # note is active — keep it alive while above sustain floor
                if prob < sustain_threshold:
                    in_note = False
                    if (t - start) >= min_duration_frames:
                        vel = _estimate_velocity(frame_prob[start:t, p])
                        notes.append({
                            "midi": midi_note,
                            "string": string,
                            "fret": fret,
                            "start": start * frame_sec,
                            "end": t * frame_sec,
                            "velocity": vel,
                        })

        # close any still-open note at end of track
        if in_note and (T - start) >= min_duration_frames:
            vel = _estimate_velocity(frame_prob[start:T, p])
            notes.append({
                "midi": midi_note,
                "string": string,
                "fret": fret,
                "start": start * frame_sec,
                "end": T * frame_sec,
                "velocity": vel,
            })

    notes.sort(key=lambda n: n["start"])

    # --- Tag hammer-on articulations ----------------------------------
    if art_prob is not None:
        frame_sec = HOP_LENGTH / SAMPLE_RATE
        for note in notes:
            start_f = int(round(note["start"] / frame_sec))
            end_f = int(round(note["end"] / frame_sec))
            start_f = max(0, min(start_f, art_prob.shape[0] - 1))
            end_f = max(start_f + 1, min(end_f, art_prob.shape[0]))
            # Find the class column for this (string, fret)
            from model.constants import string_fret_to_class
            cls = string_fret_to_class(note["string"], note["fret"])
            mean_art = float(np.mean(art_prob[start_f:end_f, cls]))
            note["articulation"] = "hammer_on" if mean_art >= art_threshold else "pluck"
    else:
        for note in notes:
            note["articulation"] = "pluck"

    return notes


def _estimate_velocity(probs: np.ndarray, lo: int = 40, hi: int = 110) -> int:
    """Map mean frame probability to a MIDI velocity in [lo, hi]."""
    mean_p = float(np.mean(probs))
    return int(np.clip(lo + (hi - lo) * mean_p, lo, hi))


def write_midi(notes: list[dict], output_path: Path, bpm: int = 120):
    """Write note events to a MIDI file.

    Hammer-on notes are marked with MIDI CC 68 (Legato Pedal, value 127)
    immediately before their ``note_on``.  A CC 68 value 0 is emitted at
    the ``note_off`` to close the legato region.  DAWs and notation
    software that understand CC 68 will render these as slurs/legato.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Set instrument to Acoustic Guitar (Steel) so DAWs / MuseScore import
    # it correctly instead of defaulting to piano (program 0).
    track.append(mido.MetaMessage("track_name", name="Acoustic Guitar", time=0))
    track.append(mido.Message("program_change", program=25, channel=0, time=0))

    # Build a flat list of timed MIDI messages
    #   Each entry: (time_sec, priority, message)
    #   priority ensures CC comes just before note_on at the same tick,
    #   and note_off comes before the next note_on.
    CC_LEGATO = 68
    msg_list: list[tuple[float, int, mido.Message]] = []

    for n in notes:
        vel = n.get("velocity", 80)
        is_hammer = n.get("articulation") == "hammer_on"

        if is_hammer:
            # CC 68 = 127 right before note_on
            msg_list.append((
                n["start"], 0,
                mido.Message("control_change", control=CC_LEGATO, value=127, channel=0, time=0),
            ))

        msg_list.append((
            n["start"], 1,
            mido.Message("note_on", note=n["midi"], velocity=vel, channel=0, time=0),
        ))
        msg_list.append((
            n["end"], 0,
            mido.Message("note_off", note=n["midi"], velocity=0, channel=0, time=0),
        ))

        if is_hammer:
            # CC 68 = 0 at note_off to close legato region
            msg_list.append((
                n["end"], 1,
                mido.Message("control_change", control=CC_LEGATO, value=0, channel=0, time=0),
            ))

    msg_list.sort(key=lambda e: (e[0], e[1]))

    current_tick = 0
    for time_sec, _prio, msg in msg_list:
        abs_tick = int(round(mido.second2tick(time_sec, midi_file.ticks_per_beat, tempo)))
        delta = max(0, abs_tick - current_tick)
        msg.time = delta
        track.append(msg)
        current_tick = abs_tick

    midi_file.save(str(output_path))


@torch.no_grad()
def predict(audio_path: Path, checkpoint_path: Path, device: torch.device):
    """Run model on full audio and return frame-level sigmoid piano-roll."""
    mel = load_cqt(audio_path)                          # (n_bins, T)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # (1, n_bins, T)

    model = GuitarTranscriptionModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frame_logits, onset_logits, art_logits = model(mel_t)
    frame_prob = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()  # (T, P)
    onset_prob = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
    art_prob = torch.sigmoid(art_logits).squeeze(0).cpu().numpy()      # (T, P)

    return frame_prob, onset_prob, art_prob


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

    frame_prob, onset_prob, art_prob = predict(args.audio_file, args.checkpoint, device)

    notes = pianoroll_to_notes(
        frame_prob,
        onset_prob,
        art_prob=art_prob,
        onset_threshold=args.threshold,
    )
    write_midi(notes, args.output)

    unique_pitches = sorted(set(n["midi"] for n in notes))
    unique_positions = sorted(set((n["string"], n["fret"]) for n in notes))
    n_hammers = sum(1 for n in notes if n.get("articulation") == "hammer_on")
    print(f"Detected {len(notes)} notes ({n_hammers} hammer-ons) across "
          f"{len(unique_pitches)} unique pitches, "
          f"{len(unique_positions)} unique (string, fret) positions")
    print(f"Saved MIDI → {args.output}")


if __name__ == "__main__":
    main()
