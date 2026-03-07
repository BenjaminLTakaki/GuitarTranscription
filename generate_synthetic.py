#!/usr/bin/env python3
"""Generate synthetic guitar audio + JAMS annotations for training.

Uses Karplus-Strong physical modelling synthesis to produce plucked-string
audio that is far more varied than a single SoundFont sampler.  Outputs are
written in the same layout as GuitarSet so the existing dataset loader
(``model.guitarset_dataset.GuitarSetDataset``) can consume them directly.

Usage
-----
    # Generate 500 tracks (~42 hours at 5 min each) into SyntheticGuitar/
    python generate_synthetic.py --num-tracks 500 --duration 30

    # Quick smoke test (10 short clips)
    python generate_synthetic.py --num-tracks 10 --duration 10 --out-dir SyntheticTest
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050

# Standard tuning: MIDI note for each open string (low E → high e)
GUITAR_TUNING = (40, 45, 50, 55, 59, 64)
NUM_STRINGS = 6
NUM_FRETS = 21  # frets 0–20

# Musical intervals for common chord voicings (relative to root fret)
_CHORD_SHAPES = {
    "major":    [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    "minor":    [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
    "power":    [(0, 0), (1, 2), (2, 2)],
    "sus2":     [(0, 0), (1, 2), (2, 2), (3, 2), (4, 0), (5, 0)],
    "dom7":     [(0, 0), (1, 2), (2, 0), (3, 1), (4, 0)],
    "barremaj": [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    "barrmin":  [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
}

# Common scale patterns (intervals from root in semitones)
_SCALE_PATTERNS = {
    "major":      [0, 2, 4, 5, 7, 9, 11, 12],
    "minor":      [0, 2, 3, 5, 7, 8, 10, 12],
    "pentatonic": [0, 3, 5, 7, 10, 12],
    "blues":      [0, 3, 5, 6, 7, 10, 12],
    "dorian":     [0, 2, 3, 5, 7, 9, 10, 12],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10, 12],
}


# ---------------------------------------------------------------------------
# Karplus-Strong plucked string synthesis
# ---------------------------------------------------------------------------

def _midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def karplus_strong(
    frequency: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    pluck_position: float | None = None,
    brightness: float | None = None,
    body_resonance: float | None = None,
    decay_factor: float | None = None,
    is_hammer_on: bool = False,
) -> np.ndarray:
    """Synthesise a single plucked string via extended Karplus-Strong.

    Parameters
    ----------
    frequency : Hz of the note.
    duration : seconds.
    sr : sample rate.
    pluck_position : 0–1, affects harmonic content (0.5 = mellow, 0.1 = bright).
    brightness : 0–1, controls the low-pass smoothing amount.
    body_resonance : 0–1, strength of a simple body‐resonance filter.
    decay_factor : 0–1, additional per-sample energy loss.
    is_hammer_on : if True, use a low-energy impulse instead of a full pick attack.
    """
    if pluck_position is None:
        pluck_position = random.uniform(0.05, 0.5)
    if brightness is None:
        brightness = random.uniform(0.3, 0.95)
    if decay_factor is None:
        decay_factor = random.uniform(0.994, 0.9999)

    n_samples = int(sr * duration)
    delay_len = max(2, int(round(sr / frequency)))

    # Initial excitation
    if is_hammer_on:
        # No pick attack — low-energy impulse simulates finger hitting fretboard
        buf = np.random.uniform(-0.1, 0.1, delay_len).astype(np.float64)
    else:
        # Standard pick burst
        buf = np.random.uniform(-1, 1, delay_len).astype(np.float64)

    # Pluck position comb filter — notches harmonics at 1/pluck_position
    comb_delay = max(1, int(round(delay_len * pluck_position)))
    comb_buf = buf.copy()
    for i in range(comb_delay, delay_len):
        comb_buf[i] = buf[i] - buf[i - comb_delay]
    buf = comb_buf

    # Output buffer
    out = np.zeros(n_samples, dtype=np.float64)
    out[:delay_len] = buf

    # Smoothing coefficient (higher = brighter)
    a = 0.5 + 0.5 * brightness  # range [0.5, 1.0]

    idx = 0
    for i in range(delay_len, n_samples):
        # Low-pass averaging (basic KS)
        prev = out[idx]
        nxt = out[(idx + 1) % delay_len]
        new_val = decay_factor * (a * prev + (1 - a) * nxt)
        out[i] = new_val
        out[idx] = new_val  # write back into the ring-buffer portion
        idx = (idx + 1) % delay_len

    # Simple amplitude envelope: short attack, natural decay
    attack_samples = min(int(0.003 * sr), n_samples)
    if attack_samples > 0:
        out[:attack_samples] *= np.linspace(0, 1, attack_samples)

    # Optional body resonance — simple 2nd-order peak around 100–250 Hz
    if body_resonance is not None and body_resonance > 0:
        from scipy.signal import lfilter
        body_freq = random.uniform(80, 250)
        w0 = 2 * np.pi * body_freq / sr
        Q = random.uniform(2, 8)
        alpha = np.sin(w0) / (2 * Q)
        b0 = 1 + body_resonance * alpha
        b1 = -2 * np.cos(w0)
        b2 = 1 - body_resonance * alpha
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        out = lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], out)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Audio augmentation  (applied to the full mix, simulates recording conditions)
# ---------------------------------------------------------------------------

def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply random audio augmentations to simulate real recording conditions."""
    from scipy.signal import lfilter

    # 1. Random EQ — gentle low/high shelf
    if random.random() < 0.7:
        # Simple 1-pole low-pass or high-pass tilt
        alpha = random.uniform(0.85, 0.99)
        if random.random() < 0.5:
            # Low-pass tilt (warmer tone)
            audio = lfilter([1 - alpha], [1, -alpha], audio).astype(np.float32)
        else:
            # High-pass tilt (brighter / thinner)
            audio = lfilter([1, -1], [1, -alpha], audio).astype(np.float32)

    # 2. Add subtle noise floor
    if random.random() < 0.6:
        noise_level = random.uniform(0.001, 0.01)
        audio = audio + noise_level * np.random.randn(len(audio)).astype(np.float32)

    # 3. Random gain
    gain = random.uniform(0.5, 1.5)
    audio = audio * gain

    # 4. Soft clipping (simulates mild amp drive)
    if random.random() < 0.3:
        drive = random.uniform(1.0, 3.0)
        audio = np.tanh(drive * audio).astype(np.float32)

    # 5. Simple reverb (FIR comb filter approximation)
    if random.random() < 0.5:
        delay_ms = random.uniform(15, 60)
        delay_samples = int(delay_ms * sr / 1000)
        decay = random.uniform(0.1, 0.35)
        reverbed = np.zeros(len(audio) + delay_samples, dtype=np.float32)
        reverbed[: len(audio)] += audio
        reverbed[delay_samples : delay_samples + len(audio)] += decay * audio
        audio = reverbed[: len(audio)]

    # Normalize peak to avoid clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * random.uniform(0.6, 0.95)

    return audio


# ---------------------------------------------------------------------------
# Musical pattern generators
# ---------------------------------------------------------------------------

def _string_for_midi(midi: int) -> Tuple[int, int] | None:
    """Find a valid (string, fret) for a MIDI note, preferring lower positions."""
    candidates = []
    for s, open_midi in enumerate(GUITAR_TUNING):
        fret = midi - open_midi
        if 0 <= fret < NUM_FRETS:
            candidates.append((s, fret))
    if not candidates:
        return None
    # Prefer lower fret positions
    candidates.sort(key=lambda x: x[1])
    return candidates[0]


def generate_single_notes(
    duration: float,
    bpm: float,
) -> List[dict]:
    """Generate a sequence of single melodic notes (scale runs, intervals)."""
    events = []
    beat_dur = 60.0 / bpm
    t = random.uniform(0.0, 0.5)  # small initial offset

    # Pick a scale
    scale_name = random.choice(list(_SCALE_PATTERNS.keys()))
    intervals = _SCALE_PATTERNS[scale_name]
    root_midi = random.randint(40, 60)  # E2 to C4

    while t < duration - 0.1:
        # Choose note from scale
        octave_shift = random.choice([0, 0, 0, 12, -12]) if random.random() < 0.2 else 0
        note_midi = root_midi + random.choice(intervals) + octave_shift

        pos = _string_for_midi(note_midi)
        if pos is None:
            t += beat_dur * random.choice([0.5, 1])
            continue

        string, fret = pos

        # Note duration: eighth, quarter, half, or full beat
        note_dur_beats = random.choice([0.25, 0.5, 0.5, 1.0, 1.0, 2.0])
        note_dur = note_dur_beats * beat_dur
        note_dur = min(note_dur, duration - t)
        note_dur = max(note_dur, 0.08)

        velocity = random.uniform(0.4, 1.0)
        events.append({
            "string": string,
            "fret": fret,
            "midi": note_midi,
            "onset": t,
            "duration": note_dur,
            "velocity": velocity,
        })

        # Gap between notes (sometimes legato, sometimes staccato)
        gap = random.uniform(0.0, 0.15) if random.random() < 0.5 else 0.0
        t += note_dur + gap

    return events


def generate_chords(
    duration: float,
    bpm: float,
) -> List[dict]:
    """Generate strummed chord patterns."""
    events = []
    beat_dur = 60.0 / bpm
    t = random.uniform(0.0, 0.3)

    # Pick rhythm pattern
    patterns = [
        [1, 1, 1, 1],             # straight quarters
        [2, 1, 1],                 # half + two quarters
        [1.5, 0.5, 1, 1],         # dotted quarter + eighth + quarters
        [2, 2],                    # half notes
        [0.5, 0.5, 1, 0.5, 0.5, 1],  # syncopated
    ]
    pattern = random.choice(patterns)
    pat_idx = 0

    while t < duration - 0.2:
        # Pick a chord shape
        shape_name = random.choice(list(_CHORD_SHAPES.keys()))
        shape = _CHORD_SHAPES[shape_name]
        root_fret = random.randint(0, 12)

        # Strum direction (down = low→high, up = high→low)
        strum_delay = random.uniform(0.005, 0.04)
        reverse_strum = random.random() < 0.3

        chord_dur_beats = pattern[pat_idx % len(pattern)]
        chord_dur = chord_dur_beats * beat_dur
        chord_dur = min(chord_dur, duration - t)

        ordered_shape = list(shape)
        if reverse_strum:
            ordered_shape = list(reversed(ordered_shape))

        for i, (rel_string, rel_fret) in enumerate(ordered_shape):
            s = rel_string
            f = root_fret + rel_fret
            if f >= NUM_FRETS:
                continue
            midi = GUITAR_TUNING[s] + f
            note_onset = t + i * strum_delay

            if note_onset >= duration:
                break

            velocity = random.uniform(0.5, 1.0)
            note_dur = max(0.1, chord_dur - i * strum_delay)
            note_dur = min(note_dur, duration - note_onset)

            events.append({
                "string": s,
                "fret": f,
                "midi": midi,
                "onset": note_onset,
                "duration": note_dur,
                "velocity": velocity,
            })

        gap = random.uniform(0.0, 0.1)
        t += chord_dur + gap
        pat_idx += 1

    return events


def generate_arpeggios(
    duration: float,
    bpm: float,
) -> List[dict]:
    """Generate arpeggiated patterns (notes of a chord played sequentially)."""
    events = []
    beat_dur = 60.0 / bpm
    t = random.uniform(0.0, 0.3)

    while t < duration - 0.5:
        # Pick a chord shape and root
        shape_name = random.choice(list(_CHORD_SHAPES.keys()))
        shape = _CHORD_SHAPES[shape_name]
        root_fret = random.randint(0, 10)

        # Arpeggio pattern: up, down, up-down, random
        direction = random.choice(["up", "down", "updown", "random"])
        note_indices = list(range(len(shape)))

        if direction == "down":
            note_indices = list(reversed(note_indices))
        elif direction == "updown":
            note_indices = note_indices + list(reversed(note_indices[1:-1]))
        elif direction == "random":
            random.shuffle(note_indices)

        note_dur_beats = random.choice([0.25, 0.5])
        note_dur = note_dur_beats * beat_dur

        for idx in note_indices:
            if t >= duration - 0.05:
                break
            s, rel_fret = shape[idx]
            f = root_fret + rel_fret
            if f >= NUM_FRETS:
                continue
            midi = GUITAR_TUNING[s] + f

            # Arpeggiated notes may ring together (let ring)
            ring_dur = random.uniform(note_dur, note_dur * 4)
            ring_dur = min(ring_dur, duration - t)

            velocity = random.uniform(0.4, 0.9)
            events.append({
                "string": s,
                "fret": f,
                "midi": midi,
                "onset": t,
                "duration": ring_dur,
                "velocity": velocity,
            })
            t += note_dur

        # Gap between arpeggio groups
        gap = random.uniform(0.0, beat_dur)
        t += gap

    return events


def generate_hammer_ons(
    duration: float,
    bpm: float,
) -> List[dict]:
    """Generate melodic phrases that include hammer-on articulations.

    A hammer-on occurs when a note is 1–2 frets *higher* on the same string
    as the previous note, played without re-picking.
    """
    events: List[dict] = []
    beat_dur = 60.0 / bpm
    t = random.uniform(0.0, 0.5)

    scale_name = random.choice(list(_SCALE_PATTERNS.keys()))
    intervals = _SCALE_PATTERNS[scale_name]
    root_midi = random.randint(40, 60)

    prev_string: int | None = None
    prev_fret: int | None = None

    while t < duration - 0.1:
        note_midi = root_midi + random.choice(intervals)
        pos = _string_for_midi(note_midi)
        if pos is None:
            t += beat_dur * random.choice([0.5, 1])
            prev_string = None
            prev_fret = None
            continue

        string, fret = pos

        # Determine if this note qualifies as a hammer-on
        is_hammer = False
        if (
            prev_string is not None
            and string == prev_string
            and prev_fret is not None
            and 1 <= (fret - prev_fret) <= 2
        ):
            is_hammer = True

        # Hammer-on notes are typically fast
        if is_hammer:
            note_dur_beats = random.choice([0.25, 0.25, 0.5])
        else:
            note_dur_beats = random.choice([0.5, 1.0, 1.0])
        note_dur = note_dur_beats * beat_dur
        note_dur = min(note_dur, duration - t)
        note_dur = max(note_dur, 0.08)

        velocity = random.uniform(0.3, 0.7) if is_hammer else random.uniform(0.5, 1.0)

        events.append({
            "string": string,
            "fret": fret,
            "midi": note_midi,
            "onset": t,
            "duration": note_dur,
            "velocity": velocity,
            "articulation": "hammer_on" if is_hammer else "pluck",
        })

        prev_string = string
        prev_fret = fret

        gap = random.uniform(0.0, 0.05) if is_hammer else random.uniform(0.0, 0.15)
        t += note_dur + gap

        # Occasionally reset so not every note chains
        if random.random() < 0.2:
            prev_string = None
            prev_fret = None

    return events


def generate_mixed_pattern(duration: float, bpm: float) -> List[dict]:
    """Generate a mix of single notes, chords, and arpeggios in sections."""
    events = []
    t = 0.0
    generators = [generate_single_notes, generate_chords, generate_arpeggios, generate_hammer_ons]

    while t < duration:
        section_dur = random.uniform(3.0, min(10.0, duration - t))
        if section_dur < 1.0:
            break
        gen = random.choice(generators)
        section_events = gen(section_dur, bpm)
        # Shift onsets by current position
        for e in section_events:
            e["onset"] += t
        events.extend(section_events)
        t += section_dur + random.uniform(0.1, 0.5)

    return events


# ---------------------------------------------------------------------------
# Render events to audio
# ---------------------------------------------------------------------------

def render_events(
    events: List[dict],
    total_duration: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render note events to a mono audio buffer using Karplus-Strong."""
    n_samples = int(total_duration * sr) + sr  # extra second for ring-out
    audio = np.zeros(n_samples, dtype=np.float32)

    # Randomise per-track timbral characteristics
    base_brightness = random.uniform(0.3, 0.9)
    base_body = random.uniform(0.0, 0.6)
    base_decay = random.uniform(0.995, 0.9999)

    for ev in events:
        freq = _midi_to_hz(ev["midi"])
        # Per-note variation
        brightness = np.clip(base_brightness + random.uniform(-0.1, 0.1), 0.1, 0.99)
        body = np.clip(base_body + random.uniform(-0.1, 0.1), 0.0, 0.8)
        decay = np.clip(base_decay + random.uniform(-0.001, 0.001), 0.99, 0.99999)
        pluck_pos = random.uniform(0.05, 0.45)

        dur = ev["duration"] + random.uniform(0.1, 0.5)  # ring-out tail

        is_hammer = ev.get("articulation") == "hammer_on"

        note_audio = karplus_strong(
            freq, dur, sr,
            pluck_position=pluck_pos,
            brightness=brightness,
            body_resonance=body,
            decay_factor=decay,
            is_hammer_on=is_hammer,
        )

        # Apply velocity
        note_audio *= ev["velocity"]

        # Mix into output
        start_sample = int(ev["onset"] * sr)
        end_sample = start_sample + len(note_audio)
        if start_sample >= n_samples:
            continue
        end_sample = min(end_sample, n_samples)
        audio[start_sample:end_sample] += note_audio[: end_sample - start_sample]

    # Trim to target duration
    audio = audio[: int(total_duration * sr)]
    return audio


# ---------------------------------------------------------------------------
# JAMS export (compatible with GuitarSet format)
# ---------------------------------------------------------------------------

def events_to_jams(
    events: List[dict],
    duration: float,
    title: str = "synthetic",
) -> dict:
    """Build a minimal JAMS dict with per-string note_midi annotations.

    This produces the exact structure that ``jams_to_tab_events()`` in
    ``model/guitarset_dataset.py`` expects:
    - Alternating (pitch_contour, note_midi) annotation pairs per string
    - note_midi observations have {time, duration, value, confidence}
    """
    annotations = []

    for string_idx in range(NUM_STRINGS):
        # Empty pitch_contour (required to keep annotation indices aligned)
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "synthetic", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "SyntheticGuitar",
                "annotation_tools": "generate_synthetic.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "pitch_contour",
            "data": {"time": [], "duration": [], "value": [], "confidence": []},
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })

        # note_midi observations for this string
        string_notes = [e for e in events if e["string"] == string_idx]
        string_notes.sort(key=lambda e: e["onset"])

        obs_list = []
        for e in string_notes:
            obs_list.append({
                "time": e["onset"],
                "duration": e["duration"],
                "value": float(e["midi"]),
                "confidence": None,
            })

        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "synthetic", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "SyntheticGuitar",
                "annotation_tools": "generate_synthetic.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "note_midi",
            "data": obs_list,
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })

    # ---- articulation annotation (one for entire track) -----
    art_obs = []
    for e in events:
        art_label = e.get("articulation", "pluck")
        art_obs.append({
            "time": e["onset"],
            "duration": e["duration"],
            "value": art_label,
            "confidence": 1.0 if art_label != "pluck" else None,
            "string": e["string"],
            "midi": e["midi"],
        })
    annotations.append({
        "annotation_metadata": {
            "curator": {"name": "synthetic", "email": ""},
            "annotator": {},
            "version": "1.0",
            "corpus": "SyntheticGuitar",
            "annotation_tools": "generate_synthetic.py",
            "annotation_rules": "",
            "validation": "",
            "data_source": "all",
        },
        "namespace": "articulation",
        "data": art_obs,
        "sandbox": {},
        "time": 0,
        "duration": duration,
    })

    jams = {
        "annotations": annotations,
        "file_metadata": {
            "title": title,
            "artist": "synthetic",
            "release": "",
            "duration": duration,
            "identifiers": {},
            "jams_version": "0.3.4",
        },
        "sandbox": {},
    }
    return jams


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_track(
    track_id: str,
    duration: float,
    bpm: float | None = None,
    augment: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Generate one synthetic guitar track.

    Returns (audio_array, jams_dict).
    """
    if bpm is None:
        bpm = random.uniform(60, 180)

    # Pick a pattern generator
    gen = random.choice([
        generate_single_notes,
        generate_chords,
        generate_arpeggios,
        generate_hammer_ons,
        generate_mixed_pattern,
    ])
    events = gen(duration, bpm)

    # Render audio
    audio = render_events(events, duration)

    # Apply recording augmentations
    if augment:
        audio = augment_audio(audio)

    # Build JAMS
    jams = events_to_jams(events, duration, title=track_id)

    return audio, jams


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic guitar training data"
    )
    parser.add_argument(
        "--num-tracks", type=int, default=500,
        help="Number of tracks to generate (default: 500)",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Duration of each track in seconds (default: 30)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("SyntheticGuitar"),
        help="Output directory (default: SyntheticGuitar/)",
    )
    parser.add_argument(
        "--player-id", type=str, default="SY",
        help="Player ID prefix for filenames (default: SY)",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable audio augmentations (reverb, EQ, noise, etc.)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    ann_dir = args.out_dir / "annotation"
    audio_dir = args.out_dir / "audio_mono-mic"
    ann_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    styles = ["Single", "Chord", "Arpeg", "Mixed"]
    total_duration = 0.0

    print(f"Generating {args.num_tracks} synthetic tracks ({args.duration}s each)...")
    print(f"Output: {args.out_dir}/")

    for i in range(args.num_tracks):
        style = styles[i % len(styles)]
        bpm = random.uniform(60, 180)
        track_id = f"{args.player_id}_{style}{i:04d}-{int(bpm)}"

        audio, jams = generate_track(
            track_id,
            duration=args.duration,
            bpm=bpm,
            augment=not args.no_augment,
        )

        # Write audio
        wav_name = f"{track_id}_mic.wav"
        sf.write(str(audio_dir / wav_name), audio, SAMPLE_RATE)

        # Write JAMS (filename without _mic suffix)
        jams_name = f"{track_id}.jams"
        with open(ann_dir / jams_name, "w") as f:
            json.dump(jams, f, indent=2)

        total_duration += args.duration

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:5d}/{args.num_tracks}] {total_duration/60:.1f} min generated")

    print(f"\nDone! {args.num_tracks} tracks, {total_duration/60:.1f} min total.")
    print(f"  Annotations: {ann_dir}/")
    print(f"  Audio:       {audio_dir}/")
    print(f"\nTo train on this data:")
    print(f"  python -m model.train --root {args.out_dir} --epochs 100")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
