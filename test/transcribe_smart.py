#!/usr/bin/env python3
"""
Smart guitar transcription: ML pitch detection + music21 post-processing brain.

Pipeline
--------
1. Pitch detection — uses the project's existing ML model (CQT + GuitarSet fine-tune).
   Falls back to librosa pyin if the checkpoint is missing.
2. music21 analysis — key detection, scale-based note filtering, chord labelling.
3. Fingering optimisation — picks the most position-efficient string/fret combo
   for each detected MIDI pitch, minimising left-hand position jumps.
4. Output — refined MIDI file + printed ASCII tab + key / chord summary.

NOTE on Basic Pitch
-------------------
basic-pitch (Spotify) is an excellent polyphonic neural pitch detector but its
TensorFlow dependency caps at <2.15.1, which has no wheels for Python 3.12.
The code below has a hook (see `_basic_pitch_notes`) showing exactly where it
would slot in once that incompatibility is resolved.

Usage
-----
    python test/transcribe_smart.py audio.wav
    python test/transcribe_smart.py audio.wav -o test/output/my_song.mid
    python test/transcribe_smart.py audio.wav --backend pyin   # force pyin fallback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import mido
import numpy as np

# music21 is the "brain" — key detection, chord analysis, scale filtering
import music21
from music21 import stream, note, chord, analysis, key as m21key

# ── project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.constants import (
    CHECKPOINT_DIR,
    GUITAR_TUNING,
    NUM_STRINGS,
    NUM_FRETS,
    HOP_LENGTH,
    SAMPLE_RATE,
)

# ── guitar constants ──────────────────────────────────────────────────────────
GUITAR_MIN_MIDI = 40   # E2
GUITAR_MAX_MIDI = 88   # E6


# ─────────────────────────────────────────────────────────────────────────────
# 1. PITCH DETECTION BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

def _ml_model_notes(audio_path: Path) -> list[dict]:
    """Run the project's trained CQT model and return raw note events."""
    import torch
    from model.predict import predict, pianoroll_to_notes

    checkpoint = PROJECT_ROOT / CHECKPOINT_DIR / "best_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_prob, onset_prob, art_prob = predict(audio_path, checkpoint, device)
    notes = pianoroll_to_notes(frame_prob, onset_prob, art_prob=art_prob)
    return notes   # list of {midi, string, fret, start, end, velocity, articulation}


def _pyin_notes(audio_path: Path) -> list[dict]:
    """Monophonic fallback using librosa pyin."""
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH,
    )
    frame_sec = HOP_LENGTH / SAMPLE_RATE
    notes, active_midi, active_start = [], None, 0.0
    for t_idx, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        t = t_idx * frame_sec
        midi = int(np.round(librosa.hz_to_midi(freq))) if voiced and not np.isnan(freq) else None
        if midi is not None:
            if active_midi is None:
                active_midi, active_start = midi, t
            elif midi != active_midi:
                notes.append({"midi": active_midi, "start": active_start, "end": t,
                               "velocity": 80, "string": None, "fret": None})
                active_midi, active_start = midi, t
        else:
            if active_midi is not None:
                notes.append({"midi": active_midi, "start": active_start, "end": t,
                               "velocity": 80, "string": None, "fret": None})
                active_midi = None
    if active_midi is not None:
        notes.append({"midi": active_midi, "start": active_start,
                      "end": active_start + frame_sec, "velocity": 80,
                      "string": None, "fret": None})
    return notes


def _basic_pitch_notes(audio_path: Path) -> list[dict]:
    """
    Hook for Spotify Basic Pitch (polyphonic neural detector).

    Currently blocked by basic-pitch pinning tensorflow<2.15.1, which has no
    Python 3.12 wheels.  When that's resolved, uncomment the body below.
    """
    raise NotImplementedError(
        "basic-pitch is not yet compatible with Python 3.13 on Linux.\n"
        "It pins tensorflow<2.15.1 which has no Python 3.13 wheels.\n"
        "Use --backend ml (default) or --backend pyin instead."
    )
    # ── future implementation ────────────────────────────────────────────────
    # from basic_pitch.inference import predict as bp_predict
    # from basic_pitch import ICASSP_2022_MODEL_PATH
    # _, midi_data, note_events = bp_predict(str(audio_path), ICASSP_2022_MODEL_PATH)
    # notes = []
    # for start_s, end_s, pitch_midi, confidence, _ in note_events:
    #     notes.append({"midi": int(pitch_midi), "start": float(start_s),
    #                   "end": float(end_s), "velocity": int(confidence * 127),
    #                   "string": None, "fret": None})
    # return notes


BACKENDS = {
    "ml":          _ml_model_notes,
    "pyin":        _pyin_notes,
    "basic-pitch": _basic_pitch_notes,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. MUSIC21 BRAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_music21_stream(notes: list[dict]) -> stream.Stream:
    """Convert raw note dicts to a music21 Stream for analysis."""
    s = stream.Stream()
    for n in notes:
        midi_pitch = n["midi"]
        duration_sec = max(n["end"] - n["start"], 0.05)
        # music21 duration in quarter notes; assume 120 BPM → 1 quarter = 0.5 s
        quarter_length = duration_sec / 0.5
        p = note.Note()
        p.pitch.midi = midi_pitch
        p.duration.quarterLength = max(quarter_length, 0.125)
        p.offset = n["start"] / 0.5   # convert seconds → quarter-note offset
        s.insert(p.offset, p)
    return s


def detect_key(s: stream.Stream) -> m21key.Key:
    """Use music21's Krumhansl-Schmuckler key-finding algorithm."""
    detected = s.analyze("key")
    return detected


def filter_by_key(notes: list[dict], detected_key: m21key.Key,
                  tolerance_semitones: int = 0) -> list[dict]:
    """
    Drop notes that fall outside the detected key's scale.

    `tolerance_semitones=0` keeps only notes strictly in the scale.
    Set to 1 to also allow chromatic neighbours (passing tones etc.).
    """
    scale_pitches = set(p.midi % 12 for p in detected_key.pitches)

    # Expand tolerance: accept notes within ±tolerance_semitones of a scale tone
    if tolerance_semitones > 0:
        expanded = set()
        for sp in scale_pitches:
            for delta in range(-tolerance_semitones, tolerance_semitones + 1):
                expanded.add((sp + delta) % 12)
        scale_pitches = expanded

    kept, dropped = [], []
    for n in notes:
        if n["midi"] % 12 in scale_pitches:
            kept.append(n)
        else:
            dropped.append(n)

    return kept, dropped


def label_chords(s: stream.Stream) -> list[tuple[float, str]]:
    """
    Analyse the stream in 2-bar windows and return (offset, chord_label) pairs.
    Returns an empty list if music21 can't identify chords (too sparse).
    """
    try:
        chords_found = s.chordify()
        labels = []
        for c in chords_found.flatten().getElementsByClass(chord.Chord):
            label = c.commonName or c.pitchedCommonName
            labels.append((float(c.offset) * 0.5, label))  # back to seconds
        return labels
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 3. GUITAR FINGERING OPTIMISER
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_positions(midi_pitch: int) -> list[tuple[int, int]]:
    """Return all valid (string, fret) pairs for a given MIDI pitch."""
    candidates = []
    for string_idx, open_midi in enumerate(GUITAR_TUNING):
        fret = midi_pitch - open_midi
        if 0 <= fret < NUM_FRETS:
            candidates.append((string_idx, fret))
    return candidates


def assign_fingering(notes: list[dict]) -> list[dict]:
    """
    For notes where string/fret is unknown (pyin / basic-pitch output),
    assign the most position-efficient fingering using a greedy algorithm:
    prefer the candidate that minimises the fret-position jump from the
    previous note, tie-breaking toward lower frets (open strings preferred).

    Notes that already have string/fret set (ML model output) are kept as-is.
    """
    result = []
    prev_fret = 0  # start at open position

    for n in notes:
        n = dict(n)  # don't mutate original
        if n.get("string") is None or n.get("fret") is None:
            candidates = _candidate_positions(n["midi"])
            if not candidates:
                # pitch out of guitar range — skip
                continue
            # pick candidate minimising |fret - prev_fret|, then lowest fret
            best = min(candidates, key=lambda sf: (abs(sf[1] - prev_fret), sf[1]))
            n["string"], n["fret"] = best
            prev_fret = n["fret"]
        else:
            prev_fret = n["fret"]
        result.append(n)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def write_midi(notes: list[dict], output_path: Path, bpm: int = 120):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage("track_name", name="Guitar (Smart)", time=0))
    track.append(mido.Message("program_change", program=25, channel=0, time=0))

    events = []
    for n in notes:
        vel = n.get("velocity", 80)
        events.append(("on",  n["start"], n["midi"], vel))
        events.append(("off", n["end"],   n["midi"], vel))
    events.sort(key=lambda e: e[1])

    current_tick = 0
    for kind, time_sec, midi_note, velocity in events:
        abs_tick = int(round(mido.second2tick(time_sec, midi_file.ticks_per_beat, tempo)))
        delta = max(0, abs_tick - current_tick)
        if kind == "on":
            track.append(mido.Message("note_on",  note=midi_note, velocity=velocity, time=delta))
        else:
            track.append(mido.Message("note_off", note=midi_note, velocity=0,        time=delta))
        current_tick = abs_tick

    midi_file.save(str(output_path))


_STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "e4"]

def print_ascii_tab(notes: list[dict], max_cols: int = 72):
    """Print a simple ASCII tab of the detected notes."""
    if not notes:
        print("  (no notes to display)")
        return

    # Group notes into time slots (one slot ≈ 0.25 s)
    slot_sec = 0.25
    max_time = max(n["end"] for n in notes)
    n_slots = int(max_time / slot_sec) + 1

    # 6 strings, each slot holds a fret number or '-'
    grid = [["-"] * n_slots for _ in range(NUM_STRINGS)]
    for n in notes:
        slot = int(n["start"] / slot_sec)
        s, f = n.get("string"), n.get("fret")
        if s is not None and f is not None and slot < n_slots:
            grid[s][slot] = str(f)

    print("\n-- ASCII Tab --------------------------------------------")
    for str_idx in range(NUM_STRINGS - 1, -1, -1):  # high-e first
        label = f"{_STRING_NAMES[str_idx]:>3}|"
        row = ""
        for slot in range(n_slots):
            cell = grid[str_idx][slot]
            row += cell.ljust(2, "-")
            if len(label) + len(row) > max_cols:
                print(label + row)
                row = ""
                label = "   |"
        if row:
            print(label + row)
    print("-" * 50)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smart guitar transcription: ML/pyin pitch detection + music21 brain"
    )
    parser.add_argument("audio_file", type=Path, help="Input audio (wav/mp3/flac)")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("test/output/smart_transcription.mid"))
    parser.add_argument(
        "--backend", choices=list(BACKENDS), default="ml",
        help="Pitch detection backend (default: ml — uses the trained checkpoint)"
    )
    parser.add_argument(
        "--key-tolerance", type=int, default=1,
        help="Semitone tolerance when filtering off-key notes (0=strict, 1=allow passing tones)"
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Skip key-based note filtering (keep all detected notes)"
    )
    parser.add_argument("--tab", action="store_true", default=True,
                        help="Print ASCII tab (default: on)")
    parser.add_argument("--no-tab", dest="tab", action="store_false")
    args = parser.parse_args()

    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    # ── Step 1: pitch detection ──────────────────────────────────────────────
    print(f"\n[1/4] Detecting pitches with backend: {args.backend!r}")
    backend_fn = BACKENDS[args.backend]
    try:
        raw_notes = backend_fn(args.audio_file)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Falling back to pyin...")
        raw_notes = _pyin_notes(args.audio_file)

    print(f"  Raw notes detected: {len(raw_notes)}")

    # ── Step 2: music21 brain ────────────────────────────────────────────────
    print("\n[2/4] Running music21 analysis...")
    m21_stream = build_music21_stream(raw_notes)
    detected_key = detect_key(m21_stream)
    print(f"  Detected key : {detected_key} (confidence: {detected_key.correlationCoefficient:.3f})")

    if not args.no_filter:
        kept, dropped = filter_by_key(raw_notes, detected_key, args.key_tolerance)
        print(f"  Notes kept   : {len(kept)}  (dropped {len(dropped)} off-key)")
        refined_notes = kept
    else:
        refined_notes = raw_notes
        dropped = []

    chord_labels = label_chords(build_music21_stream(refined_notes))
    if chord_labels:
        print(f"  Chord events : {len(chord_labels)}")

    # ── Step 3: fingering assignment ─────────────────────────────────────────
    print("\n[3/4] Optimising guitar fingering...")
    fingered_notes = assign_fingering(refined_notes)
    print(f"  Notes with fingering: {len(fingered_notes)}")

    # ── Step 4: output ───────────────────────────────────────────────────────
    print(f"\n[4/4] Writing output...")
    write_midi(fingered_notes, args.output)
    print(f"  MIDI saved -> {args.output}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n-- Summary ----------------------------------------------")
    print(f"  Audio          : {args.audio_file.name}")
    print(f"  Backend        : {args.backend}")
    print(f"  Detected key   : {detected_key}")
    print(f"  Raw notes      : {len(raw_notes)}")
    print(f"  After filtering: {len(refined_notes)}")
    if dropped:
        dropped_names = sorted(set(librosa.midi_to_note(n["midi"]) for n in dropped))
        print(f"  Dropped notes  : {', '.join(dropped_names)}")
    if chord_labels:
        print("\n  Chord progression (first 10 events):")
        for t, label in chord_labels[:10]:
            print(f"    {t:6.2f}s  {label}")
    unique_pos = sorted(set((n["string"], n["fret"]) for n in fingered_notes
                            if n.get("string") is not None))
    print(f"\n  Unique positions used: {len(unique_pos)}")

    if args.tab:
        print_ascii_tab(fingered_notes)


if __name__ == "__main__":
    main()
