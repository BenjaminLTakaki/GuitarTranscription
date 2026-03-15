#!/usr/bin/env python3
"""Generate a short chord progression test audio using FluidSynth + SF2.

Chords (each 1.5 s, 0.2 s gap):
  C major : C3, E3, G3   (MIDI 48, 52, 55)
  G major : G2, B2, D3, G3 (MIDI 43, 47, 50, 55)
  A minor : A2, E3, A3   (MIDI 45, 52, 57)
  F major : F2, A2, C3, F3 (MIDI 41, 45, 48, 53)

Output:
  test/audioChords.wav   — 22050 Hz mono
  test/audioChords.jams  — ground truth annotation
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
GUITAR_TUNING = (40, 45, 50, 55, 59, 64)
NUM_FRETS = 21

CHORD_DUR = 1.5    # seconds each chord sustains
GAP = 0.2          # seconds silence between chords

# Ground truth chords (sounding MIDI pitches)
CHORDS = [
    {
        "name": "C major",
        "notes": [
            {"midi": 48, "string": 1, "fret": 3},   # C3 on A string fret 3
            {"midi": 52, "string": 2, "fret": 2},   # E3 on D string fret 2
            {"midi": 55, "string": 3, "fret": 0},   # G3 on G string open
        ],
    },
    {
        "name": "G major",
        "notes": [
            {"midi": 43, "string": 0, "fret": 3},   # G2 on low E fret 3
            {"midi": 47, "string": 1, "fret": 2},   # B2 on A string fret 2
            {"midi": 50, "string": 2, "fret": 0},   # D3 on D string open
            {"midi": 55, "string": 3, "fret": 0},   # G3 on G string open
        ],
    },
    {
        "name": "A minor",
        "notes": [
            {"midi": 45, "string": 1, "fret": 0},   # A2 on A string open
            {"midi": 52, "string": 2, "fret": 2},   # E3 on D string fret 2
            {"midi": 57, "string": 3, "fret": 2},   # A3 on G string fret 2
        ],
    },
    {
        "name": "F major",
        "notes": [
            {"midi": 41, "string": 0, "fret": 1},   # F2 on low E fret 1
            {"midi": 45, "string": 1, "fret": 0},   # A2 on A string open
            {"midi": 48, "string": 2, "fret": -2},  # C3 — we'll use string 1 fret 3 alt
            {"midi": 53, "string": 2, "fret": 3},   # F3 on D string fret 3
        ],
    },
]

# Fix F major C3: string 1 fret 3 = A(45)+3 = 48 ✓
CHORDS[3]["notes"][2] = {"midi": 48, "string": 1, "fret": 3}


def _string_for_midi(midi: int):
    """Find best (string, fret) for a MIDI pitch."""
    candidates = []
    for s, open_m in enumerate(GUITAR_TUNING):
        fret = midi - open_m
        if 0 <= fret < NUM_FRETS:
            candidates.append((s, fret))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1])[0]


def build_events():
    """Build flat note event list with exact onset/offset times."""
    events = []
    t = 0.0
    for chord in CHORDS:
        for note in chord["notes"]:
            s, f = note["string"], note["fret"]
            # Validate
            if f < 0:
                pos = _string_for_midi(note["midi"])
                if pos:
                    s, f = pos
                else:
                    continue
            events.append({
                "midi": note["midi"],
                "string": s,
                "fret": f,
                "onset": round(t, 4),
                "duration": CHORD_DUR,
                "velocity": 80,
                "chord_name": chord["name"],
            })
        t += CHORD_DUR + GAP
    return events


def events_to_midi(events, bpm=120, program=25):
    """Convert note events to a MIDI file for FluidSynth rendering."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("program_change", program=program, channel=0, time=0))
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    raw = []
    for e in events:
        raw.append((e["onset"], "on", e["midi"], e["velocity"]))
        raw.append((e["onset"] + e["duration"], "off", e["midi"], 0))
    raw.sort(key=lambda x: (x[0], 0 if x[1] == "on" else 1))

    cur_tick = 0
    for t_sec, kind, note, vel in raw:
        abs_tick = int(mido.second2tick(t_sec, mid.ticks_per_beat, tempo))
        delta = max(0, abs_tick - cur_tick)
        if kind == "on":
            track.append(mido.Message("note_on", note=note, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=note, velocity=0, time=delta))
        cur_tick = abs_tick
    return mid


def render_fluidsynth(midi_path: Path, wav_path: Path, sf2: Path):
    """Render MIDI to WAV using FluidSynth CLI."""
    cmd = [
        "fluidsynth", "-ni", str(sf2),
        str(midi_path),
        "-F", str(wav_path),
        "-r", str(SAMPLE_RATE),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth failed: {result.stderr}")
    # Convert to mono if stereo
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(str(wav_path), data, sr)


def build_jams(events, total_duration):
    """Build a JAMS annotation dict for ground truth."""
    jams = {
        "file_metadata": {
            "title": "Chord Benchmark Test",
            "duration": total_duration,
        },
        "sandbox": {},
        "annotations": [],
    }

    # Per-string note_midi annotations (matching GuitarSet format)
    for string_idx in range(6):
        string_events = [e for e in events if e["string"] == string_idx]
        obs = []
        for e in string_events:
            obs.append({
                "time": e["onset"],
                "duration": e["duration"],
                "value": e["midi"],
                "confidence": 1.0,
            })
        jams["annotations"].append({
            "namespace": "note_midi",
            "data": obs,
            "annotation_metadata": {
                "data_source": "ground_truth",
                "corpus": "chord_benchmark",
            },
            "sandbox": {"string_index": string_idx},
        })

    return jams


def main():
    out_dir = Path("test")
    out_dir.mkdir(parents=True, exist_ok=True)

    events = build_events()
    total_dur = max(e["onset"] + e["duration"] for e in events) + 0.5

    print(f"Ground truth: {len(events)} notes across {len(CHORDS)} chords")
    for chord in CHORDS:
        notes_str = ", ".join(f"MIDI {n['midi']}" for n in chord["notes"])
        print(f"  {chord['name']}: {notes_str}")

    # Build MIDI
    mid = events_to_midi(events)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid.save(tmp.name)
        midi_path = Path(tmp.name)

    # Render with FluidSynth
    wav_path = out_dir / "audioChords.wav"
    sf2_path = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    if not sf2_path.exists():
        sf2_path = Path("/usr/share/sounds/sf2/TimGM6mb.sf2")
    if not sf2_path.exists():
        sf2_path = Path("/usr/share/sounds/sf2/default-GM.sf2")

    print(f"Rendering with FluidSynth using {sf2_path.name}...")
    render_fluidsynth(midi_path, wav_path, sf2_path)
    midi_path.unlink()

    # Save JAMS ground truth
    jams = build_jams(events, total_dur)
    jams_path = out_dir / "audioChords.jams"
    with open(jams_path, "w") as f:
        json.dump(jams, f, indent=2)

    print(f"Audio saved: {wav_path}")
    print(f"JAMS saved:  {jams_path}")
    print(f"Duration:    {total_dur:.1f}s")


if __name__ == "__main__":
    main()
