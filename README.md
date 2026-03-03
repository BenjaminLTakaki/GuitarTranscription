# GuitarTranscription (First Iteration)

This first iteration only detects **pitch over time** from an audio file.

It intentionally does **not** handle:
- fingering
- hammer-ons / pull-offs
- slides, bends, articulations
- full tablature inference

## What it does

- Loads a mono audio file
- Runs monophonic pitch estimation (`librosa.pyin`)
- Converts estimated frequency to note segments
- Writes a MIDI file to `output/` (by default)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python detect_pitches.py /path/to/guitar_audio.wav
```

Optional tuning parameters:

```bash
python detect_pitches.py input.wav -o output/my_take.mid --frame-length 2048 --hop-length 256 --min-note-duration 0.05
```

## Output format

The script outputs a MIDI file (default: `output/detected_pitches.mid`).

Notes:
- Pitch is detected frame-by-frame, then merged into note events.
- This works best for **single-note lines** (monophonic audio).
