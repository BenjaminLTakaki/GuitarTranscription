#!/usr/bin/env python3
"""Generate synthetic guitar audio using FluidSynth + SoundFont (SF2).

Audio is rendered from MIDI using real guitar samples, producing far more
realistic training data than the Karplus-Strong synthesiser.

Prerequisites
-------------
Install fluidsynth from a real (non-Flatpak) host terminal:
    sudo apt install fluidsynth

Then run from VS Code's terminal as normal:
    python generate_sf2.py --num-tracks 500 --duration 30

Output layout matches GuitarSet so the existing dataset loader works:
    SyntheticGuitar_SF2/
        annotation/*.jams
        audio_mono-mic/*_mic.wav
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import mido
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 22050

# Standard tuning: MIDI note for each open string (low E → high e)
GUITAR_TUNING = (40, 45, 50, 55, 59, 64)
NUM_STRINGS = 6
NUM_FRETS = 21  # 0–20

# GM MIDI program numbers best suited to acoustic/electric guitar
# 24 = Acoustic Nylon Guitar  25 = Acoustic Steel Guitar
# 26 = Electric Jazz Guitar   27 = Electric Clean Guitar  28 = Electric Muted
_GUITAR_PROGRAMS = [24, 25, 26, 27, 28]

# Candidate SoundFont paths — listed as (sandbox_path, host_path) pairs.
# When running via flatpak-spawn --host, the host path is passed to the CLI
# because /run/host/... does not exist from the host's perspective.
_SOUNDFONT_SEARCH = [
    (Path("/run/host/usr/share/sounds/sf2/TimGM6mb.sf2"),  Path("/usr/share/sounds/sf2/TimGM6mb.sf2")),
    (Path("/run/host/usr/share/sounds/sf2/default-GM.sf2"), Path("/usr/share/sounds/sf2/default-GM.sf2")),
    (Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),           Path("/usr/share/sounds/sf2/TimGM6mb.sf2")),
    (Path("/usr/share/sounds/sf2/default-GM.sf2"),          Path("/usr/share/sounds/sf2/default-GM.sf2")),
]

# Flatpak sandbox: the fluidsynth binary lives on the host, not inside VS Code
_FLUIDSYNTH_CMD: list[str] | None = None  # resolved lazily by _get_fluidsynth()
_USE_FLATPAK_SPAWN: bool = False           # set True when using flatpak-spawn --host


def _get_fluidsynth() -> list[str]:
    """Return the command prefix to invoke fluidsynth (inside or outside Flatpak)."""
    global _FLUIDSYNTH_CMD
    if _FLUIDSYNTH_CMD is not None:
        return _FLUIDSYNTH_CMD

    # Direct (non-Flatpak) or Flatpak with fluidsynth in sandbox
    if shutil.which("fluidsynth"):
        _FLUIDSYNTH_CMD = ["fluidsynth"]
        return _FLUIDSYNTH_CMD

    # VS Code Flatpak sandbox — escape to host via flatpak-spawn
    if shutil.which("flatpak-spawn"):
        result = subprocess.run(
            ["flatpak-spawn", "--host", "which", "fluidsynth"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            global _USE_FLATPAK_SPAWN
            _USE_FLATPAK_SPAWN = True
            _FLUIDSYNTH_CMD = ["flatpak-spawn", "--host", "fluidsynth"]
            return _FLUIDSYNTH_CMD

    raise RuntimeError(
        "fluidsynth not found.\n"
        "Install it from a real (non-VS Code) terminal:\n"
        "  sudo apt install fluidsynth\n"
        "Then re-run this script."
    )


def _get_soundfont() -> tuple[Path, Path]:
    """Return (sandbox_path, host_path) for the first available SoundFont.

    sandbox_path is used to check existence (visible inside Flatpak).
    host_path is what gets passed to the fluidsynth CLI on the host.
    """
    for sandbox_p, host_p in _SOUNDFONT_SEARCH:
        if sandbox_p.exists():
            return sandbox_p, host_p
    raise RuntimeError(
        f"No SoundFont file found. Searched:\n"
        + "\n".join(f"  {sp}" for sp, _ in _SOUNDFONT_SEARCH)
    )


# ---------------------------------------------------------------------------
# Musical pattern generators (identical logic to generate_synthetic.py)
# ---------------------------------------------------------------------------

_CHORD_SHAPES = {
    "major":    [(0, 0), (1, 2), (2, 2), (3, 1), (4, 0), (5, 0)],
    "minor":    [(0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)],
    "power":    [(0, 0), (1, 2), (2, 2)],
    "sus2":     [(0, 0), (1, 2), (2, 2), (3, 2), (4, 0), (5, 0)],
    "dom7":     [(0, 0), (1, 2), (2, 0), (3, 1), (4, 0)],
}
_SCALE_PATTERNS = {
    "major":      [0, 2, 4, 5, 7, 9, 11, 12],
    "minor":      [0, 2, 3, 5, 7, 8, 10, 12],
    "pentatonic": [0, 3, 5, 7, 10, 12],
    "blues":      [0, 3, 5, 6, 7, 10, 12],
    "dorian":     [0, 2, 3, 5, 7, 9, 10, 12],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10, 12],
}


def _string_for_midi(midi: int) -> Tuple[int, int] | None:
    candidates = []
    for s, open_m in enumerate(GUITAR_TUNING):
        fret = midi - open_m
        if 0 <= fret < NUM_FRETS:
            candidates.append((s, fret))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1])[0]


def generate_single_notes(duration: float, bpm: float) -> List[dict]:
    events, t = [], random.uniform(0.0, 0.5)
    beat = 60.0 / bpm
    scale = _SCALE_PATTERNS[random.choice(list(_SCALE_PATTERNS))]
    root = random.randint(40, 60)
    while t < duration - 0.1:
        oct_shift = random.choice([0, 0, 0, 12, -12]) if random.random() < 0.2 else 0
        midi = root + random.choice(scale) + oct_shift
        pos = _string_for_midi(midi)
        if pos is None:
            t += beat * random.choice([0.5, 1])
            continue
        string, fret = pos
        dur = random.choice([0.25, 0.5, 0.5, 1.0, 1.0, 2.0]) * beat
        dur = max(0.08, min(dur, duration - t))
        vel = int(random.uniform(50, 110))
        events.append({"string": string, "fret": fret, "midi": midi,
                        "onset": t, "duration": dur, "velocity": vel})
        t += dur + random.uniform(0.0, 0.15) * (random.random() < 0.5)
    return events


def generate_chords(duration: float, bpm: float) -> List[dict]:
    events, t = [], random.uniform(0.0, 0.3)
    beat = 60.0 / bpm
    patterns = [[1, 1, 1, 1], [2, 1, 1], [2, 2], [0.5, 0.5, 1, 0.5, 0.5, 1]]
    pattern = random.choice(patterns)
    idx = 0
    while t < duration - 0.2:
        shape = _CHORD_SHAPES[random.choice(list(_CHORD_SHAPES))]
        root_fret = random.randint(0, 12)
        strum_delay = random.uniform(0.005, 0.04)
        reverse = random.random() < 0.3
        chord_dur = pattern[idx % len(pattern)] * beat
        chord_dur = min(chord_dur, duration - t)
        ordered = list(reversed(shape)) if reverse else list(shape)
        for i, (rel_s, rel_f) in enumerate(ordered):
            f = root_fret + rel_f
            if f >= NUM_FRETS:
                continue
            onset = t + i * strum_delay
            if onset >= duration:
                break
            midi = GUITAR_TUNING[rel_s] + f
            note_dur = max(0.1, min(chord_dur - i * strum_delay, duration - onset))
            vel = int(random.uniform(55, 110))
            events.append({"string": rel_s, "fret": f, "midi": midi,
                            "onset": onset, "duration": note_dur, "velocity": vel})
        t += chord_dur + random.uniform(0.0, 0.1)
        idx += 1
    return events


def generate_arpeggios(duration: float, bpm: float) -> List[dict]:
    events, t = [], random.uniform(0.0, 0.3)
    beat = 60.0 / bpm
    while t < duration - 0.5:
        shape = _CHORD_SHAPES[random.choice(list(_CHORD_SHAPES))]
        root_fret = random.randint(0, 10)
        direction = random.choice(["up", "down", "updown", "random"])
        note_idxs = list(range(len(shape)))
        if direction == "down":
            note_idxs = list(reversed(note_idxs))
        elif direction == "updown":
            note_idxs = note_idxs + list(reversed(note_idxs[1:-1]))
        elif direction == "random":
            random.shuffle(note_idxs)
        note_dur = random.choice([0.25, 0.5]) * beat
        for i in note_idxs:
            if t >= duration - 0.05:
                break
            rel_s, rel_f = shape[i]
            f = root_fret + rel_f
            if f >= NUM_FRETS:
                continue
            midi = GUITAR_TUNING[rel_s] + f
            ring = min(random.uniform(note_dur, note_dur * 4), duration - t)
            vel = int(random.uniform(45, 95))
            events.append({"string": rel_s, "fret": f, "midi": midi,
                            "onset": t, "duration": ring, "velocity": vel})
            t += note_dur
        t += random.uniform(0.0, beat)
    return events


def generate_mixed(duration: float, bpm: float) -> List[dict]:
    events, t = [], 0.0
    gens = [generate_single_notes, generate_chords, generate_arpeggios]
    while t < duration:
        sec = random.uniform(3.0, min(10.0, duration - t))
        if sec < 1.0:
            break
        gen = random.choice(gens)
        for e in gen(sec, bpm):
            e["onset"] += t
            events.append(e)
        t += sec + random.uniform(0.1, 0.5)
    return events


# ---------------------------------------------------------------------------
# MIDI file generation
# ---------------------------------------------------------------------------

def events_to_midi(
    events: List[dict],
    bpm: float,
    program: int = 25,
) -> mido.MidiFile:
    """Convert note events to a mido MidiFile, all on channel 0."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message("program_change", program=program, channel=0, time=0))
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    raw: list[tuple[float, str, int, int]] = []
    for e in events:
        raw.append((e["onset"], "on", e["midi"], e["velocity"]))
        raw.append((e["onset"] + e["duration"], "off", e["midi"], 0))
    raw.sort(key=lambda x: x[0])

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


# ---------------------------------------------------------------------------
# FluidSynth rendering: MIDI file → WAV via CLI
# ---------------------------------------------------------------------------

def _to_host_path(p: Path) -> str:
    """Convert a sandbox path to the equivalent host path for flatpak-spawn calls.

    Inside the Flatpak sandbox, the host filesystem is mounted at /run/host.
    But when the command runs *on* the host, /run/host does not exist.
    We also need to use the real home directory, not the sandbox /home.
    """
    if not _USE_FLATPAK_SPAWN:
        return str(p)
    s = str(p)
    # /run/host/usr/... → /usr/...
    if s.startswith("/run/host/"):
        return s[len("/run/host"):]
    # Sandbox /tmp is not shared with the host; redirect to the real home cache
    if s.startswith("/tmp/"):
        host_cache = Path.home() / ".cache" / "guitar_synth_tmp"
        host_cache.mkdir(parents=True, exist_ok=True)
        return str(host_cache / Path(s).name)
    return s


def render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    soundfont_host_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> bool:
    """Render a MIDI file to WAV using the fluidsynth CLI.

    Returns True on success.
    """
    # When using flatpak-spawn, MIDI must be written to a path the host can see
    host_midi = _to_host_path(midi_path)
    host_wav  = _to_host_path(wav_path)
    host_sf   = str(soundfont_host_path)

    # If sandbox MIDI path differs from host path, copy the file there first
    if host_midi != str(midi_path):
        import shutil as _sh
        _sh.copy2(str(midi_path), host_midi)

    cmd = _get_fluidsynth() + [
        "-ni",                   # non-interactive, no audio output
        "-g", "0.8",             # gain
        "-r", str(sample_rate),  # sample rate
        host_sf,                 # SoundFont (host path)
        host_midi,               # input MIDI (host path)
        "-F", host_wav,          # output WAV (host path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # If output was written to host path, move it back to the sandbox path
    if result.returncode == 0 and host_wav != str(wav_path):
        import shutil as _sh
        _sh.move(host_wav, str(wav_path))

    return result.returncode == 0


def postprocess_wav(wav_path: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Normalize and apply minor augmentations in-place to rendered WAV."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    # Down-mix to mono if stereo
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed (fluidsynth may output at a different rate)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    # Random EQ tilt
    if random.random() < 0.6:
        from scipy.signal import lfilter
        alpha = random.uniform(0.85, 0.99)
        if random.random() < 0.5:
            audio = lfilter([1 - alpha], [1, -alpha], audio).astype(np.float32)
        else:
            audio = lfilter([1, -1], [1, -alpha], audio).astype(np.float32)

    # Noise floor
    if random.random() < 0.5:
        audio += random.uniform(0.001, 0.008) * np.random.randn(len(audio)).astype(np.float32)

    # Random gain
    audio *= random.uniform(0.6, 1.0)

    # Simple comb reverb
    if random.random() < 0.5:
        delay = int(random.uniform(15, 60) * sample_rate / 1000)
        decay = random.uniform(0.1, 0.3)
        reverbed = np.zeros(len(audio) + delay, dtype=np.float32)
        reverbed[:len(audio)] += audio
        reverbed[delay:delay + len(audio)] += decay * audio
        audio = reverbed[:len(audio)]

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * random.uniform(0.7, 0.95)

    sf.write(str(wav_path), audio, sample_rate)


# ---------------------------------------------------------------------------
# JAMS export
# ---------------------------------------------------------------------------

def events_to_jams(events: List[dict], duration: float, title: str = "synthetic") -> dict:
    """Build a JAMS dict with per-string note_midi annotations (GuitarSet format)."""
    annotations = []
    for string_idx in range(NUM_STRINGS):
        # Interleaved pitch_contour (empty) + note_midi per string
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "synthetic-sf2", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "SyntheticGuitar_SF2",
                "annotation_tools": "generate_sf2.py",
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
        string_notes = sorted(
            [e for e in events if e["string"] == string_idx],
            key=lambda e: e["onset"]
        )
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "synthetic-sf2", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "SyntheticGuitar_SF2",
                "annotation_tools": "generate_sf2.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "note_midi",
            "data": [
                {"time": e["onset"], "duration": e["duration"],
                 "value": float(e["midi"]), "confidence": None}
                for e in string_notes
            ],
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })
    return {
        "annotations": annotations,
        "file_metadata": {
            "title": title, "artist": "synthetic-sf2",
            "release": "", "duration": duration,
            "identifiers": {}, "jams_version": "0.3.4",
        },
        "sandbox": {},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate realistic synthetic guitar data via FluidSynth SF2 rendering"
    )
    parser.add_argument("--num-tracks", type=int, default=500)
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Track duration in seconds (default: 30)")
    parser.add_argument("--out-dir", type=Path, default=Path("SyntheticGuitar_SF2"))
    parser.add_argument("--player-id", type=str, default="SF",
                        help="Filename prefix (default: SF)")
    parser.add_argument("--soundfont", type=Path, default=None,
                        help="Path to .sf2 file (auto-detected if omitted)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip post-processing augmentations")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Resolve tools
    try:
        cmd = _get_fluidsynth()
        print(f"FluidSynth command: {' '.join(cmd)}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    soundfont_sandbox, soundfont_host = (
        (args.soundfont, args.soundfont) if args.soundfont
        else _get_soundfont()
    )
    print(f"SoundFont:  {soundfont_sandbox} (host path: {soundfont_host})")

    ann_dir = args.out_dir.resolve() / "annotation"
    audio_dir = args.out_dir.resolve() / "audio_mono-mic"
    ann_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    styles = ["Single", "Chord", "Arpeg", "Mixed"]
    gens = [generate_single_notes, generate_chords, generate_arpeggios, generate_mixed]
    total_dur = 0.0
    errors = 0

    print(f"\nGenerating {args.num_tracks} tracks ({args.duration}s each)…")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i in range(args.num_tracks):
            style = styles[i % 4]
            bpm = random.uniform(60, 180)
            program = random.choice(_GUITAR_PROGRAMS)
            track_id = f"{args.player_id}_{style}{i:04d}-{int(bpm)}"

            gen = gens[i % 4]
            events = gen(args.duration, bpm)
            if not events:
                errors += 1
                continue

            # Write MIDI
            mid_path = tmp_dir / f"{track_id}.mid"
            mid = events_to_midi(events, bpm, program=program)
            mid.save(str(mid_path))

            # Render MIDI → WAV
            wav_path = audio_dir / f"{track_id}_mic.wav"
            ok = render_midi_to_wav(mid_path, wav_path, soundfont_host)
            if not ok:
                errors += 1
                continue

            # Post-process WAV (augment + resample to 22050 Hz mono)
            if not args.no_augment:
                try:
                    postprocess_wav(wav_path)
                except Exception:
                    pass  # keep the raw render if postprocessing fails

            # Write JAMS
            jams = events_to_jams(events, args.duration, title=track_id)
            with open(ann_dir / f"{track_id}.jams", "w") as f:
                json.dump(jams, f, indent=2)

            total_dur += args.duration
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1:5d}/{args.num_tracks}] "
                      f"{total_dur/60:.1f} min generated  (errors: {errors})")

    ok_count = args.num_tracks - errors
    print(f"\nDone! {ok_count}/{args.num_tracks} tracks, "
          f"{ok_count * args.duration / 60:.1f} min total.")
    print(f"  Annotations: {ann_dir}/")
    print(f"  Audio:       {audio_dir}/")
    print(f"\nTo train:")
    print(f"  python -m model.train --synth-root {args.out_dir} --epochs 50 --lr 6e-4")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
