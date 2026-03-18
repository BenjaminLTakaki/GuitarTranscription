"""Phase 2A — Synthesize tab MIDI to audio via FluidSynth.

Renders a MIDI file to WAV for use as the DTW reference signal.
Reuses the FluidSynth helpers from generate_sf2.py.
"""

from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from generate_sf2 import _get_soundfont, render_midi_to_wav
from model.constants import SAMPLE_RATE


def synthesize_midi(midi_path: Path, output_wav: Path) -> bool:
    """Render a MIDI file to WAV using FluidSynth.

    Returns True on success.
    """
    try:
        _, sf_host = _get_soundfont()
        return render_midi_to_wav(midi_path, output_wav, sf_host, sample_rate=SAMPLE_RATE)
    except Exception as exc:
        print(f"  [WARN] FluidSynth unavailable ({exc}); using sine fallback synth")
        return _synthesize_midi_fallback(midi_path, output_wav, sample_rate=SAMPLE_RATE)


def _synthesize_midi_fallback(
    midi_path: Path,
    output_wav: Path,
    sample_rate: int = 22050,
) -> bool:
    """Minimal MIDI-to-audio fallback synthesizer for alignment reference.

    Uses simple sine waves with short attack/release envelopes. This is not
    meant for audio quality; it provides a deterministic signal for DTW.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception as exc:
        print(f"  [FAIL] Could not read MIDI for fallback synth: {exc}")
        return False

    tempo = 500000  # default 120 BPM in microseconds per beat
    ticks_per_beat = mid.ticks_per_beat

    active: dict[int, tuple[float, int]] = {}
    notes: list[tuple[float, float, int, int]] = []

    for track in mid.tracks:
        cur_sec = 0.0
        for msg in track:
            cur_sec += mido.tick2second(msg.time, ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (cur_sec, msg.velocity)
            elif msg.type in {"note_off", "note_on"} and msg.note in active:
                start, vel = active.pop(msg.note)
                end = max(cur_sec, start + 0.03)
                notes.append((start, end, msg.note, vel))

    if not notes:
        print("  [FAIL] MIDI contained no note events for fallback synth")
        return False

    end_time = max(n[1] for n in notes) + 0.2
    n_samples = int(end_time * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    for start, end, midi_note, vel in notes:
        i0 = max(0, int(start * sample_rate))
        i1 = min(n_samples, int(end * sample_rate))
        if i1 <= i0:
            continue

        t = np.arange(i1 - i0, dtype=np.float32) / sample_rate
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        amp = (vel / 127.0) * 0.2
        wave = amp * np.sin(2.0 * np.pi * freq * t)

        # Short linear attack/release to avoid clicks.
        n = wave.shape[0]
        attack = min(int(0.008 * sample_rate), n // 4)
        release = min(int(0.020 * sample_rate), n // 4)
        if attack > 0:
            wave[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release > 0:
            wave[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)

        audio[i0:i1] += wave

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio /= peak

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, sample_rate)
    return True


def synthesize_batch(
    midi_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Synthesize a batch of MIDI files to WAV.

    Returns a mapping of MIDI stem name → synthesized WAV path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for i, midi_path in enumerate(midi_paths):
        wav_path = output_dir / f"{midi_path.stem}.wav"
        if wav_path.exists():
            results[midi_path.stem] = wav_path
            continue

        ok = synthesize_midi(midi_path, wav_path)
        if ok:
            results[midi_path.stem] = wav_path
        else:
            print(f"  [FAIL] FluidSynth render failed: {midi_path.name}")

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(midi_paths)}] synthesized")

    print(f"Synthesized {len(results)}/{len(midi_paths)} MIDI files")
    return results
