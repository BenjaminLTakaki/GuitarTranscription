"""Phase 2A — Synthesize tab MIDI to audio via FluidSynth.

Renders a MIDI file to WAV for use as the DTW reference signal.
Reuses the FluidSynth helpers from generate_sf2.py.
"""

from __future__ import annotations

from pathlib import Path

from generate_sf2 import _get_soundfont, render_midi_to_wav
from model.constants import SAMPLE_RATE


def synthesize_midi(midi_path: Path, output_wav: Path) -> bool:
    """Render a MIDI file to WAV using FluidSynth.

    Returns True on success.
    """
    _, sf_host = _get_soundfont()
    return render_midi_to_wav(midi_path, output_wav, sf_host, sample_rate=SAMPLE_RATE)


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
