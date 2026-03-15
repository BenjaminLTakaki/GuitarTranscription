"""Phase 1C — Source separation with Demucs.

Isolates the guitar/other stem from a full mix using Demucs,
then resamples to 22050 Hz mono for downstream processing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from model.constants import SAMPLE_RATE


def isolate_guitar(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
) -> Path | None:
    """Run Demucs to extract the guitar/other stem.

    Tries htdemucs_6s first (dedicated guitar stem), falls back to
    the standard 4-stem 'other' stem.

    Returns path to the isolated guitar WAV, or None on failure.
    """
    cmd = [
        "python3", "-m", "demucs",
        "--two-stems", "other",
        "-n", model,
        "--out", str(output_dir),
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Demucs timed out for: {audio_path.name}")
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip()[:200] if result.stderr else ""
        print(f"  [FAIL] Demucs error for '{audio_path.name}': {stderr}")
        return None

    # Find the output stem
    stem_dir = output_dir / model / audio_path.stem
    other_stem = stem_dir / "other.wav"
    if not other_stem.exists():
        # Try guitar stem from 6-stem model
        guitar_stem = stem_dir / "guitar.wav"
        if guitar_stem.exists():
            other_stem = guitar_stem
        else:
            print(f"  [FAIL] Demucs output not found: {other_stem}")
            return None

    # Resample to project standard: 22050 Hz mono
    resampled = _resample_mono(other_stem)
    return resampled


def _resample_mono(wav_path: Path) -> Path:
    """Resample a WAV file to 22050 Hz mono in-place, return the path."""
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    sf.write(str(wav_path), y, SAMPLE_RATE)
    return wav_path


def isolate_batch(
    audio_paths: list[Path],
    output_dir: Path,
    model: str = "htdemucs",
) -> dict[str, Path]:
    """Isolate guitar stems for a batch of audio files.

    Returns a mapping of input stem name → isolated WAV path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for i, audio_path in enumerate(audio_paths):
        print(f"  [{i+1}/{len(audio_paths)}] Separating: {audio_path.name}")
        isolated = isolate_guitar(audio_path, output_dir, model)
        if isolated is not None:
            results[audio_path.stem] = isolated

    print(f"Separated {len(results)}/{len(audio_paths)} tracks")
    return results
