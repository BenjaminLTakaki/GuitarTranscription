"""Rhythmic quantization: snap note onsets/offsets to a musical grid."""

from __future__ import annotations

import librosa
import numpy as np

from model.constants import SAMPLE_RATE, HOP_LENGTH


def detect_tempo(audio_path: str) -> float:
    """Estimate tempo in BPM using librosa's beat tracker."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    # librosa may return an array; extract scalar
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    return float(tempo)


def quantize_notes(
    notes: list[dict],
    bpm: float,
    grid_subdivision: int = 16,
    fill_gaps: bool = True,
) -> list[dict]:
    """Snap note onsets and offsets to the nearest grid position.

    grid_subdivision=16 means snap to 16th notes.
    grid_subdivision=8 means snap to 8th notes.

    When ``fill_gaps`` is True, if the gap between a note's quantized offset
    and the next note on the same string is less than one grid unit, the
    note's offset is extended to meet the next onset.  This produces cleaner
    notation where chord tones ring until the next strum.
    """
    beat_duration = 60.0 / bpm
    grid_duration = beat_duration / (grid_subdivision / 4)  # e.g. 16th = beat/4

    quantized = []
    for n in notes:
        n = dict(n)
        # Snap onset to nearest grid
        n["start"] = round(n["start"] / grid_duration) * grid_duration
        # Snap offset to nearest grid, ensuring minimum duration of one grid unit
        n["end"] = round(n["end"] / grid_duration) * grid_duration
        if n["end"] <= n["start"]:
            n["end"] = n["start"] + grid_duration
        quantized.append(n)

    # Gap filling: extend notes to meet the next onset on the same string
    if fill_gaps:
        from collections import defaultdict

        by_string: dict[int | None, list[dict]] = defaultdict(list)
        for n in quantized:
            by_string[n.get("string")].append(n)

        for string, group in by_string.items():
            group.sort(key=lambda n: n["start"])
            for i in range(len(group) - 1):
                gap = group[i + 1]["start"] - group[i]["end"]
                if 0 < gap < grid_duration:
                    group[i]["end"] = group[i + 1]["start"]

    return quantized
