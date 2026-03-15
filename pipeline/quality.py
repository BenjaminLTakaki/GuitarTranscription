"""Phase 3 — Automated quality control: chunk, filter, and export.

Slices aligned audio + labels into fixed-length chunks, filters by
alignment quality, and exports passing chunks in GuitarSet-compatible
format (annotation/*.jams + audio_mono-mic/*.wav).
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from model.constants import HOP_LENGTH, NUM_STRINGS, SAMPLE_RATE


def events_to_jams(events: list[dict], duration: float, title: str = "") -> dict:
    """Convert events to GuitarSet-compatible JAMS format.

    Events must have keys: onset, offset, string, fret, midi, velocity.
    """
    annotations = []
    for string_idx in range(NUM_STRINGS):
        # Pitch contour placeholder
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "aligned-pipeline", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "AlignedDataset",
                "annotation_tools": "pipeline/quality.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "pitch_contour",
            "data": [],
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })
        # Note MIDI events for this string
        string_notes = sorted(
            [e for e in events if e.get("string") == string_idx],
            key=lambda e: e["onset"],
        )
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "aligned-pipeline", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "AlignedDataset",
                "annotation_tools": "pipeline/quality.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "note_midi",
            "data": [
                {
                    "time": e["onset"],
                    "duration": e["offset"] - e["onset"],
                    "value": float(e["midi"]),
                    "confidence": None,
                }
                for e in string_notes
            ],
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })

    return {
        "annotations": annotations,
        "file_metadata": {
            "title": title,
            "artist": "aligned-pipeline",
            "release": "",
            "duration": duration,
            "identifiers": {},
            "jams_version": "0.3.4",
        },
        "sandbox": {},
    }


def chunk_and_filter(
    real_audio_path: str,
    warped_events: list[dict],
    warp_path: np.ndarray,
    output_dir: Path,
    song_id: str,
    chunk_duration: float = 5.0,
    max_dtw_cost_per_frame: float = 0.5,
) -> int:
    """Slice aligned audio+labels into chunks, filter, and export.

    Returns the number of chunks saved.
    """
    # Load real audio
    y, sr = librosa.load(str(real_audio_path), sr=SAMPLE_RATE, mono=True)
    total_duration = len(y) / SAMPLE_RATE

    ann_dir = output_dir / "annotation"
    audio_dir = output_dir / "audio_mono-mic"
    ann_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    frame_dur = HOP_LENGTH / SAMPLE_RATE
    chunks_saved = 0

    # Compute local alignment quality from warp path
    real_frames = warp_path[:, 0]
    synth_frames = warp_path[:, 1]

    # Advancement ratios: how evenly the warp path advances
    if len(warp_path) > 1:
        real_diffs = np.diff(real_frames).astype(float)
        synth_diffs = np.diff(synth_frames).astype(float)
        # Avoid division by zero
        synth_diffs = np.where(synth_diffs == 0, 1.0, synth_diffs)
        ratios = real_diffs / synth_diffs
    else:
        ratios = np.array([1.0])

    # Chunk the audio
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    num_chunks = int(np.ceil(total_duration / chunk_duration))

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_duration
        chunk_end = min((chunk_idx + 1) * chunk_duration, total_duration)
        actual_duration = chunk_end - chunk_start

        # Skip short chunks (< 90% of target)
        if actual_duration < chunk_duration * 0.9:
            continue

        # Find events in this chunk
        chunk_events = []
        for ev in warped_events:
            # Event overlaps with chunk
            if ev["onset"] < chunk_end and ev["offset"] > chunk_start:
                e = dict(ev)
                # Clip to chunk boundaries and adjust to chunk-relative time
                e["onset"] = max(0.0, e["onset"] - chunk_start)
                e["offset"] = min(actual_duration, e["offset"] - chunk_start)
                if e["offset"] - e["onset"] >= 0.03:  # minimum 30ms
                    chunk_events.append(e)

        # Skip chunks with too few notes
        if len(chunk_events) < 2:
            continue

        # Compute local alignment quality for this chunk
        chunk_start_frame = int(chunk_start / frame_dur)
        chunk_end_frame = int(chunk_end / frame_dur)

        # Find warp path entries in this chunk's frame range
        mask = (real_frames >= chunk_start_frame) & (real_frames < chunk_end_frame)
        local_ratios = ratios[mask[:-1] if len(mask) > len(ratios) else mask[:len(ratios)]]

        if len(local_ratios) > 2:
            # High std of ratios means poor local alignment
            ratio_std = float(np.std(local_ratios))
            if ratio_std > max_dtw_cost_per_frame:
                continue

        # Extract audio chunk
        start_sample = int(chunk_start * SAMPLE_RATE)
        end_sample = min(start_sample + chunk_samples, len(y))
        chunk_audio = y[start_sample:end_sample]

        # Pad if needed
        if len(chunk_audio) < chunk_samples:
            chunk_audio = np.pad(chunk_audio, (0, chunk_samples - len(chunk_audio)))

        # Export
        chunk_id = f"{song_id}_chunk{chunk_idx:04d}"

        wav_path = audio_dir / f"{chunk_id}_mic.wav"
        sf.write(str(wav_path), chunk_audio, SAMPLE_RATE)

        jams = events_to_jams(chunk_events, actual_duration, title=chunk_id)
        jams_path = ann_dir / f"{chunk_id}.jams"
        with open(jams_path, "w", encoding="utf-8") as f:
            json.dump(jams, f, indent=2)

        chunks_saved += 1

    return chunks_saved
