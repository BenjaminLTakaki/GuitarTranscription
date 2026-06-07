#!/usr/bin/env python3
"""Pseudo-label guitar audio files using the trained model.

For each audio file found under --audio-dir:
  1. Run the model at --threshold (higher = fewer but more accurate labels)
  2. Skip files with fewer than --min-notes detected (likely no guitar content)
  3. Write a GuitarSet-compatible JAMS annotation + save mono WAV

Output directory follows GuitarSet layout so train.py --mixed --synth-root works directly:

    output_dir/
        annotation/<stem>.jams
        audio_mono-mic/<stem>_mic.wav

Typical workflow
----------------
# 1. Separate guitar stems with Demucs (htdemucs_6s has a dedicated guitar stem)
python -m demucs -n htdemucs_6s --two-stems guitar /path/to/songs/ --out guitar_stems/

# 2. Pseudo-label the separated stems
python pseudo_label.py guitar_stems/ PseudoLabeled/ --checkpoint checkpoints/best_model_v2.pt

# 3. Train with real + pseudo-labeled data combined
python -m model.train --mixed --synth-root PseudoLabeled/ --split trainval \\
    --resume checkpoints/best_model_v2.pt --lr 3e-4 --scheduler cosine --epochs 850
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from model.constants import (
    GUITAR_TUNING,
    HOP_LENGTH,
    NUM_STRINGS,
    SAMPLE_RATE,
)
from model.predict import _detect_model_version, load_cqt, pianoroll_to_notes


# ---------------------------------------------------------------------------
# Model loading (once, reused across all files)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    version = _detect_model_version(state_dict)
    if version == "v2":
        from model.network_v2 import GuitarTranscriptionModelV2
        model = GuitarTranscriptionModelV2().to(device)
    else:
        from model.network import GuitarTranscriptionModel
        model = GuitarTranscriptionModel().to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_model(
    model: torch.nn.Module,
    audio_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    spec = load_cqt(audio_path)
    spec_t = torch.from_numpy(spec).unsqueeze(0).to(device)
    outputs = model(spec_t)
    frame_prob = torch.sigmoid(outputs[0]).squeeze(0).cpu().numpy()
    onset_prob = torch.sigmoid(outputs[1]).squeeze(0).cpu().numpy()
    art_prob = (
        torch.sigmoid(outputs[2]).squeeze(0).cpu().numpy()
        if len(outputs) > 2
        else None
    )
    return frame_prob, onset_prob, art_prob


# ---------------------------------------------------------------------------
# JAMS serialisation
# ---------------------------------------------------------------------------

def _notes_to_jams(notes: list[dict], duration: float) -> dict:
    """Convert note events to a minimal JAMS dict GuitarSetDataset can read."""
    per_string: dict[int, list] = {i: [] for i in range(NUM_STRINGS)}
    for n in notes:
        per_string[n["string"]].append(n)

    annotations = []
    for string_idx in range(NUM_STRINGS):
        data = [
            {
                "time": n["start"],
                "duration": max(0.05, n["end"] - n["start"]),
                "value": GUITAR_TUNING[string_idx] + n["fret"],
                "confidence": 1.0,
            }
            for n in per_string[string_idx]
        ]
        annotations.append({"namespace": "note_midi", "data": data, "sandbox": {}})

    return {"file_metadata": {"duration": duration}, "annotations": annotations}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pseudo-label guitar audio with trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio_dir", type=Path, help="Directory of guitar audio files (searched recursively)")
    parser.add_argument("output_dir", type=Path, help="Output directory (GuitarSet-compatible layout)")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/best_model_v2.pt"),
        help="Model checkpoint (default: checkpoints/best_model_v2.pt)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Frame probability threshold for note detection. "
             "Higher = fewer but more accurate pseudo-labels. "
             "Recommended: 0.80-0.90 (default: 0.85)",
    )
    parser.add_argument(
        "--min-notes", type=int, default=10,
        help="Skip files with fewer detected notes — likely no guitar content (default: 10)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=5.0,
        help="Skip files shorter than this many seconds (default: 5.0)",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Threshold   : {args.threshold}")
    print(f"Min notes   : {args.min_notes}")
    print(f"Min duration: {args.min_duration}s")

    model = _load_model(args.checkpoint, device)
    print("Model loaded.\n")

    ann_dir = args.output_dir / "annotation"
    audio_out = args.output_dir / "audio_mono-mic"
    ann_dir.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    audio_files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"):
        audio_files.extend(args.audio_dir.rglob(ext))
    audio_files = sorted(set(audio_files))

    # If running on MUSDB18-style directories, keep only the "other" stem
    # (drums/bass/vocals already separated; other = guitar + remaining instruments)
    musdb_stems = {"bass", "drums", "mixture", "vocals"}
    musdb_files = [f for f in audio_files if f.stem in musdb_stems]
    if musdb_files:
        audio_files = [f for f in audio_files if f.stem not in musdb_stems]
        print(f"MUSDB18 layout detected — keeping 'other' stems only ({len(audio_files)} files, skipped {len(musdb_files)})\n")

    print(f"Found {len(audio_files)} audio files under {args.audio_dir}\n")

    ok = skipped_notes = skipped_duration = errors = 0

    for i, audio_path in enumerate(audio_files, 1):
        prefix = f"[{i:>{len(str(len(audio_files)))}}/{len(audio_files)}] {audio_path.name}"

        # Skip very short files — not useful as training segments
        try:
            duration = librosa.get_duration(path=str(audio_path))
        except Exception:
            duration = 0.0
        if duration < args.min_duration:
            print(f"{prefix} ... SKIP (duration {duration:.1f}s < {args.min_duration}s)")
            skipped_duration += 1
            continue

        # Run model
        try:
            frame_prob, onset_prob, art_prob = _run_model(model, audio_path, device)
        except Exception as e:
            print(f"{prefix} ... ERROR: {e}")
            errors += 1
            continue

        # Note detection at high threshold
        notes = pianoroll_to_notes(
            frame_prob,
            onset_prob,
            art_prob=art_prob,
            onset_threshold=args.threshold,
            sustain_threshold=args.threshold * 0.15,
        )

        if len(notes) < args.min_notes:
            print(f"{prefix} ... SKIP ({len(notes)} notes — below --min-notes {args.min_notes})")
            skipped_notes += 1
            continue

        # Write JAMS — use parent dir name to avoid collisions (e.g. MUSDB18 "other.wav")
        audio_duration = frame_prob.shape[0] * HOP_LENGTH / SAMPLE_RATE
        jams = _notes_to_jams(notes, audio_duration)
        stem = f"{audio_path.parent.name}_{audio_path.stem}".replace(" ", "_")
        (ann_dir / f"{stem}.jams").write_text(json.dumps(jams, indent=2))

        # Save mono WAV at model sample rate
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        sf.write(str(audio_out / f"{stem}_mic.wav"), y, SAMPLE_RATE)


        print(f"{prefix} ... OK ({len(notes)} notes)")
        ok += 1

    print(f"\n{'='*60}")
    print(f"Labeled  : {ok}")
    print(f"Skipped  : {skipped_notes} (too few notes)  {skipped_duration} (too short)")
    print(f"Errors   : {errors}")
    print(f"Output   : {args.output_dir}")

    if ok > 0:
        print(f"\nNext step — train with pseudo-labels mixed in:")
        print(
            f"  python -m model.train "
            f"--mixed --synth-root {args.output_dir} "
            f"--split trainval "
            f"--resume checkpoints/best_model_v2.pt "
            f"--lr 3e-4 --scheduler cosine --epochs 850"
        )


if __name__ == "__main__":
    main()
