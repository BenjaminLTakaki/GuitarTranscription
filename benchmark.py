#!/usr/bin/env python3
"""
Benchmark baseline vs hybrid on GuitarSet test split (player 05).

Metrics reported (frame-level, tablature classes):
  precision / recall / F1

Usage:
    python benchmark.py                          # baseline only
    python benchmark.py --backend basic_pitch    # baseline + basic-pitch hybrid
    python benchmark.py --backend yourmt3        # baseline + YourMT3 hybrid (slow)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import onnxruntime  # noqa: F401 — must precede jams to prevent basic_pitch caching ONNX_PRESENT=False
import jams
import numpy as np
import torch

from model.constants import (
    CHECKPOINT_DIR, GUITAR_TUNING, HOP_LENGTH,
    NUM_CLASSES, NUM_FRETS, SAMPLE_RATE,
    class_to_string_fret, string_fret_to_class,
)
from model.predict import load_cqt, predict
from model.pitch_gate import apply_pitch_gate

GUITARSET_DIR    = Path("GuitarSet")
DEFAULT_CHECKPOINT = Path(CHECKPOINT_DIR) / "best_model.pt"
FRAME_SEC        = HOP_LENGTH / SAMPLE_RATE


def jams_to_pianoroll(jam: jams.JAMS, n_frames: int) -> np.ndarray:
    """Convert GuitarSet JAMS note_midi annotations to (n_frames, NUM_CLASSES) binary."""
    roll = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)

    # GuitarSet: 6 note_midi annotations, one per string.
    # annotation_metadata.data_source is the string index (0 = low E).
    for ann in jam.annotations:
        if ann.namespace != "note_midi":
            continue
        try:
            string_idx = int(ann.annotation_metadata.data_source)
        except (TypeError, ValueError):
            continue
        if string_idx < 0 or string_idx >= 6:
            continue
        open_midi = GUITAR_TUNING[string_idx]
        for obs in ann.data:
            midi_pitch = int(round(float(obs.value)))
            fret       = midi_pitch - open_midi
            onset      = float(obs.time)
            dur        = float(obs.duration)
            if fret < 0 or fret >= NUM_FRETS:
                continue
            cls     = string_fret_to_class(string_idx, fret)
            f_start = max(0, int(onset / FRAME_SEC))
            f_end   = min(n_frames, int((onset + dur) / FRAME_SEC) + 1)
            roll[f_start:f_end, cls] = 1.0

    return roll


def frame_prf(pred: np.ndarray, target: np.ndarray, thr: float = 0.65) -> dict:
    pred_b   = pred   >= thr
    target_b = target >= 0.5
    tp = float((pred_b & target_b).sum())
    fp = float((pred_b & ~target_b).sum())
    fn = float((~pred_b & target_b).sum())
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {"p": p, "r": r, "f1": f1}


def run_benchmark(backend: str, fusion_mode: str, device: torch.device,
                  frame_threshold: float = 0.65,
                  checkpoint: Path = DEFAULT_CHECKPOINT) -> None:
    test_wavs = sorted(GUITARSET_DIR.glob("05_*_mic.wav"))
    if not test_wavs:
        raise FileNotFoundError(f"No test files found in {GUITARSET_DIR}")

    detector = None
    if backend != "none":
        from model.mt3_wrapper import load_detector
        detector = load_detector(backend)

    base_metrics   = []
    hybrid_metrics = []

    print(f"\nEvaluating {len(test_wavs)} test files  (player 05)")
    print(f"Backend: {backend}  |  Fusion: {fusion_mode}")
    print("=" * 60)

    for wav_path in test_wavs:
        jams_path = wav_path.with_suffix(".jams").with_name(
            wav_path.stem.replace("_mic", "") + ".jams"
        )
        if not jams_path.exists():
            print(f"  SKIP {wav_path.name} (no JAMS)")
            continue

        t0 = time.time()
        frame_prob, onset_prob, _ = predict(wav_path, checkpoint, device)
        n_frames = frame_prob.shape[0]

        jam    = jams.load(str(jams_path))
        target = jams_to_pianoroll(jam, n_frames)

        base_m = frame_prf(frame_prob, target, thr=frame_threshold)
        base_metrics.append(base_m)

        if detector is not None:
            pitch_probs    = detector.predict(wav_path, n_frames)
            hybrid_frame   = apply_pitch_gate(frame_prob, pitch_probs, mode=fusion_mode)
            hybrid_m       = frame_prf(hybrid_frame, target, thr=frame_threshold)
            hybrid_metrics.append(hybrid_m)
            elapsed = time.time() - t0
            print(
                f"  {wav_path.stem[:40]:40s}  "
                f"base F1={base_m['f1']:.3f}  hybrid F1={hybrid_m['f1']:.3f}  "
                f"({elapsed:.1f}s)"
            )
        else:
            elapsed = time.time() - t0
            print(
                f"  {wav_path.stem[:40]:40s}  "
                f"base F1={base_m['f1']:.3f}  ({elapsed:.1f}s)"
            )

    def avg(metrics, key):
        return np.mean([m[key] for m in metrics]) if metrics else 0.0

    print("\n" + "=" * 60)
    print(f"{'':42s}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}")
    print(f"  {'Baseline':40s}  {avg(base_metrics,'p'):9.4f}  "
          f"{avg(base_metrics,'r'):6.4f}  {avg(base_metrics,'f1'):6.4f}")
    if hybrid_metrics:
        df1 = avg(hybrid_metrics, 'f1') - avg(base_metrics, 'f1')
        print(f"  {'Hybrid (' + backend + ', ' + fusion_mode + ')':40s}  "
              f"{avg(hybrid_metrics,'p'):9.4f}  "
              f"{avg(hybrid_metrics,'r'):6.4f}  {avg(hybrid_metrics,'f1'):6.4f}  "
              f"(delta F1: {df1:+.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint (default: checkpoints/best_model.pt)")
    parser.add_argument("--backend", default="none",
                        choices=["none", "basic_pitch", "yourmt3", "mt3"])
    parser.add_argument("--fusion-mode", default="soft", choices=["soft", "rule"])
    parser.add_argument("--frame-threshold", type=float, default=0.65,
                        help="Classification threshold for frame-level F1 (default: 0.65)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    run_benchmark(args.backend, args.fusion_mode, device, args.frame_threshold, args.checkpoint)


if __name__ == "__main__":
    main()
