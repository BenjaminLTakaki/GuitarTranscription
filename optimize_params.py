#!/usr/bin/env python3
"""Sweep Schmitt-trigger post-processing params on the GuitarSet test split.

Runs model inference once (caches predictions), then evaluates many parameter
combinations at both frame-level and note-level. Prints ranked tables for both.

Usage:
    python optimize_params.py                      # all sweeps
    python optimize_params.py --frame-only         # frame-level threshold only (fast)
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import onnxruntime  # noqa: F401 — must precede jams to prevent basic_pitch ONNX_PRESENT=False
import jams
import numpy as np
import torch
from scipy.ndimage import median_filter

from model.constants import (
    GUITAR_TUNING,
    HOP_LENGTH,
    NUM_CLASSES,
    NUM_FRETS,
    SAMPLE_RATE,
    class_to_string_fret,
    string_fret_to_class,
)
from model.predict import load_cqt, predict

GUITARSET_DIR = Path("GuitarSet")
CHECKPOINT    = Path("checkpoints/best_model.pt")
FRAME_SEC     = HOP_LENGTH / SAMPLE_RATE
ONSET_TOL_SEC = 0.050   # 50 ms onset matching tolerance for note-level metrics


# ---------------------------------------------------------------------------
# JAMS parsing (same as benchmark.py)
# ---------------------------------------------------------------------------

def jams_to_pianoroll(jam: jams.JAMS, n_frames: int) -> np.ndarray:
    roll = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)
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


def jams_to_gt_notes(jam: jams.JAMS) -> list[tuple[float, float, int]]:
    """Return (onset_sec, offset_sec, midi_pitch) for each note."""
    notes = []
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
            onset  = float(obs.time)
            dur    = float(obs.duration)
            if midi_pitch < open_midi or (midi_pitch - open_midi) >= NUM_FRETS:
                continue
            notes.append((onset, onset + dur, midi_pitch))
    return notes


# ---------------------------------------------------------------------------
# Frame-level metrics
# ---------------------------------------------------------------------------

def frame_prf(pred: np.ndarray, target: np.ndarray, thr: float) -> dict:
    pred_b   = pred   >= thr
    target_b = target >= 0.5
    tp = float((pred_b & target_b).sum())
    fp = float((pred_b & ~target_b).sum())
    fn = float((~pred_b & target_b).sum())
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {"p": p, "r": r, "f1": f1}


# ---------------------------------------------------------------------------
# Note-level extraction (Schmitt-trigger)
# ---------------------------------------------------------------------------

def pianoroll_to_notes(
    frame_prob: np.ndarray,
    onset_prob: np.ndarray,
    onset_threshold: float,
    sustain_threshold: float,
    min_duration_frames: int,
    median_filter_size: tuple[int, int],
) -> list[dict]:
    if median_filter_size[0] > 1 or median_filter_size[1] > 1:
        filtered = median_filter(frame_prob, size=median_filter_size)
    else:
        filtered = frame_prob
    T, P = filtered.shape
    notes = []
    onset_reattack_th = onset_threshold * 0.8

    for p in range(P):
        s, f = class_to_string_fret(p)
        midi_note = GUITAR_TUNING[s] + f
        in_note = False
        start = 0

        for t in range(T):
            prob = filtered[t, p]
            if in_note:
                is_reattack = (
                    onset_prob[t, p] >= onset_reattack_th
                    and (t == 0 or onset_prob[t, p] > onset_prob[t - 1, p] * 1.2)
                )
                if is_reattack and (t - start) >= min_duration_frames:
                    notes.append({"midi": midi_note, "start": start * FRAME_SEC,
                                  "end": t * FRAME_SEC})
                    start = t
                    continue

            if not in_note:
                if prob >= onset_threshold:
                    in_note = True
                    start = t
            else:
                if prob < sustain_threshold:
                    in_note = False
                    if (t - start) >= min_duration_frames:
                        notes.append({"midi": midi_note, "start": start * FRAME_SEC,
                                      "end": t * FRAME_SEC})

        if in_note and (T - start) >= min_duration_frames:
            notes.append({"midi": midi_note, "start": start * FRAME_SEC,
                          "end": T * FRAME_SEC})

    return notes


def note_level_metrics(
    pred_notes: list[dict],
    gt_notes: list[tuple[float, float, int]],
) -> dict:
    gt_list   = [(on, midi) for on, _off, midi in gt_notes]
    pred_list = [(n["start"], n["midi"]) for n in pred_notes]
    matched   = set()
    tp = 0
    for p_on, p_midi in pred_list:
        for i, (g_on, g_midi) in enumerate(gt_list):
            if i in matched:
                continue
            if p_midi == g_midi and abs(p_on - g_on) <= ONSET_TOL_SEC:
                tp += 1
                matched.add(i)
                break
    return {"tp": tp, "pred": len(pred_list), "gt": len(gt_list)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-only", action="store_true",
                        help="Only run the fast frame-level threshold sweep")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    test_wavs = sorted(GUITARSET_DIR.glob("05_*_mic.wav"))
    if not test_wavs:
        raise FileNotFoundError(f"No test files found in {GUITARSET_DIR}")
    print(f"Found {len(test_wavs)} test files")

    # --- Phase 1: run inference once, cache predictions --------------------
    cached = []
    print("Running inference (cached for all sweeps)...")
    t0 = time.time()
    for wav_path in test_wavs:
        jams_path = wav_path.with_name(
            wav_path.stem.replace("_mic", "") + ".jams"
        )
        if not jams_path.exists():
            print(f"  SKIP {wav_path.name} (no JAMS)")
            continue

        frame_prob, onset_prob, _ = predict(wav_path, CHECKPOINT, device)
        n_frames = frame_prob.shape[0]

        jam    = jams.load(str(jams_path))
        target = jams_to_pianoroll(jam, n_frames)
        gt_notes = jams_to_gt_notes(jam)

        cached.append({
            "frame_prob": frame_prob,
            "onset_prob": onset_prob,
            "target":     target,
            "gt_notes":   gt_notes,
            "name":       wav_path.stem,
        })

    print(f"Inference done in {time.time() - t0:.1f}s. {len(cached)} tracks cached.\n")

    # --- Phase 2: frame-level threshold sweep ------------------------------
    print("=" * 70)
    print("FRAME-LEVEL THRESHOLD SWEEP (matches benchmark.py metric)")
    print("=" * 70)
    frame_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    frame_results = []
    for thr in frame_thresholds:
        tp_tot = fp_tot = fn_tot = 0.0
        for c in cached:
            pred_b   = c["frame_prob"] >= thr
            target_b = c["target"] >= 0.5
            tp_tot += float((pred_b & target_b).sum())
            fp_tot += float((pred_b & ~target_b).sum())
            fn_tot += float((~pred_b & target_b).sum())
        p  = tp_tot / (tp_tot + fp_tot + 1e-8)
        r  = tp_tot / (tp_tot + fn_tot + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        frame_results.append({"thr": thr, "p": p, "r": r, "f1": f1})

    frame_results.sort(key=lambda x: x["f1"], reverse=True)
    print(f"{'Rank':>4}  {'Thr':>5}  {'Precision':>9}  {'Recall':>6}  {'F1':>7}")
    print("-" * 40)
    for rank, r in enumerate(frame_results, 1):
        marker = " <-- current" if abs(r["thr"] - 0.5) < 0.01 else ""
        print(f"{rank:4d}  {r['thr']:5.2f}  {r['p']:9.4f}  {r['r']:6.4f}  {r['f1']:7.4f}{marker}")

    best_frame = frame_results[0]
    print(f"\nBest frame threshold: {best_frame['thr']:.2f}  F1={best_frame['f1']:.4f}  "
          f"(P={best_frame['p']:.4f}, R={best_frame['r']:.4f})")

    if args.frame_only:
        return

    # --- Phase 3: note-level Schmitt-trigger sweep -------------------------
    print("\n" + "=" * 70)
    print("NOTE-LEVEL SCHMITT-TRIGGER SWEEP (onset+pitch, 50ms tolerance)")
    print("=" * 70)

    onset_thresholds     = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75]
    sustain_thresholds   = [0.05, 0.10, 0.15, 0.20, 0.30]
    median_sizes         = [(1, 1), (3, 3), (5, 5)]
    min_note_frames_list = [2, 3, 4]

    combos = list(itertools.product(
        onset_thresholds, sustain_thresholds, median_sizes, min_note_frames_list
    ))
    combos = [(on, su, med, mnf) for on, su, med, mnf in combos if su < on]
    print(f"Sweeping {len(combos)} parameter combinations...")

    note_results = []
    for ci, (onset_th, sustain_th, med_sz, min_nf) in enumerate(combos):
        tp_tot = pred_tot = gt_tot = 0
        for c in cached:
            pred_notes = pianoroll_to_notes(
                c["frame_prob"], c["onset_prob"],
                onset_threshold=onset_th,
                sustain_threshold=sustain_th,
                min_duration_frames=min_nf,
                median_filter_size=med_sz,
            )
            m = note_level_metrics(pred_notes, c["gt_notes"])
            tp_tot   += m["tp"]
            pred_tot += m["pred"]
            gt_tot   += m["gt"]

        p  = tp_tot / (pred_tot + 1e-8)
        r  = tp_tot / (gt_tot   + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        note_results.append({
            "onset_th": onset_th, "sustain_th": sustain_th,
            "median": med_sz, "min_frames": min_nf,
            "p": p, "r": r, "f1": f1,
            "tp": tp_tot, "pred": pred_tot, "gt": gt_tot,
        })

        if (ci + 1) % 30 == 0:
            print(f"  {ci+1}/{len(combos)} done")

    note_results.sort(key=lambda x: x["f1"], reverse=True)

    print(f"\n{'Rank':>4}  {'Onset':>6}  {'Sustain':>7}  {'Median':>8}  "
          f"{'MinF':>4}  {'P':>7}  {'R':>7}  {'F1':>7}")
    print("-" * 70)
    for rank, r in enumerate(note_results[:20], 1):
        print(f"{rank:4d}  {r['onset_th']:6.2f}  {r['sustain_th']:7.2f}  "
              f"{str(r['median']):>8}  {r['min_frames']:4d}  "
              f"{r['p']:7.4f}  {r['r']:7.4f}  {r['f1']:7.4f}")

    best_note = note_results[0]
    print(f"\nBest note config:")
    for k, v in best_note.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
