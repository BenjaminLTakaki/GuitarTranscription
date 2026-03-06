#!/usr/bin/env python3
"""Sweep Schmitt-trigger post-processing params on the GuitarSet test split.

Runs model inference once, caches predictions, then evaluates many parameter
combos at both frame-level and note-level. Prints a ranked table of the best
parameter configurations.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

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
)
from model.guitarset_dataset import GuitarSetDataset, jams_to_tab_events
from model.network import GuitarTranscriptionModel


# ---------------------------------------------------------------------------
# Note-level matching (onset + pitch)
# ---------------------------------------------------------------------------

ONSET_TOL_SEC = 0.050  # 50 ms tolerance for onset matching


def _class_events_to_midi_notes(
    events: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    """Convert (onset, offset, class_idx) → (onset, offset, midi_pitch)."""
    notes = []
    for onset, offset, cls in events:
        s, f = class_to_string_fret(cls)
        midi = GUITAR_TUNING[s] + f
        notes.append((onset, offset, midi))
    return notes


def note_level_metrics(
    pred_notes: list[dict],
    gt_notes: list[tuple[float, float, int]],
    onset_tol: float = ONSET_TOL_SEC,
) -> dict[str, float]:
    """Note-level P / R / F1 with onset + pitch matching.

    pred_notes: list of dicts with keys 'start', 'midi'
    gt_notes:   list of (onset_sec, offset_sec, midi_pitch)
    """
    # Build gt set
    gt_list = [(on, midi) for on, _off, midi in gt_notes]
    pred_list = [(n["start"], n["midi"]) for n in pred_notes]

    matched_gt = set()
    tp = 0
    for p_on, p_midi in pred_list:
        for i, (g_on, g_midi) in enumerate(gt_list):
            if i in matched_gt:
                continue
            if p_midi == g_midi and abs(p_on - g_on) <= onset_tol:
                tp += 1
                matched_gt.add(i)
                break

    prec = tp / (len(pred_list) + 1e-8)
    rec = tp / (len(gt_list) + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp,
            "pred_count": len(pred_list), "gt_count": len(gt_list)}


# ---------------------------------------------------------------------------
# Schmitt-trigger pianoroll_to_notes (local copy to avoid import issues)
# ---------------------------------------------------------------------------

def pianoroll_to_notes(
    frame_prob: np.ndarray,
    onset_prob: np.ndarray | None,
    onset_threshold: float,
    sustain_threshold: float,
    min_duration_frames: int,
    median_filter_size: tuple[int, int],
) -> list[dict]:
    """Schmitt-trigger note extraction — same logic as predict.py."""
    if median_filter_size[0] > 1 or median_filter_size[1] > 1:
        filtered = median_filter(frame_prob, size=median_filter_size)
    else:
        filtered = frame_prob
    T, P = filtered.shape
    frame_sec = HOP_LENGTH / SAMPLE_RATE
    notes = []
    onset_reattack_th = onset_threshold * 0.8

    for p in range(P):
        s, f = class_to_string_fret(p)
        midi_note = GUITAR_TUNING[s] + f
        in_note = False
        start = 0

        for t in range(T):
            prob = filtered[t, p]
            if in_note and onset_prob is not None:
                is_reattack = (
                    onset_prob[t, p] >= onset_reattack_th
                    and (t == 0 or onset_prob[t, p] > onset_prob[t - 1, p] * 1.2)
                )
                if is_reattack and (t - start) >= min_duration_frames:
                    notes.append({
                        "midi": midi_note, "string": s, "fret": f,
                        "start": start * frame_sec,
                        "end": t * frame_sec,
                    })
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
                        notes.append({
                            "midi": midi_note, "string": s, "fret": f,
                            "start": start * frame_sec,
                            "end": t * frame_sec,
                        })

        if in_note and (T - start) >= min_duration_frames:
            notes.append({
                "midi": midi_note, "string": s, "fret": f,
                "start": start * frame_sec,
                "end": T * frame_sec,
            })

    return notes


# ---------------------------------------------------------------------------
# Deduplicate same MIDI pitch sounding on multiple strings at same time
# ---------------------------------------------------------------------------

def dedup_notes(notes: list[dict], merge_tol: float = 0.03) -> list[dict]:
    """If the same MIDI pitch has overlapping notes, keep the stronger one."""
    from collections import defaultdict
    by_midi = defaultdict(list)
    for n in notes:
        by_midi[n["midi"]].append(n)

    deduped = []
    for midi, group in by_midi.items():
        group.sort(key=lambda n: n["start"])
        merged = [group[0]]
        for n in group[1:]:
            prev = merged[-1]
            # overlapping = note starts before previous ends
            if n["start"] < prev["end"] + merge_tol:
                # extend previous if this one is longer
                prev["end"] = max(prev["end"], n["end"])
            else:
                merged.append(n)
        deduped.extend(merged)
    deduped.sort(key=lambda n: n["start"])
    return deduped


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    print("Loading model...")
    model = GuitarTranscriptionModel().to(device)
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = GuitarSetDataset(root="GuitarSet", split="test", augment=False)
    print(f"Test items: {len(test_ds)}")

    # --- Phase 1: run inference once, cache predictions ---
    cached = []
    print("Running inference on test set...")
    with torch.no_grad():
        for i in range(len(test_ds)):
            mel, frame_t, onset_t = test_ds[i]
            logits, onset_logits = model(mel.unsqueeze(0).to(device))
            frame_prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            onset_prob = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()

            # Ground truth notes from JAMS
            jams_path = test_ds.items[i]["jams_path"]
            gt_events = jams_to_tab_events(jams_path)
            gt_notes = _class_events_to_midi_notes(gt_events)

            cached.append({
                "frame_prob": frame_prob,
                "onset_prob": onset_prob,
                "gt_notes": gt_notes,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(test_ds)}")

    print(f"Inference done. Cached {len(cached)} tracks.\n")

    # --- Phase 2: sweep post-processing parameters ---
    onset_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    sustain_thresholds = [0.10, 0.15, 0.20, 0.30]
    median_sizes = [(1, 1), (3, 3), (5, 5)]
    min_note_frames_list = [2, 3]
    dedup_options = [False, True]

    combos = list(itertools.product(
        onset_thresholds, sustain_thresholds, median_sizes,
        min_note_frames_list, dedup_options
    ))
    print(f"Sweeping {len(combos)} parameter combinations...\n")

    results = []
    for ci, (onset_th, sustain_th, med_sz, min_nf, do_dedup) in enumerate(combos):
        if sustain_th >= onset_th:
            continue  # sustain must be lower than onset

        all_prec, all_rec, all_f1 = [], [], []
        total_tp, total_pred, total_gt = 0, 0, 0

        for c in cached:
            pred_notes = pianoroll_to_notes(
                c["frame_prob"], c["onset_prob"],
                onset_threshold=onset_th,
                sustain_threshold=sustain_th,
                min_duration_frames=min_nf,
                median_filter_size=med_sz,
            )
            if do_dedup:
                pred_notes = dedup_notes(pred_notes)

            m = note_level_metrics(pred_notes, c["gt_notes"])
            total_tp += m["tp"]
            total_pred += m["pred_count"]
            total_gt += m["gt_count"]

        # Micro-averaged metrics
        prec = total_tp / (total_pred + 1e-8)
        rec = total_tp / (total_gt + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        results.append({
            "onset_th": onset_th,
            "sustain_th": sustain_th,
            "median": med_sz,
            "min_frames": min_nf,
            "dedup": do_dedup,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": total_tp,
            "pred": total_pred,
            "gt": total_gt,
        })

        if (ci + 1) % 50 == 0:
            print(f"  {ci+1}/{len(combos)} combos evaluated")

    # --- Phase 3: rank and display top results ---
    results.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 100)
    print(f"{'Rank':>4}  {'Onset':>6}  {'Sustain':>7}  {'Median':>8}  {'MinF':>4}  "
          f"{'Dedup':>5}  {'P':>7}  {'R':>7}  {'F1':>7}  "
          f"{'TP':>5}  {'Pred':>5}  {'GT':>5}")
    print("=" * 100)
    for rank, r in enumerate(results[:30], 1):
        print(f"{rank:4d}  {r['onset_th']:6.2f}  {r['sustain_th']:7.2f}  "
              f"{str(r['median']):>8}  {r['min_frames']:4d}  "
              f"{str(r['dedup']):>5}  {r['precision']:7.4f}  {r['recall']:7.4f}  "
              f"{r['f1']:7.4f}  {r['tp']:5d}  {r['pred']:5d}  {r['gt']:5d}")

    print("\n--- Best configuration ---")
    best = results[0]
    for k, v in best.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
