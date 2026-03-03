"""Evaluate transcription quality: precision, recall, F1 at frame level."""

from __future__ import annotations

import numpy as np


def frame_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute frame-level precision / recall / F1.

    Parameters
    ----------
    pred   : (T, P) float  — sigmoid outputs
    target : (T, P) float  — binary ground truth
    threshold : binarisation threshold for predictions

    Returns
    -------
    dict with keys: precision, recall, f1
    """
    pred_bin = (pred >= threshold).astype(np.float32)
    target_bin = (target >= 0.5).astype(np.float32)

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def note_level_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    onset_tolerance_frames: int = 2,
) -> dict[str, float]:
    """Simple note-level P/R/F1 using onset matching.

    A predicted onset is a true positive if a target onset of the same pitch
    exists within ``onset_tolerance_frames`` frames.
    """
    pred_onsets = _extract_onsets(pred, threshold)
    target_onsets = _extract_onsets(target, 0.5)

    tp = 0
    matched_target: set[tuple[int, int]] = set()
    for frame, pitch in pred_onsets:
        for df in range(-onset_tolerance_frames, onset_tolerance_frames + 1):
            key = (frame + df, pitch)
            if key in target_onsets and key not in matched_target:
                tp += 1
                matched_target.add(key)
                break

    precision = tp / (len(pred_onsets) + 1e-8)
    recall = tp / (len(target_onsets) + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "note_precision": float(precision),
        "note_recall": float(recall),
        "note_f1": float(f1),
    }


def _extract_onsets(
    roll: np.ndarray, threshold: float
) -> set[tuple[int, int]]:
    """Return set of (frame, pitch) onset positions from a piano-roll."""
    binary = (roll >= threshold).astype(np.int8)
    onsets: set[tuple[int, int]] = set()
    T, P = binary.shape
    for p in range(P):
        for t in range(T):
            if binary[t, p] == 1 and (t == 0 or binary[t - 1, p] == 0):
                onsets.add((t, p))
    return onsets
