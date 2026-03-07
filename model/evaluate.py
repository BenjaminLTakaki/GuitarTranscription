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


# ---------------------------------------------------------------------------
# Articulation-level evaluation (e.g. hammer-on / pull-off detection)
# ---------------------------------------------------------------------------

def articulation_metrics(
    pred_events: list[dict],
    target_events: list[dict],
    tolerance_frames: int = 2,
    hop_sec: float = 0.01,
) -> dict[str, float]:
    """Evaluate how accurately the model classifies note articulations.

    Both *pred_events* and *target_events* are lists of dicts that must
    contain at least the keys ``midi``, ``string``, ``onset`` (seconds),
    and ``articulation`` (e.g. ``"hammer_on"`` or ``"pluck"``).

    Parameters
    ----------
    pred_events : predicted note events with articulation labels.
    target_events : ground-truth note events with articulation labels.
    tolerance_frames : how many frames of onset‐time tolerance to allow.
    hop_sec : duration of one frame in seconds (default 10 ms).

    Returns
    -------
    dict with hammer_on_precision, hammer_on_recall, hammer_on_f1.
    """
    target_hammers = [
        e for e in target_events if e.get("articulation") == "hammer_on"
    ]
    pred_hammers = [
        e for e in pred_events if e.get("articulation") == "hammer_on"
    ]

    tp_hammer = 0
    matched_targets: set[int] = set()

    for p_event in pred_hammers:
        for i, t_event in enumerate(target_hammers):
            if i in matched_targets:
                continue
            time_diff = abs(p_event["onset"] - t_event["onset"])
            if (
                p_event["midi"] == t_event["midi"]
                and p_event["string"] == t_event["string"]
                and time_diff <= tolerance_frames * hop_sec
            ):
                tp_hammer += 1
                matched_targets.add(i)
                break

    fp_hammer = len(pred_hammers) - tp_hammer
    fn_hammer = len(target_hammers) - len(matched_targets)

    precision = tp_hammer / (tp_hammer + fp_hammer + 1e-8)
    recall = tp_hammer / (tp_hammer + fn_hammer + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "hammer_on_precision": float(precision),
        "hammer_on_recall": float(recall),
        "hammer_on_f1": float(f1),
    }
