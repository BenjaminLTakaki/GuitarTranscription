"""Phase 2B — DTW alignment engine.

Aligns real (separated) guitar audio to synthesized tab audio using
Dynamic Time Warping on CQT spectrograms. The warp path is then used
to remap MIDI event timestamps from synth time to real audio time.
"""

from __future__ import annotations

import librosa
import numpy as np

from model.constants import (
    CQT_BINS_PER_OCTAVE,
    CQT_FMIN,
    CQT_N_BINS,
    HOP_LENGTH,
    SAMPLE_RATE,
)


def compute_cqt(audio_path: str | None = None, y: np.ndarray | None = None) -> np.ndarray:
    """Compute log-amplitude CQT spectrogram.

    Provide either audio_path or pre-loaded audio y. Returns (n_bins, T).
    """
    if y is None:
        y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    cqt = np.abs(librosa.cqt(
        y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=CQT_FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCTAVE,
    ))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    # Normalize to [0, 1]
    log_cqt = (log_cqt - log_cqt.min()) / (log_cqt.max() - log_cqt.min() + 1e-8)
    return log_cqt.astype(np.float32)


def align_audio(
    real_audio_path: str,
    synth_audio_path: str,
) -> tuple[np.ndarray, float]:
    """Align real audio to synthesized tab audio using DTW.

    Returns:
        warp_path: (N, 2) array of [real_frame, synth_frame] pairs
        cost: normalized total DTW cost (lower = better alignment)
    """
    cqt_real = _prepare_features(compute_cqt(audio_path=real_audio_path))   # (n_bins, T_real)
    cqt_synth = _prepare_features(compute_cqt(audio_path=synth_audio_path))  # (n_bins, T_synth)

    if cqt_real.shape[1] == 0 or cqt_synth.shape[1] == 0:
        raise ValueError("Empty CQT features for DTW alignment")

    # DTW on features (n_bins, T). Use Euclidean on unit-normalized columns
    # to avoid cosine NaNs from near-zero vectors.
    D, wp = librosa.sequence.dtw(
        X=cqt_real,
        Y=cqt_synth,
        metric="euclidean",
    )

    # wp is (N, 2) with [real_frame, synth_frame], reverse-ordered
    wp = wp[::-1]  # sort ascending by time

    # Normalized cost: total cost / path length
    total_cost = D[wp[-1, 0], wp[-1, 1]]
    normalized_cost = total_cost / len(wp)

    return wp, normalized_cost


def _prepare_features(cqt: np.ndarray) -> np.ndarray:
    """Sanitize and L2-normalize CQT columns for stable DTW distance."""
    x = np.nan_to_num(cqt.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(x, axis=0, keepdims=True)
    x = x / np.maximum(norms, 1e-8)
    return x


def warp_events(
    events: list[dict],
    warp_path: np.ndarray,
) -> list[dict]:
    """Remap event timestamps from synth time to real audio time.

    Uses the DTW warp path to build a synth_frame → real_frame mapping,
    then interpolates to convert each event's onset/offset.

    Args:
        events: list of dicts with 'onset' and 'offset' keys (in seconds)
        warp_path: (N, 2) array of [real_frame, synth_frame] pairs

    Returns:
        Warped events with updated onset/offset times.
    """
    frame_dur = HOP_LENGTH / SAMPLE_RATE

    # Build synth_frame → real_time lookup
    synth_frames = warp_path[:, 1].astype(float)
    real_frames = warp_path[:, 0].astype(float)

    # Convert frames to seconds
    synth_times = synth_frames * frame_dur
    real_times = real_frames * frame_dur

    warped: list[dict] = []
    for ev in events:
        ev = dict(ev)  # copy
        onset_synth = ev["onset"]
        offset_synth = ev["offset"]

        # Interpolate synth time → real time
        onset_real = float(np.interp(onset_synth, synth_times, real_times))
        offset_real = float(np.interp(offset_synth, synth_times, real_times))

        # Ensure minimum duration
        if offset_real - onset_real < 0.05:
            offset_real = onset_real + 0.05

        ev["onset"] = onset_real
        ev["offset"] = offset_real
        warped.append(ev)

    return warped
