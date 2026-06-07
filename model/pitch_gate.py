"""
Pitch gate: fuses an external pitch detector's output with the model's
126-dim string/fret predictions to produce a better final transcription.

Core idea
---------
The external detector (MT3, basic-pitch) answers "what pitches are playing?"
Our model answers "which string/fret combination?"  Both questions are needed
for guitar tablature, and combining them suppresses each model's blind spots.

Fusion modes
------------
rule   Hard mask: silence class slots whose pitch the detector didn't detect.
       Fast, zero hyper-parameters, good when the detector is very reliable.

soft   Geometric mean of model confidence and expanded pitch confidence.
       High only when BOTH are confident -> fewer false positives.
       Works well without any extra training.

learn  Trainable 2-layer MLP (~2 K params) on top of both outputs.
       Best quality after fine-tuning with train_fusion.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from model.constants import (
    GUITAR_TUNING,
    MIDI_MIN,
    NUM_CLASSES,
    NUM_FRETS,
    NUM_PITCHES,
    NUM_STRINGS,
)


# ---------------------------------------------------------------------------
# Pitch <-> class lookup table (built once at import time)
# ---------------------------------------------------------------------------

def _build_expand_matrix() -> np.ndarray:
    """
    (NUM_PITCHES, NUM_CLASSES) sparse binary matrix.
    expand[p, c] = 1 iff class c produces MIDI pitch (MIDI_MIN + p).
    """
    mat = np.zeros((NUM_PITCHES, NUM_CLASSES), dtype=np.float32)
    for string_idx in range(NUM_STRINGS):
        for fret in range(NUM_FRETS):
            midi      = GUITAR_TUNING[string_idx] + fret
            pitch_idx = midi - MIDI_MIN
            class_idx = string_idx * NUM_FRETS + fret
            if 0 <= pitch_idx < NUM_PITCHES:
                mat[pitch_idx, class_idx] = 1.0
    return mat


_EXPAND = _build_expand_matrix()          # (NUM_PITCHES, NUM_CLASSES)
_EXPAND_T = torch.from_numpy(_EXPAND)     # same, as a tensor (lazy-moved to device)


# ---------------------------------------------------------------------------
# NumPy path (used during inference)
# ---------------------------------------------------------------------------

def expand_pitch_to_classes(pitch_probs: np.ndarray) -> np.ndarray:
    """
    Broadcast pitch detector output into the 126-dim class space.

    Parameters
    ----------
    pitch_probs : (T, NUM_PITCHES) -- confidence per MIDI pitch per frame

    Returns
    -------
    class_probs : (T, NUM_CLASSES) -- confidence per (string, fret) slot per frame
                  Each slot gets the confidence of the pitch it produces.
    """
    return pitch_probs @ _EXPAND   # (T, NUM_PITCHES) @ (NUM_PITCHES, NUM_CLASSES)


def apply_pitch_gate(
    model_probs: np.ndarray,
    pitch_probs: np.ndarray,
    mode: str = "soft",
    rule_threshold: float = 0.25,
) -> np.ndarray:
    """
    Fuse model frame probabilities with external pitch detector output.

    Parameters
    ----------
    model_probs     : (T, NUM_CLASSES)  sigmoid output from our CNN+GRU
    pitch_probs     : (T, NUM_PITCHES)  from BasicPitchDetector / MT3Detector
    mode            : 'rule' or 'soft'
    rule_threshold  : in 'rule' mode, pitch slots below this are masked to 0

    Returns
    -------
    fused : (T, NUM_CLASSES) float32
    """
    expanded = expand_pitch_to_classes(pitch_probs)   # (T, NUM_CLASSES)

    if mode == "rule":
        # Zero out classes whose pitch the detector doesn't confirm.
        gate = (expanded >= rule_threshold).astype(np.float32)
        return (model_probs * gate).astype(np.float32)

    if mode == "soft":
        # Additive boost: raise confidence where both sources agree.
        # Detector adds up to +0.15 at full pitch confidence; never suppresses.
        # This helps borderline cases clear the onset threshold without hurting recall.
        BOOST = 0.15
        return np.clip(model_probs + BOOST * expanded, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unknown mode {mode!r}. Choose 'rule' or 'soft'.")


# ---------------------------------------------------------------------------
# Torch path (used during training of the learned fusion layer)
# ---------------------------------------------------------------------------

def expand_pitch_to_classes_torch(
    pitch_probs: torch.Tensor,   # (B, T, NUM_PITCHES)
    device: torch.device | None = None,
) -> torch.Tensor:
    """Batch version of expand_pitch_to_classes for use in training."""
    expand = _EXPAND_T.to(pitch_probs.device if device is None else device)
    return torch.bmm(pitch_probs, expand.unsqueeze(0).expand(pitch_probs.size(0), -1, -1))


class LearnedFusion(nn.Module):
    """
    Tiny MLP that learns the optimal combination of model and pitch-detector
    confidence for each class slot.

    Input  : [model_prob, expanded_pitch_prob] per (batch, frame, class)
    Output : fused logits  (B, T, NUM_CLASSES)

    Only ~2 K parameters -- trains in minutes on GuitarSet.
    See train_fusion.py for the fine-tuning script.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        model_logits: torch.Tensor,     # (B, T, NUM_CLASSES)
        pitch_expanded: torch.Tensor,   # (B, T, NUM_CLASSES) -- already in class space
    ) -> torch.Tensor:
        """Returns fused logits (B, T, NUM_CLASSES)."""
        model_p = torch.sigmoid(model_logits)              # (B, T, C)
        x = torch.stack([model_p, pitch_expanded], dim=-1) # (B, T, C, 2)
        B, T, C, _ = x.shape
        out = self.net(x.reshape(B * T * C, 2))            # (B*T*C, 1)
        return out.reshape(B, T, C)                        # (B, T, C)
