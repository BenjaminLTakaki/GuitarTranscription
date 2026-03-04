"""CNN + BiGRU network for frame-level guitar transcription."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.constants import N_MELS, NUM_PITCHES


class ConvBlock(nn.Module):
    """Two conv layers with batch-norm, ReLU, and optional pooling."""

    def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (1, 2)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
            nn.Dropout(0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GuitarTranscriptionModel(nn.Module):
    """
    Mel spectrogram → frame-level piano-roll (multi-pitch) prediction.

    Architecture
    ============
    1. **Acoustic encoder** — three ``ConvBlock`` layers that progressively
       shrink the frequency axis while preserving temporal resolution.
    2. **Recurrent** — one bidirectional GRU layer that captures temporal
       context across frames.
    3. **Heads** — two independent linear projections:
       * *frame* head → sigmoid → which pitches are active each frame
       * *onset* head → sigmoid → where note onsets occur

    Input
    -----
    mel : (B, n_mels, T)   log-mel spectrogram

    Output
    ------
    frame_logits : (B, T, NUM_PITCHES)
    onset_logits : (B, T, NUM_PITCHES)
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        num_pitches: int = NUM_PITCHES,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.num_pitches = num_pitches

        # CNN encoder — pool only on frequency axis (pool=(1,2))
        layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.append(ConvBlock(in_ch, out_ch, pool=(1, 2)))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # After 3 pools of (1,2) on freq axis: n_mels // 8
        cnn_freq_out = n_mels // (2 ** len(cnn_channels))
        cnn_feat_dim = cnn_channels[-1] * cnn_freq_out

        # BiGRU
        self.rnn = nn.GRU(
            input_size=cnn_feat_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        rnn_out_dim = rnn_hidden * 2  # bidirectional

        # Output heads — deeper than single Linear
        head_hidden = rnn_out_dim // 2
        self.onset_head = nn.Sequential(
            nn.Linear(rnn_out_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_pitches),
        )
        # Frame head takes RNN output + onset probabilities (Onsets & Frames)
        self.frame_head = nn.Sequential(
            nn.Linear(rnn_out_dim + num_pitches, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_pitches),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, mel: torch.Tensor):
        """
        Parameters
        ----------
        mel : (B, n_mels, T)

        Returns
        -------
        frame_logits : (B, T, num_pitches)
        onset_logits : (B, T, num_pitches)
        """
        # Add channel dim → (B, 1, n_mels, T)  — treat freq as height, time as width
        # We want to pool freq but keep time, so transpose to (B, 1, T, n_mels)
        x = mel.unsqueeze(1)                         # (B, 1, n_mels, T)
        x = x.permute(0, 1, 3, 2)                    # (B, 1, T, n_mels)

        x = self.cnn(x)                              # (B, C, T, n_mels//8)

        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T, C*F)

        x = self.dropout(x)
        x, _ = self.rnn(x)                           # (B, T, rnn_hidden*2)
        x = self.dropout(x)

        # Onset head first — its output feeds into the frame head
        onset_logits = self.onset_head(x)            # (B, T, num_pitches)
        onset_probs = torch.sigmoid(onset_logits)

        # Frame head receives RNN output + onset probabilities
        frame_input = torch.cat([x, onset_probs], dim=-1)  # (B, T, rnn_out+P)
        frame_logits = self.frame_head(frame_input)  # (B, T, num_pitches)

        return frame_logits, onset_logits
