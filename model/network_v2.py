"""CNN + Transformer network for guitar transcription (v2).

Replaces the BiGRU temporal model with a multi-head self-attention
Transformer encoder.  The CNN front-end is identical to v1, so the
spectrogram feature extraction is preserved — only the sequence
modelling changes.

This allows the model to evaluate the entire spectrogram contextually,
distinguishing newly plucked strings from lingering sympathetic
harmonics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.constants import N_MELS, NUM_CLASSES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ConvBlock(nn.Module):
    """Same ConvBlock as network.py."""

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


class GuitarTranscriptionModelV2(nn.Module):
    """CNN + Transformer encoder for guitar transcription.

    Same CNN front-end as v1, but replaces BiGRU with multi-head
    self-attention.  This allows the model to evaluate the entire
    spectrogram contextually, distinguishing newly plucked strings
    from lingering sympathetic harmonics.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        num_classes: int = NUM_CLASSES,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        d_model: int = 512,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.num_classes = num_classes

        # CNN encoder (same as v1)
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in cnn_channels:
            layers.append(ConvBlock(in_ch, out_ch, pool=(1, 2)))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        cnn_freq_out = n_mels // (2 ** len(cnn_channels))
        cnn_feat_dim = cnn_channels[-1] * cnn_freq_out

        # Project CNN features to transformer dimension
        self.input_proj = nn.Linear(cnn_feat_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Output heads
        self.onset_head = nn.Linear(d_model, num_classes)
        self.frame_head = nn.Linear(d_model, num_classes)
        self.articulation_head = nn.Linear(d_model, num_classes)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, mel: torch.Tensor):
        # mel: (B, n_bins, T)
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = x.permute(0, 1, 3, 2)  # (B, 1, T, n_mels)
        x = self.cnn(x)  # (B, C, T, F)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)

        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, d_model)
        x = self.dropout_layer(x)

        onset_logits = self.onset_head(x)
        frame_logits = self.frame_head(x)
        art_logits = self.articulation_head(x)

        return frame_logits, onset_logits, art_logits
