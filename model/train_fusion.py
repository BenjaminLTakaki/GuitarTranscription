#!/usr/bin/env python3
"""
Fine-tune the LearnedFusion layer on top of a frozen guitar transcription model.

What this does
--------------
1. Loads a trained model checkpoint (weights are frozen).
2. Runs basic-pitch on every GuitarSet training file to get pitch predictions.
3. Trains the 2-layer LearnedFusion MLP (~2 K params) using the ground truth
   frame labels -- effectively teaching it when to trust the model vs the
   pitch detector.
4. Saves the fusion weights alongside the original checkpoint.

Usage
-----
    python -m model.train_fusion \\
        --checkpoint checkpoints/best_model.pt \\
        --root GuitarSet \\
        --epochs 30

The saved file (default: checkpoints/fusion.pt) is a dict:
    { 'fusion_state_dict': ..., 'base_checkpoint': <original .pt path> }

To use during inference:
    python -m model.predict audio.wav --pitch-backend basic_pitch --fusion-mode learn \\
        --fusion-weights checkpoints/fusion.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    GUITARSET_DIR,
    HOP_LENGTH,
    NUM_CLASSES,
    NUM_PITCHES,
    SAMPLE_RATE,
)
from model.guitarset_dataset import GuitarSetDataset
from model.pitch_gate import LearnedFusion, expand_pitch_to_classes_torch


# ---------------------------------------------------------------------------
# Dataset wrapper: augments GuitarSet items with basic-pitch predictions
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """
    Wraps GuitarSetDataset and precomputes basic-pitch predictions for each
    audio clip so we don't re-run the detector during training.

    Parameters
    ----------
    base_dataset    : a GuitarSetDataset instance
    pitch_cache_dir : directory to cache precomputed basic-pitch arrays
                      (avoids re-running the detector on every epoch)
    """

    def __init__(self, base_dataset: GuitarSetDataset, pitch_cache_dir: Path):
        self.base    = base_dataset
        self.cache   = pitch_cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)
        self._detector = None   # lazy init so the import error is deferred

    def _get_detector(self):
        if self._detector is None:
            from model.mt3_wrapper import BasicPitchDetector
            self._detector = BasicPitchDetector()
        return self._detector

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        mel, frame_target, onset_target, art_target = self.base[idx]
        n_frames = mel.shape[-1]   # T

        cache_path = self.cache / f"item_{idx:05d}.npy"
        if cache_path.exists():
            pitch_probs = np.load(str(cache_path))
        else:
            # Recover audio path from the base dataset so we can run the detector
            audio_path = self._audio_path_for(idx)
            detector   = self._get_detector()
            pitch_probs = detector.predict(audio_path, n_frames)
            np.save(str(cache_path), pitch_probs)

        pitch_t = torch.from_numpy(pitch_probs)   # (T, NUM_PITCHES)
        return mel, frame_target, onset_target, art_target, pitch_t

    def _audio_path_for(self, idx: int) -> Path:
        """Retrieve the audio path stored in the underlying dataset item."""
        item = self.base.items[idx]
        if hasattr(item, "audio_path"):
            return item.audio_path
        if isinstance(item, dict) and "audio_path" in item:
            return Path(item["audio_path"])
        raise AttributeError(
            "Cannot retrieve audio_path from dataset item. "
            "Check GuitarSetDataset.items structure."
        )


def collate_fusion(batch):
    mels, frames, onsets, arts, pitches = zip(*batch)
    return (
        torch.stack(mels),
        torch.stack(frames),
        torch.stack(onsets),
        torch.stack(arts),
        torch.stack(pitches),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fusion(
    base_model: nn.Module,
    fusion: LearnedFusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    base_model.eval()   # frozen
    fusion.train()

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    total_loss = 0.0
    n_batches  = 0

    for mel, frame_target, _, _, pitch_probs in loader:
        mel          = mel.to(device)
        frame_target = frame_target.to(device)
        pitch_probs  = pitch_probs.to(device)   # (B, T, NUM_PITCHES)

        with torch.no_grad():
            frame_logits, _, _ = base_model(mel)   # (B, T, NUM_CLASSES)

        # Expand pitch probs into class space
        pitch_expanded = expand_pitch_to_classes_torch(pitch_probs, device)  # (B, T, NUM_CLASSES)

        fused_logits = fusion(frame_logits, pitch_expanded)   # (B, T, NUM_CLASSES)
        loss = criterion(fused_logits, frame_target)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_fusion(
    base_model: nn.Module,
    fusion: LearnedFusion,
    loader: DataLoader,
    device: torch.device,
) -> float:
    base_model.eval()
    fusion.eval()

    from model.evaluate import frame_metrics

    all_preds:   list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for mel, frame_target, _, _, pitch_probs in loader:
        mel         = mel.to(device)
        pitch_probs = pitch_probs.to(device)

        frame_logits, _, _ = base_model(mel)
        pitch_expanded     = expand_pitch_to_classes_torch(pitch_probs, device)
        fused_logits       = fusion(frame_logits, pitch_expanded)

        pred   = torch.sigmoid(fused_logits).cpu().numpy()
        target = frame_target.numpy()
        for i in range(pred.shape[0]):
            all_preds.append(pred[i])
            all_targets.append(target[i])

    if not all_preds:
        return 0.0

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return frame_metrics(preds, targets, threshold=0.5)["f1"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the LearnedFusion layer")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(CHECKPOINT_DIR) / "best_model.pt")
    parser.add_argument("--root", type=Path, default=Path(GUITARSET_DIR))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("checkpoints/pitch_cache"))
    parser.add_argument("--output", type=Path,
                        default=Path(CHECKPOINT_DIR) / "fusion.pt")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    print(f"Device: {device}")

    # Load base model (frozen)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    version = "v2" if any("transformer" in k for k in ckpt["model_state_dict"]) else "v1"
    if version == "v2":
        from model.network_v2 import GuitarTranscriptionModelV2
        base_model = GuitarTranscriptionModelV2().to(device)
    else:
        from model.network import GuitarTranscriptionModel
        base_model = GuitarTranscriptionModel().to(device)
    base_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    for p in base_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded base model ({version}) from {args.checkpoint}")

    # Datasets
    train_base = GuitarSetDataset(root=args.root, split="train", augment=False)
    val_base   = GuitarSetDataset(root=args.root, split="val",   augment=False)

    train_ds = FusionDataset(train_base, args.cache_dir / "train")
    val_ds   = FusionDataset(val_base,   args.cache_dir / "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_fusion)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=0, collate_fn=collate_fusion)

    print(f"Training items: {len(train_ds)} | Validation items: {len(val_ds)}")

    # Fusion layer
    fusion    = LearnedFusion(hidden=32).to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    print(f"\nFine-tuning LearnedFusion for {args.epochs} epochs\n{'='*50}")

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        loss = train_fusion(base_model, fusion, train_loader, optimizer, device)
        f1   = validate_fusion(base_model, fusion, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | val F1 {f1:.4f} | {time.time()-t0:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "fusion_state_dict": fusion.state_dict(),
                "base_checkpoint":   str(args.checkpoint),
            }, args.output)
            print(f"  ^ New best F1 {best_f1:.4f} -- saved {args.output}")

    print(f"\nDone. Best fusion F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
