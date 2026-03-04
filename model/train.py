#!/usr/bin/env python3
"""Train the CNN+BiGRU guitar transcription model on the GAPS dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from model.dataset import GAPSDataset
from model.evaluate import frame_metrics
from model.network import GuitarTranscriptionModel


def collate_fn(batch):
    """Stack variable-length items — training items are fixed-length segments."""
    mels, frames, onsets = zip(*batch)
    return (
        torch.stack(mels),
        torch.stack(frames),
        torch.stack(onsets),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_frame_loss = 0.0
    total_onset_loss = 0.0
    n_batches = 0

    frame_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    onset_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight * 2)

    for mel, frame_target, onset_target in loader:
        mel = mel.to(device)
        frame_target = frame_target.to(device)
        onset_target = onset_target.to(device)

        frame_logits, onset_logits = model(mel)

        loss_frame = frame_criterion(frame_logits, frame_target)
        loss_onset = onset_criterion(onset_logits, onset_target)
        loss = loss_frame + 1.0 * loss_onset

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()

        total_loss += loss.item()
        total_frame_loss += loss_frame.item()
        total_onset_loss += loss_onset.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "frame_loss": total_frame_loss / max(n_batches, 1),
        "onset_loss": total_onset_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for mel, frame_target, _onset_target in loader:
        mel = mel.to(device)
        frame_logits, _ = model(mel)
        pred = torch.sigmoid(frame_logits).cpu().numpy()
        target = frame_target.numpy()

        for i in range(pred.shape[0]):
            all_preds.append(pred[i])
            all_targets.append(target[i])

    # Concatenate across all samples (frame-level)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return frame_metrics(preds, targets, threshold=0.5)


def main():
    parser = argparse.ArgumentParser(description="Train guitar transcription model")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing the GAPS/ directory",
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--scheduler", type=str, default="plateau",
        choices=["plateau", "cosine"],
        help="LR scheduler: 'plateau' (ReduceLROnPlateau) or 'cosine' (CosineAnnealingLR)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path(CHECKPOINT_DIR)
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    print("Loading training set...")
    train_ds = GAPSDataset(root=args.root, split="train", augment=True)
    print(f"  {len(train_ds)} training items")

    print("Loading test set...")
    test_ds = GAPSDataset(root=args.root, split="test")
    print(f"  {len(test_ds)} test items")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # Model
    model = GuitarTranscriptionModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Positive-class weighting (notes are sparse → weight them higher)
    pos_weight = torch.tensor([5.0]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )

    # Checkpointing
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    print(f"\nTraining for {args.epochs} epochs\n{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        val_stats = validate(model, test_loader, device)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"P {val_stats['precision']:.3f}  R {val_stats['recall']:.3f}  "
            f"F1 {val_stats['f1']:.3f} | "
            f"lr {lr:.2e} | {elapsed:.1f}s"
        )

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_stats["f1"])

        # Save best
        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            ckpt_path = args.checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "f1": best_f1,
                },
                ckpt_path,
            )
            print(f"  ↑ New best F1={best_f1:.4f} — saved {ckpt_path}")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = args.checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "f1": val_stats["f1"],
                },
                ckpt_path,
            )

    print(f"\nDone. Best test F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
