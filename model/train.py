#!/usr/bin/env python3
"""Train the CNN+BiGRU guitar transcription model on GuitarSet."""

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
    GUITARSET_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from model.guitarset_dataset import GuitarSetDataset
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
    loader: DataLoader | None,
    device: torch.device,
) -> dict[str, float]:
    if loader is None or len(loader.dataset) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

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

    if not all_preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Concatenate across all samples (frame-level)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return frame_metrics(preds, targets, threshold=0.5)


def main():
    parser = argparse.ArgumentParser(description="Train guitar transcription model")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(GUITARSET_DIR),
        help="Path to GuitarSet/ directory (containing annotation/ and audio_mono-mic/)",
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
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--synth-root", type=Path, default=None,
        help="Path to synthetic data dir. When set, all tracks in synth-root/ "
             "are used for training while GuitarSet val/test splits are kept "
             "for evaluation (pretrain+fine-tune workflow).",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Datasets
    if args.synth_root is not None:
        print(f"Loading synthetic training set ({args.synth_root})...")
        train_ds = GuitarSetDataset(root=args.synth_root, split="all", augment=True)
        print(f"  {len(train_ds)} synthetic training items")
    else:
        print("Loading training set (GuitarSet — players 00-03)...")
        train_ds = GuitarSetDataset(root=args.root, split="train", augment=True)
        print(f"  {len(train_ds)} training items")

    print("Loading validation set (GuitarSet — player 04)...")
    val_ds = GuitarSetDataset(root=args.root, split="val")
    print(f"  {len(val_ds)} validation items")

    print("Loading test set (GuitarSet — player 05)...")
    test_ds = GuitarSetDataset(root=args.root, split="test")
    print(f"  {len(test_ds)} test items")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
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
    # With 126 tablature classes (vs 49 pitches), targets are ~2.6× sparser.
    pos_weight = torch.tensor([10.0]).to(device)

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
    start_epoch = 1

    # Resume from checkpoint
    if args.resume is not None:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("f1", 0.0)
        print(f"  Loaded epoch {ckpt.get('epoch', '?')}, F1={best_f1:.4f}")

    # Use validation set for model selection, test set for final reporting
    if len(val_ds) > 0:
        eval_loader: DataLoader | None = val_loader
        eval_label = "val"
    elif len(test_ds) > 0:
        eval_loader = test_loader
        eval_label = "test"
    else:
        eval_loader = None
        eval_label = "none"

    print(f"\nTraining for epochs {start_epoch}–{args.epochs}\n{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        val_stats = validate(model, eval_loader, device)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        val_str = (
            f"P {val_stats['precision']:.3f}  R {val_stats['recall']:.3f}  F1 {val_stats['f1']:.3f}"
            if eval_loader is not None
            else "no eval set"
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_stats['loss']:.4f} | "
            f"{val_str} | "
            f"lr {lr:.2e} | {elapsed:.1f}s"
        )

        if args.scheduler == "cosine":
            scheduler.step()
        else:
            metric = val_stats["f1"] if eval_loader is not None else -train_stats["loss"]
            scheduler.step(metric)

        # Save best — when no eval set, save based on lowest training loss
        save_best = False
        if eval_loader is not None:
            if val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                save_best = True
        else:
            # No eval data: use training loss (lower is better)
            if best_f1 == 0.0 or train_stats["loss"] < best_f1:
                best_f1 = train_stats["loss"]
                save_best = True

        if save_best:
            ckpt_path = args.checkpoint_dir / "best_model.pt"
            ckpt_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "f1": best_f1,
            }
            torch.save(ckpt_data, ckpt_path)
            metric_name = "F1" if eval_loader is not None else "loss"
            print(f"  ↑ New best {metric_name}={best_f1:.4f} — saved {ckpt_path}")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = args.checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "f1": val_stats["f1"],
                },
                ckpt_path,
            )

    print(f"\nDone. Best test F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
