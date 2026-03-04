#!/usr/bin/env python3
"""Diagnostic script: visualise raw model outputs and sweep thresholds.

Usage
-----
    python diagnose.py audio.wav                          # uses best_model.pt
    python diagnose.py audio.wav --checkpoint ckpt.pt     # specific checkpoint
    python diagnose.py audio.wav --with-ground-truth path/to/ref.mid

Outputs (saved to output/diagnostics/):
    1. raw_frame_probs.png   – heatmap of sigmoid frame probabilities
    2. raw_onset_probs.png   – heatmap of sigmoid onset probabilities
    3. threshold_sweep.png   – note count + pitch coverage vs threshold
    4. threshold_sweep.txt   – tabular summary
    5. per_threshold MIDIs   – one MIDI per threshold value for A/B listening
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.constants import (
    CHECKPOINT_DIR,
    HOP_LENGTH,
    MIDI_MIN,
    MIDI_MAX,
    NUM_PITCHES,
    SAMPLE_RATE,
)
from model.network import GuitarTranscriptionModel
from model.predict import load_mel, pianoroll_to_notes, write_midi


# ── 1. Raw probability heatmaps ──────────────────────────────────────────

def plot_pianoroll_heatmap(
    prob: np.ndarray,
    title: str,
    save_path: Path,
    figsize: tuple = (18, 6),
):
    """Plot a (T, P) probability matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        prob.T,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Sigmoid probability")

    # Y-axis: MIDI pitch labels every 6 semitones
    pitch_ticks = list(range(0, NUM_PITCHES, 6))
    ax.set_yticks(pitch_ticks)
    ax.set_yticklabels([f"MIDI {p + MIDI_MIN}" for p in pitch_ticks])

    # X-axis: time in seconds
    frame_sec = HOP_LENGTH / SAMPLE_RATE
    T = prob.shape[0]
    time_ticks = np.arange(0, T, max(1, int(1.0 / frame_sec)))  # every ~1 s
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f"{t * frame_sec:.1f}" for t in time_ticks], fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 2. Threshold sweep ───────────────────────────────────────────────────

def sweep_thresholds(
    frame_prob: np.ndarray,
    onset_prob: np.ndarray,
    out_dir: Path,
    thresholds: list[float] | None = None,
):
    """Run pianoroll_to_notes at many thresholds and report stats."""
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                      0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    results = []
    midi_dir = out_dir / "midi_by_threshold"
    midi_dir.mkdir(parents=True, exist_ok=True)

    for th in thresholds:
        notes = pianoroll_to_notes(frame_prob, onset_prob, threshold=th)
        n_notes = len(notes)
        unique_pitches = sorted(set(n["midi"] for n in notes)) if notes else []
        total_dur = sum(n["end"] - n["start"] for n in notes) if notes else 0.0
        avg_dur = total_dur / n_notes if n_notes else 0.0

        results.append({
            "threshold": th,
            "n_notes": n_notes,
            "n_unique_pitches": len(unique_pitches),
            "avg_duration_s": avg_dur,
            "pitch_range": f"{min(unique_pitches)}-{max(unique_pitches)}" if unique_pitches else "—",
        })

        # Save a MIDI for each threshold
        midi_path = midi_dir / f"th_{th:.2f}.mid"
        write_midi(notes, midi_path)

    # ── Table ──
    header = f"{'Thresh':>7} {'Notes':>6} {'Pitches':>8} {'AvgDur':>8} {'Range':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['threshold']:7.2f} {r['n_notes']:6d} {r['n_unique_pitches']:8d} "
            f"{r['avg_duration_s']:8.3f} {r['pitch_range']:>10}"
        )
    table_txt = "\n".join(lines)
    print("\n" + table_txt)

    txt_path = out_dir / "threshold_sweep.txt"
    txt_path.write_text(table_txt, encoding="utf-8")
    print(f"\n  Saved: {txt_path}")

    # ── Chart ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ths = [r["threshold"] for r in results]
    ax1.plot(ths, [r["n_notes"] for r in results], "o-", color="#e63946", label="Note count")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Note count", color="#e63946")
    ax1.tick_params(axis="y", labelcolor="#e63946")

    ax2 = ax1.twinx()
    ax2.plot(ths, [r["n_unique_pitches"] for r in results], "s-", color="#457b9d", label="Unique pitches")
    ax2.set_ylabel("Unique pitches", color="#457b9d")
    ax2.tick_params(axis="y", labelcolor="#457b9d")

    fig.suptitle("Threshold Sweep: Note Count & Pitch Coverage")
    fig.tight_layout()
    chart_path = out_dir / "threshold_sweep.png"
    fig.savefig(str(chart_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {chart_path}")

    return results


# ── 3. Probability distribution histogram ────────────────────────────────

def plot_prob_distribution(prob: np.ndarray, label: str, save_path: Path):
    """Histogram of non-zero probabilities to see model confidence."""
    flat = prob.flatten()
    nonzero = flat[flat > 0.01]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nonzero, bins=100, color="#264653", edgecolor="none", alpha=0.85)
    ax.axvline(0.5, color="#e76f51", ls="--", lw=2, label="default threshold (0.5)")
    ax.set_xlabel("Sigmoid probability")
    ax.set_ylabel("Frame×pitch count")
    ax.set_title(f"{label} — distribution of non-zero probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── 4. Checkpoint stats ──────────────────────────────────────────────────

def print_checkpoint_info(ckpt_path: Path, device: torch.device):
    """Print the stored best-F1 and epoch from the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    epoch = ckpt.get("epoch", "?")
    f1 = ckpt.get("f1", "?")
    print(f"\n+== Checkpoint info ====================+")
    print(f"|  File  : {ckpt_path.name:<28}|")
    print(f"|  Epoch : {str(epoch):<28}|")
    print(f"|  Best F1: {str(f1):<27}|")
    print(f"+=======================================+\n")
    return ckpt


# ── Main ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Diagnose model vs. post-processing quality")
    parser.add_argument("audio_file", type=Path, help="Input audio (wav/mp3/flac)")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(CHECKPOINT_DIR) / "best_model.pt",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=Path("output/diagnostics"),
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.audio_file.exists():
        raise FileNotFoundError(f"Audio not found: {args.audio_file}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint info ──
    print_checkpoint_info(args.checkpoint, device)

    # ── Run model ──
    print(f"Running inference on: {args.audio_file}")
    mel = load_mel(args.audio_file)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)

    model = GuitarTranscriptionModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    frame_logits, onset_logits = model(mel_t)
    frame_prob = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()
    onset_prob = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()

    duration_sec = mel.shape[1] * HOP_LENGTH / SAMPLE_RATE
    print(f"Audio duration: {duration_sec:.1f}s | Frames: {frame_prob.shape[0]} | Pitches: {frame_prob.shape[1]}")

    # ── Quick stats ──
    print(f"\nFrame probs — max: {frame_prob.max():.4f}, mean: {frame_prob.mean():.4f}, "
          f"% > 0.5: {(frame_prob > 0.5).mean() * 100:.2f}%")
    print(f"Onset probs — max: {onset_prob.max():.4f}, mean: {onset_prob.mean():.4f}, "
          f"% > 0.5: {(onset_prob > 0.5).mean() * 100:.2f}%")

    # ── Plots ──
    print("\nGenerating visualisations...")
    plot_pianoroll_heatmap(frame_prob, "Frame probabilities (sigmoid)", args.output_dir / "raw_frame_probs.png")
    plot_pianoroll_heatmap(onset_prob, "Onset probabilities (sigmoid)", args.output_dir / "raw_onset_probs.png")
    plot_prob_distribution(frame_prob, "Frame head", args.output_dir / "frame_prob_distribution.png")
    plot_prob_distribution(onset_prob, "Onset head", args.output_dir / "onset_prob_distribution.png")

    # ── Threshold sweep ──
    print("\nSweeping thresholds...")
    sweep_thresholds(frame_prob, onset_prob, args.output_dir)

    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("\nHow to interpret:")
    print("  • If the heatmaps show clear note activations → model is OK, post-processing is the bottleneck")
    print("  • If the heatmaps are noisy/blank → model needs improvement")
    print("  • If note count changes dramatically with threshold → threshold tuning is critical")
    print("  • Listen to midi_by_threshold/*.mid to find the best threshold by ear")


if __name__ == "__main__":
    main()
