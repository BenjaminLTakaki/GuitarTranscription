#!/usr/bin/env python3
"""
Standalone YourMT3 transcription worker.

Run via YourMT3Detector (model/mt3_wrapper.py) -- not directly.

Why a separate script?
  YourMT3 uses a 'model/' package internally (model.ymt3, model.init_train, ...)
  that conflicts with this project's own 'model/' package when both are imported
  in the same Python process.  Running YourMT3 in a subprocess gives it a clean
  sys.path where only its own 'model/' is visible.

Usage (internal):
  python third_party/yourmt3_transcribe.py --audio <path> --output-dir <path>
  Prints the absolute path of the output MIDI file to stdout on success.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---- Env vars MUST be set before any other import --------------------------
# WANDB_DISABLED / WANDB_MODE: prevent wandb from hooking sys.stdout.
#   Without this, wandb.sdk.lib.console_capture replaces stdout with a proxy
#   that uses the original cp1252 encoding, breaking emoji in YourMT3's timer.
# TORCH_COMPILE_DISABLE: PyTorch 2.5.x on Windows crashes in the inductor when
#   @torch.compile is used as a class decorator (t5mod.py).
import os
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

# Reconfigure stdout/stderr to UTF-8 so YourMT3 emoji (⏰) are safe.
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---- sys.path setup (must happen before ANY other import) ------------------
_HERE      = Path(__file__).parent.resolve()
_YMT3_ROOT = _HERE / "YourMT3"
_AMT_SRC   = _YMT3_ROOT / "amt" / "src"

# Prepend YourMT3 paths so its 'model/' and 'utils/' packages win
sys.path = [str(_YMT3_ROOT), str(_AMT_SRC)] + [
    p for p in sys.path
    # Remove any path that would expose this project's own 'model/' package.
    # We identify those as paths that contain a 'model/constants.py' but NOT
    # 'model/ymt3.py' (which is unique to YourMT3).
    if not (
        (Path(p) / "model" / "constants.py").exists()
        and not (Path(p) / "model" / "ymt3.py").exists()
    )
]
# ---------------------------------------------------------------------------

import torch

# YourMT3's t5mod.py uses @torch.compile as a class decorator.  On PyTorch
# 2.5.x + Windows the inductor backend crashes at import time before any
# inference happens.  Replace torch.compile with a no-op passthrough so we
# still get correct (un-compiled) inference.
if not hasattr(torch, '_compile_orig'):
    torch._compile_orig = torch.compile
    torch.compile = lambda fn=None, **kw: (fn if fn is not None else lambda f: f)

from model_helper import load_model_checkpoint, transcribe


def prepare_media(source_path: str, source_type: str = "audio_filepath") -> dict:
    """Minimal re-implementation of app.py's prepare_media() for local inference.
    transcribe() only needs 'filepath' and 'track_name' from this dict."""
    import torchaudio
    path = Path(source_path).resolve()
    info = torchaudio.info(str(path))
    return {
        "filepath":        str(path),
        "track_name":      path.stem,
        "sample_rate":     info.sample_rate,
        "bits_per_sample": info.bits_per_sample,
        "num_channels":    info.num_channels,
        "num_frames":      info.num_frames,
        "duration":        info.num_frames / info.sample_rate,
        "encoding":        info.encoding,
    }

_YOURMT3_ARGS = [
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
    "-p",    "2024",
    "-tk",   "mc13_full_plus_256",
    "-dec",  "multi-t5",
    "-nl",   "26",
    "-enc",  "perceiver-tf",
    "-sqr",  "1",
    "-ff",   "moe",
    "-wf",   "4",
    "-nmoe", "8",
    "-kmoe", "2",
    "-act",  "silu",
    "-epe",  "rope",
    "-rp",   "1",
    "-ac",   "spec",
    "-hop",  "300",
    "-atc",  "1",
    "-pr",   "16",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",      required=True, help="Input audio file path")
    parser.add_argument("--output-dir", required=True, help="Directory to write MIDI output")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    out_dir    = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize_trainer() builds checkpoint paths relative to CWD.
    # The checkpoints live inside the YourMT3 clone, so we must be there.
    orig_dir = os.getcwd()
    os.chdir(str(_YMT3_ROOT))
    try:
        model = load_model_checkpoint(args=list(_YOURMT3_ARGS), device=device)
        model.eval()
    finally:
        os.chdir(orig_dir)

    # Prepare audio metadata dict
    audio_info = prepare_media(str(audio_path), source_type="audio_filepath")

    # transcribe() writes to './model_output/<track>.mid' relative to CWD
    try:
        os.chdir(str(out_dir))
        midi_rel  = transcribe(model, audio_info)
        midi_abs  = out_dir / midi_rel
        # Print the absolute MIDI path to stdout -- YourMT3Detector reads this
        print(str(midi_abs), flush=True)
    finally:
        os.chdir(orig_dir)


if __name__ == "__main__":
    main()
