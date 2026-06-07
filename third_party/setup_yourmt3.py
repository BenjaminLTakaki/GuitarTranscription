#!/usr/bin/env python3
"""
Download YourMT3 into third_party/YourMT3/.

What this does
--------------
1. Verifies git-lfs is installed (required for the model checkpoint files).
2. Clones the YourMT3 HuggingFace Space, which includes pretrained model
   checkpoints (~2.8 GB via git-lfs).
3. Installs the Python dependencies (minus the HF-Spaces-only packages that
   are not needed for local inference).

Run once from the project root:
    python third_party/setup_yourmt3.py

After this, use --pitch-backend yourmt3 with model/predict.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SPACE_URL  = "https://huggingface.co/spaces/mimbres/YourMT3"
TARGET_DIR = Path(__file__).parent / "YourMT3"

# Packages only needed for the Gradio web UI -- skip them for local inference
_SKIP_PACKAGES = {"gradio", "gradio_log", "spaces", "wandb", "yt-dlp"}


def _run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    if TARGET_DIR.exists():
        print(f"YourMT3 already installed at {TARGET_DIR}")
        print("To reinstall, delete that directory and run again.")
        return

    # 1. Verify git-lfs
    print("Checking git-lfs ...")
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "git-lfs is not installed.\n"
            "Install it first:\n"
            "  Windows: winget install GitHub.GitLFS   (or https://git-lfs.com)\n"
            "  macOS:   brew install git-lfs\n"
            "  Linux:   sudo apt install git-lfs"
        )
        sys.exit(1)

    _run(["git", "lfs", "install"])

    # 2. Clone the Space (includes model checkpoints via git-lfs, ~2.8 GB)
    print(f"\nCloning YourMT3 from HuggingFace Spaces into {TARGET_DIR} ...")
    print("(This downloads ~2.8 GB of model checkpoints -- may take a few minutes)")
    _run(["git", "clone", SPACE_URL, str(TARGET_DIR)])

    # 3. Install Python dependencies, skipping HF-Spaces-only and already-present packages
    #
    # YourMT3's requirements.txt includes:
    #   --extra-index-url https://download.pytorch.org/whl/cu113   (pip option, not a package)
    #   torch / torchaudio  (skip -- user already has PyTorch installed)
    #   yt-dlp OAuth zip    (skip -- only needed for YouTube)
    _SKIP_PACKAGES_INSTALL = _SKIP_PACKAGES | {"torch", "torchaudio"}

    req_file = TARGET_DIR / "requirements.txt"
    if req_file.exists():
        pip_options: list[str] = []   # flags like --extra-index-url URL
        packages:    list[str] = []   # actual package specs

        for line in req_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Pip option lines (e.g. "--extra-index-url https://...") must be
            # split into separate tokens, not passed as a single string.
            if line.startswith("-"):
                # Skip the CUDA 11.3 index -- user has their own PyTorch build
                if "download.pytorch.org/whl/cu113" in line:
                    print(f"  Skipping PyTorch CUDA 11.3 index (using existing PyTorch): {line}")
                    continue
                pip_options.extend(line.split(None, 1))
                continue

            # Skip HF-Spaces-only and already-installed packages
            pkg_name = line.split("==")[0].split(">=")[0].split("[")[0].lower().strip()
            # Strip GitHub zip URLs to get the base package name
            if "github.com" in line and "yt-dlp" in line.lower():
                print(f"  Skipping (YouTube only): {line}")
                continue
            if pkg_name in _SKIP_PACKAGES_INSTALL:
                print(f"  Skipping (already installed or HF-only): {line}")
                continue

            packages.append(line)

        if packages:
            print(f"\nInstalling {len(packages)} dependencies ...")
            _run([sys.executable, "-m", "pip", "install"] + pip_options + packages)

    print(f"\nYourMT3 installed successfully at {TARGET_DIR}")
    print("Use it with:  python -m model.predict audio.wav --pitch-backend yourmt3")


if __name__ == "__main__":
    main()
