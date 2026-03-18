"""Phase 1B — YouTube audio downloader.

Searches YouTube for a song and downloads audio as 22050 Hz mono WAV
using yt-dlp.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def search_and_download(
    song_title: str,
    artist: str,
    output_dir: Path,
    search_suffix: str = "guitar cover",
    timeout: int = 120,
) -> Path | None:
    """Search YouTube for the song and download audio as WAV.

    Returns the path to the downloaded WAV, or None on failure.
    """
    query = f"{artist} {song_title} {search_suffix}"
    output_template = str(output_dir / "%(id)s.%(ext)s")
    ffmpeg_location = _resolve_ffmpeg_location()
    js_runtime = _resolve_js_runtime()

    cmd = [
        "yt-dlp",
        f"ytsearch1:{query}",       # first search result
        "--extractor-args", "youtube:player_skip=js",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 22050 -ac 1",  # 22050 Hz mono
        "--no-playlist",
        "--output", output_template,
        "--quiet",
    ]
    if js_runtime is not None:
        cmd.extend(["--js-runtimes", js_runtime])
    if ffmpeg_location is not None:
        cmd.extend(["--ffmpeg-location", ffmpeg_location])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] yt-dlp timed out for: {query}")
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip()[:200] if result.stderr else ""
        print(f"  [FAIL] yt-dlp error for '{query}': {stderr}")
        return None

    # Find the downloaded file
    wav_files = sorted(output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
    if wav_files:
        return wav_files[-1]
    return None


def _resolve_ffmpeg_location() -> str | None:
    """Return directory containing ffmpeg/ffprobe if found, else None."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return str(Path(ffmpeg).parent)

    # Common winget install location on Windows
    localapp = os.environ.get("LOCALAPPDATA")
    if not localapp:
        return None

    winget_root = Path(localapp) / "Microsoft" / "WinGet" / "Packages"
    if not winget_root.exists():
        return None

    for pkg_dir in winget_root.glob("Gyan.FFmpeg*"):
        bin_candidates = list(pkg_dir.glob("**/bin"))
        for bin_dir in bin_candidates:
            if (bin_dir / "ffmpeg.exe").exists() and (bin_dir / "ffprobe.exe").exists():
                return str(bin_dir)

    return None


def _resolve_js_runtime() -> str | None:
    """Pick the best available JavaScript runtime for yt-dlp."""
    for candidate in ("node", "deno", "bun"):
        if shutil.which(candidate):
            return candidate
    return None


def download_batch(
    manifest: list[dict],
    output_dir: Path,
    search_suffix: str = "guitar cover",
    delay: float = 2.5,
    max_songs: int | None = None,
) -> dict[str, Path]:
    """Download audio for a batch of songs from a manifest.

    Returns a mapping of manifest jams_path → downloaded WAV path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    items = manifest[:max_songs] if max_songs else manifest
    for i, entry in enumerate(items):
        title = entry.get("song_title", "")
        artist = entry.get("artist", "")
        jams_path = entry.get("jams_path", "")

        if not title:
            continue

        # Skip if already downloaded
        existing = list(output_dir.glob(f"*{Path(jams_path).stem}*"))
        if existing:
            results[jams_path] = existing[0]
            continue

        print(f"  [{i+1}/{len(items)}] Downloading: {artist} - {title}")
        wav = search_and_download(title, artist, output_dir, search_suffix)
        if wav is not None:
            # Rename to match the jams stem for easy pairing
            new_name = output_dir / f"{Path(jams_path).stem}.wav"
            wav.rename(new_name)
            results[jams_path] = new_name

        # Rate limiting
        if i < len(items) - 1:
            time.sleep(delay)

    print(f"Downloaded {len(results)}/{len(items)} songs")
    return results
