"""Phase 1B — YouTube audio downloader.

Searches YouTube for a song and downloads audio as 22050 Hz mono WAV
using yt-dlp.
"""

from __future__ import annotations

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

    cmd = [
        "yt-dlp",
        f"ytsearch1:{query}",       # first search result
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "-ar 22050 -ac 1",  # 22050 Hz mono
        "--max-downloads", "1",
        "--no-playlist",
        "--output", output_template,
        "--quiet",
    ]

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
