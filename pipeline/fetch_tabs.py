"""Phase 0 — Fetch Guitar Pro tab files from URL lists.

Reads a text file containing one URL per line and downloads:
- direct Guitar Pro files (.gp, .gp3, .gp4, .gp5, .gpx)
- optional archives (.zip, .tar, .tar.gz, .tgz) and extracts GP files from them

This keeps pipeline ingestion deterministic: only explicitly provided URLs are used.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

GP_EXTS = {".gp", ".gp3", ".gp4", ".gp5", ".gpx"}
ARCHIVE_EXTS = {".zip", ".tar", ".tar.gz", ".tgz"}


def _read_urls(urls_file: Path) -> list[str]:
    urls: list[str] = []
    for raw in urls_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def _safe_name_from_url(url: str, default: str) -> str:
    path = urlparse(url).path
    name = Path(path).name if path else default
    name = name.replace(" ", "_")
    return name or default


def _archive_ext(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    if lower.endswith(".tgz"):
        return ".tgz"
    suffix = Path(lower).suffix
    if suffix in ARCHIVE_EXTS:
        return suffix
    return None


def _extract_gp_from_zip(blob: bytes, output_dir: Path) -> int:
    saved = 0
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            src_name = Path(info.filename).name
            if Path(src_name).suffix.lower() not in GP_EXTS:
                continue

            out_path = output_dir / src_name
            if out_path.exists():
                out_path = output_dir / f"{out_path.stem}_dup{out_path.suffix}"

            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            saved += 1
    return saved


def _extract_gp_from_tar(blob: bytes, output_dir: Path) -> int:
    saved = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue

            src_name = Path(member.name).name
            if Path(src_name).suffix.lower() not in GP_EXTS:
                continue

            stream = tf.extractfile(member)
            if stream is None:
                continue

            out_path = output_dir / src_name
            if out_path.exists():
                out_path = output_dir / f"{out_path.stem}_dup{out_path.suffix}"

            with stream, open(out_path, "wb") as dst:
                dst.write(stream.read())
            saved += 1
    return saved


def fetch_tabs_from_urls(
    urls_file: Path,
    output_dir: Path,
    timeout: int = 90,
    allow_archives: bool = True,
    max_urls: int | None = None,
) -> dict[str, int]:
    """Download tab files listed in ``urls_file`` into ``output_dir``.

    Args:
        urls_file: Text file containing one URL per line.
        output_dir: Destination directory for fetched GP files.
        timeout: Per-URL timeout in seconds.
        allow_archives: If True, extract GP files from supported archives.
        max_urls: Optional cap on how many URLs to process.

    Returns:
        Stats dict with counts for processed, downloaded, extracted, and errors.
    """
    if not urls_file.exists():
        raise FileNotFoundError(f"Tab URL file not found: {urls_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    urls = _read_urls(urls_file)
    if max_urls is not None:
        urls = urls[:max_urls]

    stats = {
        "processed": 0,
        "downloaded": 0,
        "extracted": 0,
        "errors": 0,
    }

    for i, url in enumerate(urls):
        stats["processed"] += 1
        print(f"  [tabs {i+1}/{len(urls)}] {url}")
        try:
            with urlopen(url, timeout=timeout) as resp:
                data = resp.read()

            file_name = _safe_name_from_url(url, default=f"tab_{i+1}")
            lower_name = file_name.lower()
            ext = Path(lower_name).suffix
            archive_ext = _archive_ext(lower_name)

            if ext in GP_EXTS:
                out_path = output_dir / file_name
                if out_path.exists():
                    out_path = output_dir / f"{out_path.stem}_dup{out_path.suffix}"
                out_path.write_bytes(data)
                stats["downloaded"] += 1
                continue

            if allow_archives and archive_ext in ARCHIVE_EXTS:
                if archive_ext == ".zip":
                    n = _extract_gp_from_zip(data, output_dir)
                else:
                    n = _extract_gp_from_tar(data, output_dir)
                stats["extracted"] += n
                print(f"    extracted GP files: {n}")
                continue

            print("    [SKIP] Unsupported file type (expect GP file or archive)")

        except Exception as exc:
            print(f"    [FAIL] {exc}")
            stats["errors"] += 1

    return stats