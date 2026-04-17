"""Phase 4 — End-to-end orchestrator for the alignment pipeline.

Usage:
    python -m pipeline.run --dadagp-dir DadaGP/ --max-songs 500
    python -m pipeline.run --dadagp-dir DadaGP/ --skip-download --skip-separation
    python -m pipeline.run --tabs-urls-file pipeline/tab_urls.example.txt --max-tab-urls 100
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pipeline.ingest import ingest_directory
from pipeline.fetch_tabs import fetch_tabs_from_urls
from pipeline.fetch_audio import search_and_download
from pipeline.isolate import isolate_guitar
from pipeline.synthesize_tab import synthesize_midi
from pipeline.align import align_audio, warp_events
from pipeline.quality import chunk_and_filter


def run_pipeline(
    dadagp_dir: Path,
    output_dir: Path,
    tabs_urls_file: Path | None = None,
    tabs_output_dir: Path | None = None,
    max_tab_urls: int | None = None,
    max_songs: int | None = None,
    standard_tuning_only: bool = True,
    quality_threshold: float = 0.5,
    search_suffix: str = "guitar cover",
    skip_download: bool = False,
    skip_separation: bool = False,
) -> dict[str, int]:
    """Run the full alignment pipeline end-to-end.

    Returns stats dict with counts of processed, downloaded, separated,
    aligned, and chunks_saved.
    """
    stats = {
        "tabs_processed": 0,
        "tabs_downloaded": 0,
        "tabs_extracted": 0,
        "processed": 0,
        "downloaded": 0,
        "separated": 0,
        "aligned": 0,
        "chunks_saved": 0,
        "errors": 0,
    }

    # Subdirectories
    ingest_dir = output_dir / "ingested"
    audio_dl_dir = output_dir / "downloaded_audio"
    separated_dir = output_dir / "separated"
    synth_dir = output_dir / "synthesized"
    aligned_dir = output_dir / "AlignedDataset"

    ingest_source = dadagp_dir

    # ---- Step 0: Optionally fetch tab files from URL list ----
    if tabs_urls_file is not None:
        resolved_tabs_dir = tabs_output_dir or (output_dir / "fetched_tabs")
        print("\n" + "=" * 60)
        print("STEP 0: Fetching Guitar Pro tabs")
        print("=" * 60)

        tab_stats = fetch_tabs_from_urls(
            urls_file=tabs_urls_file,
            output_dir=resolved_tabs_dir,
            max_urls=max_tab_urls,
        )
        stats["tabs_processed"] = tab_stats["processed"]
        stats["tabs_downloaded"] = tab_stats["downloaded"]
        stats["tabs_extracted"] = tab_stats["extracted"]
        stats["errors"] += tab_stats["errors"]

        ingest_source = resolved_tabs_dir

    # ---- Step 1: Ingest Guitar Pro files ----
    print("\n" + "=" * 60)
    print("STEP 1: Ingesting Guitar Pro files")
    print("=" * 60)

    manifest = ingest_directory(
        ingest_source, ingest_dir,
        max_songs=max_songs,
        standard_tuning_only=standard_tuning_only,
    )
    if not manifest:
        print("No tracks found. Exiting.")
        return stats

    # ---- Process each track through the pipeline ----
    audio_dl_dir.mkdir(parents=True, exist_ok=True)
    separated_dir.mkdir(parents=True, exist_ok=True)
    synth_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    results_log: list[dict] = []

    for i, entry in enumerate(manifest):
        title = entry["song_title"]
        artist = entry["artist"]
        midi_path = Path(entry["midi_path"])
        jams_path = Path(entry["jams_path"])
        song_id = midi_path.stem

        print(f"\n--- [{i+1}/{len(manifest)}] {artist} - {title} ---")

        try:
            # ---- Step 2: Download audio ----
            if skip_download:
                # Look for pre-existing audio
                wav_candidates = list(audio_dl_dir.glob(f"{song_id}.*"))
                real_audio = wav_candidates[0] if wav_candidates else None
            else:
                print("  Step 2: Downloading audio...")
                real_audio = search_and_download(
                    title, artist, audio_dl_dir, search_suffix,
                )
                if real_audio is not None:
                    # Rename to match song_id
                    new_path = audio_dl_dir / f"{song_id}.wav"
                    if real_audio != new_path:
                        if new_path.exists():
                            # Reuse existing canonical file on repeated runs
                            try:
                                real_audio.unlink(missing_ok=True)
                            except Exception:
                                pass
                        else:
                            real_audio.rename(new_path)
                        real_audio = new_path
                    stats["downloaded"] += 1
                    time.sleep(2.5)  # rate limiting

            if real_audio is None or not real_audio.exists():
                print("  [SKIP] No audio available")
                stats["errors"] += 1
                continue

            # ---- Step 3: Isolate guitar stem ----
            if skip_separation:
                # Use raw audio or check for pre-existing stem
                stem_dir = separated_dir / "htdemucs" / song_id
                guitar_stem = stem_dir / "other.wav"
                if not guitar_stem.exists():
                    guitar_stem = real_audio  # fallback to raw
            else:
                print("  Step 3: Isolating guitar stem...")
                guitar_stem = isolate_guitar(real_audio, separated_dir)
                if guitar_stem is not None:
                    stats["separated"] += 1

            if guitar_stem is None or not guitar_stem.exists():
                print("  [SKIP] Separation failed")
                stats["errors"] += 1
                continue

            # ---- Step 4: Synthesize tab audio ----
            print("  Step 4: Synthesizing tab audio...")
            synth_wav = synth_dir / f"{song_id}.wav"
            if not synth_wav.exists():
                ok = synthesize_midi(midi_path, synth_wav)
                if not ok:
                    print("  [SKIP] FluidSynth render failed")
                    stats["errors"] += 1
                    continue

            # ---- Step 5: Align real ↔ synth ----
            print("  Step 5: Aligning audio...")
            warp_path, cost = align_audio(
                str(guitar_stem), str(synth_wav),
            )
            print(f"  DTW cost: {cost:.4f}")

            # ---- Step 6: Warp events ----
            print("  Step 6: Warping events...")
            with open(jams_path, "r", encoding="utf-8") as f:
                jams_data = json.load(f)

            # Extract events from JAMS
            events = _jams_to_events(jams_data)
            warped_events = warp_events(events, warp_path)
            stats["aligned"] += 1

            # ---- Step 7: Chunk, filter, export ----
            print("  Step 7: Chunking and filtering...")
            n_chunks = chunk_and_filter(
                str(guitar_stem),
                warped_events,
                warp_path,
                aligned_dir,
                song_id=song_id,
                max_dtw_cost_per_frame=quality_threshold,
            )
            stats["chunks_saved"] += n_chunks
            print(f"  Saved {n_chunks} chunks")

            stats["processed"] += 1
            results_log.append({
                "song_id": song_id,
                "title": title,
                "artist": artist,
                "dtw_cost": round(cost, 4),
                "chunks_saved": n_chunks,
                "status": "ok",
            })

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            stats["errors"] += 1
            results_log.append({
                "song_id": song_id,
                "title": title,
                "artist": artist,
                "dtw_cost": -1,
                "chunks_saved": 0,
                "status": f"error: {exc}",
            })

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Output: {aligned_dir}/")

    # Save results log
    if results_log:
        log_path = output_dir / "pipeline_results.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_log[0].keys())
            writer.writeheader()
            writer.writerows(results_log)
        print(f"  Results log: {log_path}")

    return stats


def _jams_to_events(jams_data: dict) -> list[dict]:
    """Extract note events from a JAMS dict.

    Returns list of dicts with: onset, offset, string, fret, midi, velocity.
    """
    events = []
    string_idx = 0

    for ann in jams_data.get("annotations", []):
        if ann.get("namespace") != "note_midi":
            continue

        ds = ann.get("annotation_metadata", {}).get("data_source", str(string_idx))
        try:
            s = int(ds)
        except (ValueError, TypeError):
            s = string_idx

        for obs in ann.get("data", []):
            onset = obs["time"]
            duration = obs["duration"]
            midi_pitch = int(round(obs["value"]))

            events.append({
                "onset": onset,
                "offset": onset + duration,
                "string": s,
                "fret": 0,  # not always available in JAMS
                "midi": midi_pitch,
                "velocity": 80,
            })

        string_idx += 1

    return events


def main():
    parser = argparse.ArgumentParser(
        description="Audio-to-Tab Alignment Dataset Pipeline"
    )
    parser.add_argument(
        "--dadagp-dir", type=Path, default=Path("DadaGP"),
        help="Path to DadaGP directory containing Guitar Pro files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("pipeline_output"),
        help="Output directory for all pipeline artifacts",
    )
    parser.add_argument(
        "--tabs-urls-file", type=Path, default=None,
        help=(
            "Optional text file with one tab URL per line. "
            "If provided, tabs are fetched first and used as ingestion input."
        ),
    )
    parser.add_argument(
        "--tabs-output-dir", type=Path, default=None,
        help="Where fetched tab files are stored (default: <output-dir>/fetched_tabs)",
    )
    parser.add_argument(
        "--max-tab-urls", type=int, default=None,
        help="Maximum number of tab URLs to fetch from --tabs-urls-file",
    )
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument(
        "--allow-non-standard-tuning", action="store_true",
        help="Include guitar tracks that are not in standard EADGBE tuning",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.5,
        help="Max DTW cost per frame for chunk acceptance (lower = stricter)",
    )
    parser.add_argument(
        "--search-suffix", type=str, default="guitar cover",
        help="Suffix for YouTube search queries",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip YouTube download (use pre-existing audio)",
    )
    parser.add_argument(
        "--skip-separation", action="store_true",
        help="Skip Demucs source separation",
    )
    args = parser.parse_args()

    run_pipeline(
        dadagp_dir=args.dadagp_dir,
        output_dir=args.output_dir,
        tabs_urls_file=args.tabs_urls_file,
        tabs_output_dir=args.tabs_output_dir,
        max_tab_urls=args.max_tab_urls,
        max_songs=args.max_songs,
        standard_tuning_only=not args.allow_non_standard_tuning,
        quality_threshold=args.quality_threshold,
        search_suffix=args.search_suffix,
        skip_download=args.skip_download,
        skip_separation=args.skip_separation,
    )


if __name__ == "__main__":
    main()
