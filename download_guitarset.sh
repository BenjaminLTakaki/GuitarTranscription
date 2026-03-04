#!/usr/bin/env bash
# Download GuitarSet from Zenodo and extract into GuitarSet/ directory.
#
# We only need:
#   - annotation.zip   (~39 MB)  → GuitarSet/annotation/*.jams
#   - audio_mono-mic.zip (~657 MB) → GuitarSet/audio_mono-mic/*.wav
#
# Usage:
#   chmod +x download_guitarset.sh
#   ./download_guitarset.sh

set -euo pipefail

DEST="GuitarSet"
ZENODO="https://zenodo.org/records/3371780/files"

mkdir -p "$DEST"

echo "=== Downloading GuitarSet annotations (~39 MB) ==="
wget -N -P "$DEST" "${ZENODO}/annotation.zip"

echo "=== Downloading GuitarSet mono-mic audio (~657 MB) ==="
wget -N -P "$DEST" "${ZENODO}/audio_mono-mic.zip"

echo "=== Extracting annotation.zip ==="
unzip -o -q "$DEST/annotation.zip" -d "$DEST"

echo "=== Extracting audio_mono-mic.zip ==="
unzip -o -q "$DEST/audio_mono-mic.zip" -d "$DEST"

# The zips may extract flat (no subdirectory).  Sort files into the expected
# layout if annotation/ and audio_mono-mic/ subdirectories are missing.
if [ ! -d "$DEST/annotation" ]; then
    echo "=== Sorting .jams into annotation/ ==="
    mkdir -p "$DEST/annotation"
    mv "$DEST"/*.jams "$DEST/annotation/"
fi
if [ ! -d "$DEST/audio_mono-mic" ]; then
    echo "=== Sorting .wav into audio_mono-mic/ ==="
    mkdir -p "$DEST/audio_mono-mic"
    mv "$DEST"/*_mic.wav "$DEST/audio_mono-mic/"
fi

echo ""
echo "Done!  Layout:"
echo "  GuitarSet/annotation/*.jams   ($(ls "$DEST"/annotation/*.jams 2>/dev/null | wc -l) files)"
echo "  GuitarSet/audio_mono-mic/*.wav ($(ls "$DEST"/audio_mono-mic/*.wav 2>/dev/null | wc -l) files)"
echo ""
echo "You can now train with:  python -m model.train"
