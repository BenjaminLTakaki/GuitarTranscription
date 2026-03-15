#!/usr/bin/env python3
"""Run all three transcription approaches on C major scale audio and capture results."""

import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np

# Ground truth: written pitch C4-C5 (treble 8vb clef, sounding C3-C4)
GROUND_TRUTH_WRITTEN = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
GROUND_TRUTH_SOUNDING = ["C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4"]
GROUND_TRUTH_MIDI_SOUNDING = [48, 50, 52, 53, 55, 57, 59, 60]  # C3-C4
GROUND_TRUTH_MIDI_WRITTEN = [60, 62, 64, 65, 67, 69, 71, 72]   # C4-C5

AUDIO_PATH = Path("/home/blt/Projects/GuitarTranscription/bechmarkaudio/audioCMajor.mp3")
CHECKPOINT = Path("/home/blt/Projects/GuitarTranscription/checkpoints/best_model.pt")
OUTPUT_DIR = Path("output/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}


def compare_notes(written_midi_detected, label):
    """Compare detected written-pitch MIDI numbers against ground truth.

    Uses pitch-class matching: for each GT note, check if that pitch appears
    in the detected set. Also tracks extra/missing notes.
    """
    detected_set = list(written_midi_detected)  # mutable copy for consumption
    matches = []

    for i, gt_midi in enumerate(GROUND_TRUTH_MIDI_WRITTEN):
        gt_note = GROUND_TRUTH_WRITTEN[i]
        # Find the first occurrence of this pitch in detected
        found_idx = None
        for j, det_midi in enumerate(detected_set):
            if det_midi == gt_midi:
                found_idx = j
                break

        if found_idx is not None:
            det_note = librosa.midi_to_note(detected_set[found_idx])
            matches.append({
                "position": i + 1,
                "expected": gt_note,
                "detected": det_note,
                "match": True
            })
            detected_set.pop(found_idx)  # consume this detection
        else:
            # Find closest detected note if any remain
            if detected_set:
                # Use the first remaining unmatched note
                det_note = librosa.midi_to_note(detected_set[0])
                matches.append({
                    "position": i + 1,
                    "expected": gt_note,
                    "detected": det_note,
                    "match": False
                })
                detected_set.pop(0)
            else:
                matches.append({
                    "position": i + 1,
                    "expected": gt_note,
                    "detected": "MISSING",
                    "match": False
                })

    # Remaining unmatched detections are extras
    extra_notes = [librosa.midi_to_note(m) for m in detected_set]

    correct = sum(1 for m in matches if m["match"])
    accuracy = correct / len(GROUND_TRUTH_WRITTEN) * 100
    return matches, correct, accuracy, extra_notes


# ============================================================
# APPROACH 1: detect_pitches.py (iteration 1 - pyin)
# ============================================================
print("=" * 60)
print("APPROACH 1: detect_pitches.py (librosa pyin)")
print("=" * 60)

from detect_pitches import detect_pitches, rows_to_note_segments, write_midi as write_midi_v1

rows, frame_step_sec = detect_pitches(AUDIO_PATH)
segments = rows_to_note_segments(rows, frame_step_sec, min_note_duration=0.05)

midi_out_1 = OUTPUT_DIR / "iter1_pyin.mid"
write_midi_v1(segments, midi_out_1)

# These are SOUNDING pitches (C3-C4 range for guitar)
sounding_midi_1 = [s["midi"] for s in segments]
sounding_notes_1 = [librosa.midi_to_note(m) for m in sounding_midi_1]
# Apply +12 semitone offset for written pitch
written_midi_1 = [m + 12 for m in sounding_midi_1]
written_notes_1 = [librosa.midi_to_note(m) for m in written_midi_1]

print(f"  Sounding notes detected: {sounding_notes_1}")
print(f"  Written notes (+12): {written_notes_1}")
print(f"  Note count: {len(segments)} (expected: 8)")
print(f"  MIDI output: {midi_out_1}")

matches_1, correct_1, accuracy_1, extras_1 = compare_notes(written_midi_1, "iter1")

results["iteration1"] = {
    "name": "Iteration 1 (librosa pyin)",
    "short_name": "pyin",
    "sounding_notes": sounding_notes_1,
    "written_notes": written_notes_1,
    "sounding_midi": sounding_midi_1,
    "written_midi": written_midi_1,
    "note_count": len(segments),
    "matches": matches_1,
    "correct": correct_1,
    "accuracy": accuracy_1,
    "midi_file": str(midi_out_1),
    "extra_notes": extras_1,
}

print(f"  Accuracy: {accuracy_1:.1f}% ({correct_1}/8)")
for m in matches_1:
    status = "YES" if m["match"] else "NO"
    print(f"    Note {m['position']}: expected {m['expected']}, got {m['detected']} -> {status}")
if extras_1:
    print(f"  Extra notes: {extras_1}")
print()

# ============================================================
# APPROACH 2: model/predict.py (iteration 2 - ML)
# ============================================================
print("=" * 60)
print("APPROACH 2: model/predict.py (CNN+BiGRU ML model)")
print("=" * 60)

import torch
from model.predict import load_cqt, pianoroll_to_notes, write_midi as write_midi_v2
from model.network import GuitarTranscriptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# Load model with strict=False to handle missing articulation_head in old checkpoint
mel = load_cqt(AUDIO_PATH)
mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
model = GuitarTranscriptionModel().to(device)
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
with torch.no_grad():
    frame_logits, onset_logits, art_logits = model(mel_t)
    frame_prob = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()
    onset_prob = torch.sigmoid(onset_logits).squeeze(0).cpu().numpy()
    art_prob = torch.sigmoid(art_logits).squeeze(0).cpu().numpy()
notes_2 = pianoroll_to_notes(frame_prob, onset_prob, art_prob=art_prob)

midi_out_2 = OUTPUT_DIR / "iter2_ml.mid"
write_midi_v2(notes_2, midi_out_2)

# ML model: GUITAR_TUNING[string] + fret = sounding MIDI pitch
# For written pitch comparison, add +12
sounding_midi_2 = [n["midi"] for n in notes_2]
sounding_notes_2 = [librosa.midi_to_note(m) for m in sounding_midi_2]
written_midi_2 = [m + 12 for m in sounding_midi_2]
written_notes_2 = [librosa.midi_to_note(m) for m in written_midi_2]

print(f"  Sounding notes detected: {sounding_notes_2}")
print(f"  Written notes (+12): {written_notes_2}")
print(f"  Note count: {len(notes_2)} (expected: 8)")
print(f"  MIDI output: {midi_out_2}")

for n in notes_2:
    print(f"    String {n['string']}, Fret {n['fret']} -> MIDI {n['midi']} "
          f"({librosa.midi_to_note(n['midi'])}) [{n.get('articulation', 'pluck')}]")

matches_2, correct_2, accuracy_2, extras_2 = compare_notes(written_midi_2, "iter2")

results["iteration2"] = {
    "name": "Iteration 2 (CNN+BiGRU ML)",
    "short_name": "ML",
    "sounding_notes": sounding_notes_2,
    "written_notes": written_notes_2,
    "sounding_midi": sounding_midi_2,
    "written_midi": written_midi_2,
    "note_count": len(notes_2),
    "matches": matches_2,
    "correct": correct_2,
    "accuracy": accuracy_2,
    "midi_file": str(midi_out_2),
    "string_fret_info": [{"string": n["string"], "fret": n["fret"], "midi": n["midi"],
                           "articulation": n.get("articulation", "pluck")} for n in notes_2],
    "extra_notes": extras_2,
}

print(f"  Accuracy: {accuracy_2:.1f}% ({correct_2}/8)")
for m in matches_2:
    status = "YES" if m["match"] else "NO"
    print(f"    Note {m['position']}: expected {m['expected']}, got {m['detected']} -> {status}")
if extras_2:
    print(f"  Extra notes: {extras_2}")
print()

# ============================================================
# APPROACH 3: transcribe_smart.py (smart pipeline)
# ============================================================
print("=" * 60)
print("APPROACH 3: transcribe_smart.py (ML + music21)")
print("=" * 60)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "transcribe_smart",
    str(PROJECT_ROOT / "test" / "transcribe_smart.py"),
)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)

_ml_model_notes = _ts._ml_model_notes
build_music21_stream = _ts.build_music21_stream
detect_key_fn = _ts.detect_key
filter_by_key = _ts.filter_by_key
assign_fingering = _ts.assign_fingering
write_midi_v3 = _ts.write_midi

# Step 1: ML backend - reuse the model we already loaded (with strict=False)
print("  [1/4] Detecting pitches with ML backend...")
# Reuse frame_prob/onset_prob/art_prob from approach 2 (same model, same audio)
raw_notes_3 = pianoroll_to_notes(frame_prob, onset_prob, art_prob=art_prob)
print(f"  Raw notes detected: {len(raw_notes_3)}")

# Step 2: music21 analysis
print("  [2/4] Running music21 analysis...")
m21_stream = build_music21_stream(raw_notes_3)
detected_key = detect_key_fn(m21_stream)
print(f"  Detected key: {detected_key} (confidence: {detected_key.correlationCoefficient:.3f})")

# Step 3: Filter by key
kept, dropped = filter_by_key(raw_notes_3, detected_key, tolerance_semitones=1)
print(f"  Notes kept: {len(kept)} (dropped {len(dropped)} off-key)")
if dropped:
    dropped_names = sorted(set(librosa.midi_to_note(n["midi"]) for n in dropped))
    print(f"  Dropped note names: {dropped_names}")

# Step 4: Fingering
fingered_notes = assign_fingering(kept)
print(f"  Notes with fingering: {len(fingered_notes)}")

midi_out_3 = OUTPUT_DIR / "iter3_smart.mid"
write_midi_v3(fingered_notes, midi_out_3)

# Notes from ML model are sounding pitches
sounding_midi_3 = [n["midi"] for n in fingered_notes]
sounding_notes_3 = [librosa.midi_to_note(m) for m in sounding_midi_3]
written_midi_3 = [m + 12 for m in sounding_midi_3]
written_notes_3 = [librosa.midi_to_note(m) for m in written_midi_3]

print(f"  Sounding notes: {sounding_notes_3}")
print(f"  Written notes (+12): {written_notes_3}")
print(f"  Note count: {len(fingered_notes)} (expected: 8)")
print(f"  MIDI output: {midi_out_3}")

matches_3, correct_3, accuracy_3, extras_3 = compare_notes(written_midi_3, "iter3")

results["iteration3"] = {
    "name": "Iteration 3 (Smart Pipeline)",
    "short_name": "Smart",
    "sounding_notes": sounding_notes_3,
    "written_notes": written_notes_3,
    "sounding_midi": sounding_midi_3,
    "written_midi": written_midi_3,
    "note_count": len(fingered_notes),
    "matches": matches_3,
    "correct": correct_3,
    "accuracy": accuracy_3,
    "midi_file": str(midi_out_3),
    "key_detected": str(detected_key),
    "key_confidence": float(detected_key.correlationCoefficient),
    "filtered_notes": len(dropped),
    "raw_notes_count": len(raw_notes_3),
    "extra_notes": extras_3,
}

print(f"  Accuracy: {accuracy_3:.1f}% ({correct_3}/8)")
for m in matches_3:
    status = "YES" if m["match"] else "NO"
    print(f"    Note {m['position']}: expected {m['expected']}, got {m['detected']} -> {status}")
if extras_3:
    print(f"  Extra notes: {extras_3}")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
for key in ["iteration1", "iteration2", "iteration3"]:
    r = results[key]
    print(f"  {r['name']}: {r['accuracy']:.1f}% ({r['correct']}/8), {r['note_count']} notes detected")

# Save results as JSON
output_json = OUTPUT_DIR / "comparison_results.json"
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_json}")
