#!/usr/bin/env python3
"""Run all three transcription approaches on chord audio and compute note-level metrics."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import mido
import numpy as np

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
AUDIO_PATH = Path("test/audioChords.wav")
JAMS_PATH = Path("test/audioChords.jams")
CHECKPOINT = Path("/home/blt/Projects/GuitarTranscription/checkpoints/best_model.pt")
OUTPUT_DIR = Path("test/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHORD_DUR = 1.5
GAP = 0.2
ONSET_TOL = 0.05  # 50ms onset tolerance

# Load ground truth from JAMS
with open(JAMS_PATH) as f:
    jams = json.load(f)

gt_notes = []
for ann in jams["annotations"]:
    if ann["namespace"] == "note_midi":
        string_idx = ann["sandbox"]["string_index"]
        for obs in ann["data"]:
            gt_notes.append({
                "onset": obs["time"],
                "offset": obs["time"] + obs["duration"],
                "midi": int(obs["value"]),
                "string": string_idx,
            })
gt_notes.sort(key=lambda n: (n["onset"], n["midi"]))

print(f"Ground truth: {len(gt_notes)} notes")
for n in gt_notes:
    print(f"  t={n['onset']:.2f}s  MIDI {n['midi']} ({librosa.midi_to_note(n['midi'])})"
          f"  string {n['string']}")

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_note_metrics(predicted, ground_truth, onset_tol=ONSET_TOL):
    """Compute note-level precision, recall, F1 with onset tolerance.

    A predicted note is a true positive if a ground truth note with the same
    MIDI pitch exists within onset_tol seconds of the predicted onset.
    """
    gt_matched = [False] * len(ground_truth)
    pred_matched = [False] * len(predicted)

    # Greedy matching: for each predicted note, find closest unmatched GT
    for pi, pn in enumerate(predicted):
        best_dist = float("inf")
        best_gi = None
        for gi, gn in enumerate(ground_truth):
            if gt_matched[gi]:
                continue
            if pn["midi"] != gn["midi"]:
                continue
            dist = abs(pn["onset"] - gn["onset"])
            if dist <= onset_tol and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi is not None:
            gt_matched[best_gi] = True
            pred_matched[pi] = True

    tp = sum(pred_matched)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Build per-GT-note match results
    note_matches = []
    for gi, gn in enumerate(ground_truth):
        if gt_matched[gi]:
            # Find which predicted note matched
            matched_pred = None
            for pi, pn in enumerate(predicted):
                if pred_matched[pi] and pn["midi"] == gn["midi"]:
                    if abs(pn["onset"] - gn["onset"]) <= onset_tol:
                        matched_pred = pn
                        break
            note_matches.append({
                "gt_onset": gn["onset"],
                "gt_midi": gn["midi"],
                "gt_note": librosa.midi_to_note(gn["midi"]),
                "detected": librosa.midi_to_note(matched_pred["midi"]) if matched_pred else "?",
                "match": True,
            })
        else:
            note_matches.append({
                "gt_onset": gn["onset"],
                "gt_midi": gn["midi"],
                "gt_note": librosa.midi_to_note(gn["midi"]),
                "detected": "MISSING",
                "match": False,
            })

    # Extra predictions (false positives)
    extras = []
    for pi, pn in enumerate(predicted):
        if not pred_matched[pi]:
            extras.append({
                "onset": pn["onset"],
                "midi": pn["midi"],
                "note": librosa.midi_to_note(pn["midi"]),
            })

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "note_matches": note_matches,
        "extras": extras,
    }


def extract_notes_from_midi(midi_path):
    """Extract (onset, midi) pairs from a MIDI file."""
    mid = mido.MidiFile(str(midi_path))
    notes = []
    current_time = 0.0
    tempo = 500000  # default 120 BPM
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if hasattr(msg, "time"):
            current_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "note_on" and msg.velocity > 0:
            notes.append({
                "onset": round(current_time, 4),
                "midi": msg.note,
            })
    return notes


results = {}

# ============================================================
# APPROACH 1: detect_pitches.py (pyin — monophonic)
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 1: detect_pitches.py (librosa pyin)")
print("=" * 60)

from detect_pitches import detect_pitches, rows_to_note_segments, write_midi as write_midi_v1

rows, frame_step_sec = detect_pitches(AUDIO_PATH)
segments = rows_to_note_segments(rows, frame_step_sec, min_note_duration=0.05)

midi_out_1 = OUTPUT_DIR / "chords_iter1.mid"
write_midi_v1(segments, midi_out_1)

pred_1 = [{"onset": s["start"], "midi": s["midi"]} for s in segments]
detected_notes_1 = [librosa.midi_to_note(s["midi"]) for s in segments]

print(f"  Detected {len(segments)} notes: {detected_notes_1}")
metrics_1 = compute_note_metrics(pred_1, gt_notes)
print(f"  TP={metrics_1['tp']} FP={metrics_1['fp']} FN={metrics_1['fn']}")
print(f"  Precision: {metrics_1['precision']:.3f}")
print(f"  Recall:    {metrics_1['recall']:.3f}")
print(f"  F1:        {metrics_1['f1']:.3f}")

results["iteration1"] = {
    "name": "Iteration 1 (librosa pyin)",
    "short_name": "pyin",
    "note_count": len(segments),
    "detected_notes": detected_notes_1,
    "detected_midi": [s["midi"] for s in segments],
    "metrics": {k: v for k, v in metrics_1.items() if k not in ("note_matches", "extras")},
    "note_matches": metrics_1["note_matches"],
    "extras": metrics_1["extras"],
    "midi_file": str(midi_out_1),
}

# ============================================================
# APPROACH 2: model/predict.py (ML model)
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 2: model/predict.py (CNN+BiGRU)")
print("=" * 60)

import torch
from model.predict import load_cqt, pianoroll_to_notes, write_midi as write_midi_v2
from model.network import GuitarTranscriptionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

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

midi_out_2 = OUTPUT_DIR / "chords_iter2.mid"
write_midi_v2(notes_2, midi_out_2)

pred_2 = [{"onset": n["start"], "midi": n["midi"]} for n in notes_2]
detected_notes_2 = [librosa.midi_to_note(n["midi"]) for n in notes_2]

print(f"  Detected {len(notes_2)} notes: {detected_notes_2}")
for n in notes_2:
    print(f"    t={n['start']:.2f}s  S{n['string']}F{n['fret']} -> MIDI {n['midi']} "
          f"({librosa.midi_to_note(n['midi'])}) [{n.get('articulation', 'pluck')}]")

metrics_2 = compute_note_metrics(pred_2, gt_notes)
print(f"  TP={metrics_2['tp']} FP={metrics_2['fp']} FN={metrics_2['fn']}")
print(f"  Precision: {metrics_2['precision']:.3f}")
print(f"  Recall:    {metrics_2['recall']:.3f}")
print(f"  F1:        {metrics_2['f1']:.3f}")

results["iteration2"] = {
    "name": "Iteration 2 (CNN+BiGRU ML)",
    "short_name": "ML",
    "note_count": len(notes_2),
    "detected_notes": detected_notes_2,
    "detected_midi": [n["midi"] for n in notes_2],
    "metrics": {k: v for k, v in metrics_2.items() if k not in ("note_matches", "extras")},
    "note_matches": metrics_2["note_matches"],
    "extras": metrics_2["extras"],
    "midi_file": str(midi_out_2),
    "string_fret_info": [{"string": n["string"], "fret": n["fret"], "midi": n["midi"],
                           "onset": n["start"]} for n in notes_2],
}

# ============================================================
# APPROACH 3: transcribe_smart.py (ML + music21)
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 3: transcribe_smart.py (ML + music21)")
print("=" * 60)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "transcribe_smart",
    str(PROJECT_ROOT / "test" / "transcribe_smart.py"),
)
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)

build_music21_stream = _ts.build_music21_stream
detect_key_fn = _ts.detect_key
filter_by_key = _ts.filter_by_key
assign_fingering = _ts.assign_fingering
write_midi_v3 = _ts.write_midi

# Reuse ML model output
raw_notes_3 = pianoroll_to_notes(frame_prob, onset_prob, art_prob=art_prob)
print(f"  Raw notes from ML: {len(raw_notes_3)}")

m21_stream = build_music21_stream(raw_notes_3)
detected_key = detect_key_fn(m21_stream)
print(f"  Detected key: {detected_key} (confidence: {detected_key.correlationCoefficient:.3f})")

kept, dropped = filter_by_key(raw_notes_3, detected_key, tolerance_semitones=1)
print(f"  Kept: {len(kept)}, Dropped: {len(dropped)}")
if dropped:
    dropped_names = sorted(set(librosa.midi_to_note(n["midi"]) for n in dropped))
    print(f"  Dropped: {dropped_names}")

fingered = assign_fingering(kept)

midi_out_3 = OUTPUT_DIR / "chords_iter3.mid"
write_midi_v3(fingered, midi_out_3)

pred_3 = [{"onset": n["start"], "midi": n["midi"]} for n in fingered]
detected_notes_3 = [librosa.midi_to_note(n["midi"]) for n in fingered]

print(f"  Detected {len(fingered)} notes (after filter): {detected_notes_3}")
metrics_3 = compute_note_metrics(pred_3, gt_notes)
print(f"  TP={metrics_3['tp']} FP={metrics_3['fp']} FN={metrics_3['fn']}")
print(f"  Precision: {metrics_3['precision']:.3f}")
print(f"  Recall:    {metrics_3['recall']:.3f}")
print(f"  F1:        {metrics_3['f1']:.3f}")

results["iteration3"] = {
    "name": "Iteration 3 (Smart Pipeline)",
    "short_name": "Smart",
    "note_count": len(fingered),
    "detected_notes": detected_notes_3,
    "detected_midi": [n["midi"] for n in fingered],
    "metrics": {k: v for k, v in metrics_3.items() if k not in ("note_matches", "extras")},
    "note_matches": metrics_3["note_matches"],
    "extras": metrics_3["extras"],
    "midi_file": str(midi_out_3),
    "key_detected": str(detected_key),
    "key_confidence": float(detected_key.correlationCoefficient),
    "filtered_notes": len(dropped),
    "raw_notes_count": len(raw_notes_3),
}

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Approach':<30} {'Notes':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
      f"{'Prec':>6} {'Rec':>6} {'F1':>6}")
print("-" * 76)
for key in ["iteration1", "iteration2", "iteration3"]:
    r = results[key]
    m = r["metrics"]
    print(f"{r['name']:<30} {r['note_count']:>6} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
          f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")

# Save results
output_json = OUTPUT_DIR / "chord_comparison_results.json"
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_json}")
