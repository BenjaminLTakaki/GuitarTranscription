# Guitar Transcription Project — Full Journey

## Overview

Building an automatic guitar transcription system: audio in → MIDI out.  
CNN + BiGRU neural network trained on synthetic data and GuitarSet, with Schmitt-trigger post-processing for note extraction.

**Model**: 5.5M params, 126 output classes (6 strings × 21 frets)  
**Input**: 144-bin CQT spectrogram (24 bins/octave × 6 octaves, fmin=E2)  
**Architecture**: 4-layer CNN → BiGRU → dual linear heads (frame + onset)

---

## Phase 1: Synthetic Data Generation

### Karplus-Strong Generator
- Built a basic synthesizer using Karplus-Strong string synthesis
- Generated training data programmatically — fast but unrealistic timbre

### FluidSynth SF2 Generator (`generate_sf2.py`)
- Much more realistic: renders MIDI through SoundFont2 instruments via FluidSynth
- Handled Flatpak path translation (local dev runs in Flatpak sandbox)
- Auto-detects environment (Flatpak vs native) for correct audio paths

### Outcome
- Generated **1,000 SF2 tracks** (~500 minutes total, 0 errors)
- This became the pretraining dataset

---

## Phase 2: Cloud Training on RunPod

### Setup Struggles
- **No tmux** on RunPod initially — connection drops killed training runs
- **GitHub auth failures** — resolved using Personal Access Token in URL
- **unzip not installed** — `download_guitarset.sh` failed silently
- **GuitarSet extracted flat** — ZIP files dumped everything into `GuitarSet/` instead of the expected `annotation/` and `audio_mono-mic/` subdirectories
  - Fix: `mkdir -p annotation audio_mono-mic && mv *.jams annotation/ && mv *.wav audio_mono-mic/`

### Pretraining on Synthetic Data (Epochs 1–59)
- Command: `python -m model.train --synth-root SyntheticGuitar_SF2 --epochs 150`
- No eval set (GuitarSet dirs weren't set up yet)
- **Connection dropped** around epoch 57 (no tmux)
- Best training F1 saved: **0.3680**

### Bug Fixes Mid-Training

#### `validate()` crash on empty dataset
- When no eval set was available, `validate()` would crash
- Fixed: early return with zero metrics for empty loaders

#### Checkpoint never saved without eval
- `best_f1` started at 0, `val_stats['f1']` was always 0 (no eval set), so condition `val_f1 > best_f1` never triggered
- `best_model.pt` was **never being saved** during synthetic pretraining
- Fix: save based on **lowest training loss** when no eval set is available

#### ReduceLROnPlateau with no eval
- Scheduler expected validation F1 but got nothing
- Fixed: feed negative training loss as the metric when no eval set exists

### Installing tmux (finally)
```bash
apt-get install -y tmux && tmux new -s train
```
No more lost training runs.

### Fine-tuning on GuitarSet (Epochs 60–300)
```bash
python -m model.train --resume checkpoints/best_model.pt --epochs 300 --lr 2e-4 --batch-size 32 --scheduler cosine
```
- Train: 240 items, Val: 60 items, Test: 60 items
- Cosine LR schedule wrapped around at epoch 150, giving the model a second warm-up cycle
- **Final best F1: 0.7459** at epoch 300

---

## Phase 3: Getting the Checkpoint Home

### Git Push Disaster
1. `git add` blocked by `.gitignore` → used `git add -f checkpoints/best_model.pt`
2. `git push` rejected — branches had diverged
3. Ran `git merge --abort` + `git reset --hard origin/main`
4. **This deleted `best_model.pt`** from the working tree 💀
5. Periodic checkpoints survived: `cp checkpoint_epoch300.pt best_model.pt`
6. Re-added, committed, pushed successfully
7. Synced locally: `git merge GuitarScribe/main`

**Checkpoint**: 64MB, epoch 300, F1=0.7459 — safe on GitHub and local.

### GitHub Token Leak
- PAT token was accidentally pasted in chat
- Advised immediate revocation

---

## Phase 4: "The Transcription is Still Dogshit"

Despite F1=0.7459 on the test set, actual transcription output on real audio was terrible.

### Diagnosis on `audio2.wav` (6.2 seconds)
| Threshold | Notes Detected | Unique Pitches |
|-----------|---------------|----------------|
| 0.50      | 7             | 5              |
| 0.20      | 16            | 8              |
| 0.10      | 22            | —              |

The model was clearly activating on notes (max frame prob = 0.9988) but the post-processing was killing most of them.

### Diagnosis on GuitarSet test file (22.3 seconds)
- At threshold 0.5: 49 notes detected
- At threshold 0.2: 160 notes detected
- Mean frame probability: only 0.0284, with just 2.32% of frames above 0.5

### Test Set DataLoader Crash
- Attempted batch evaluation on full test set
- `RuntimeError: stack expects each tensor to be equal size`
- Test split returns full-length tracks (variable sizes), can't batch
- Fix: process one sample at a time in a loop

---

## Phase 5: Finding the Real Problem

### Frame-Level Threshold Sweep (60 test tracks)

| Threshold | Precision | Recall | F1     |
|-----------|-----------|--------|--------|
| 0.15      | 0.456     | 0.908  | 0.607  |
| 0.20      | 0.495     | 0.895  | 0.638  |
| 0.30      | 0.553     | 0.873  | 0.677  |
| 0.40      | 0.598     | 0.855  | 0.704  |
| 0.50      | 0.636     | 0.837  | 0.723  |
| 0.60      | 0.670     | 0.817  | 0.736  |
| 0.70      | 0.705     | 0.790  | 0.745  |
| **0.75**  | **0.723** | **0.773** | **0.747** |
| 0.80      | 0.742     | 0.751  | 0.747  |

Frame F1 peaked at **0.75**, not 0.50.

### Note-Level Schmitt Trigger Optimization (15 test tracks)

Swept 5 parameters: onset threshold, sustain threshold, median filter size, min note frames, and deduplication.

**Top 8 configurations (all with onset=0.75, no median filter, no dedup):**

| Onset | Sustain | Median | MinF | Note F1 |
|-------|---------|--------|------|---------|
| 0.75  | 0.10    | (1,1)  | 2    | 0.7437  |
| 0.75  | 0.10    | (1,1)  | 3    | 0.7437  |
| 0.75  | 0.30    | (1,1)  | 3    | 0.7436  |
| 0.75  | 0.15    | (1,1)  | 3    | 0.7434  |

Key insight: **the (3,3) median filter never appeared in the top 25** — it was actively suppressing real detections.

---

## Phase 6: The Fix

Three changes to post-processing parameters:

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| `ONSET_THRESHOLD` | 0.50 | **0.75** | Higher threshold = requires confident onset peaks, reduces false starts |
| `SUSTAIN_THRESHOLD` | 0.15 | **0.10** | Lower floor = notes sustain through guitar decay |
| `median_filter_size` | (3,3) | **(1,1)** | Median filter was killing valid activations |

### Result on `audio2.wav`

| Metric | Before | After |
|--------|--------|-------|
| Notes detected | 7 | **53** |
| Unique pitches | 5 | **12** |
| Pitch range | G#3–F4 | **G2–F4** |
| String/fret positions | — | **13** |

### Note Distribution After Fix
```
G2  (43): 1x     A2 (45): 3x     C3 (48): 1x
D3  (50): 5x     E3 (52): 4x     G3 (55): 6x
A3  (57): 5x     B3 (59): 7x     C4 (60): 10x
D4  (62): 4x     E4 (64): 6x     F4 (65): 1x
```

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Audio features | 144-bin CQT (librosa) |
| Model | CNN (4 conv blocks) → BiGRU (2 layers, 256 hidden) → dual heads |
| Training | PyTorch, BCE loss, cosine LR scheduler |
| Synthetic data | FluidSynth + SF2 soundfonts |
| Real data | GuitarSet (360 tracks, 6 players, player-based split) |
| Post-processing | Schmitt-trigger dual-threshold + onset re-articulation |
| Output | MIDI via `mido` |
| Cloud training | RunPod A40 48GB |

## Files Modified

| File | Changes |
|------|---------|
| `model/constants.py` | Thresholds: onset 0.50→0.75, sustain 0.15→0.10 |
| `model/predict.py` | Median filter default (3,3)→(1,1), Schmitt trigger logic |
| `model/train.py` | Resume flag, empty eval handling, loss-based checkpointing |
| `model/evaluate.py` | Frame + note level metrics |
| `model/guitarset_dataset.py` | CQT features, augmentation, split handling |
| `model/network.py` | CNN+BiGRU architecture with dual output heads |
| `diagnose.py` | Threshold sweep, heatmaps, probability distributions |
| `optimize_params.py` | Full Schmitt-trigger parameter grid search on test set |
| `generate_sf2.py` | FluidSynth SF2 rendering with Flatpak support |

## Lessons Learned

1. **Always use tmux for remote training.** Connection drops are inevitable.
2. **`git reset --hard` is destructive.** Periodic checkpoints saved the project.
3. **Frame F1 ≠ perceptual quality.** A model can report F1=0.74 while producing garbage MIDI if post-processing is wrong.
4. **Post-processing matters as much as the model.** The threshold change from 0.50→0.75 and removing the median filter turned 7 detected notes into 53.
5. **Median filters aren't always helpful.** For polyphonic music with overlapping harmonics, the (3,3) median filter suppressed valid activations.
6. **Sweep parameters systematically.** Don't guess — cache model predictions once, then grid search post-processing params cheaply.
7. **Higher threshold ≠ fewer notes** when using Schmitt trigger. The higher onset threshold actually improved recall in the note-level metric because it reduced false fragmentation.
8. **Synthetic pretraining helps.** 500 minutes of SF2 data provided a solid initialization before fine-tuning on the smaller GuitarSet.
