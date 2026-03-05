# GuitarTranscription

Automatic guitar transcription: audio → MIDI.

## Iteration 1 — Rule-based pitch detection

Uses `librosa.pyin` for **monophonic** pitch estimation.

```bash
python detect_pitches.py /path/to/guitar_audio.wav
```

Optional tuning parameters:

```bash
python detect_pitches.py input.wav -o output/my_take.mid --frame-length 2048 --hop-length 256 --min-note-duration 0.05
```

## Iteration 2 — ML model (CNN + BiGRU)

An **Automatic Tablature Transcription** model trained exclusively on the
[GuitarSet](https://guitarset.weebly.com/) dataset (360 excerpts of solo and
comping performances with per-string JAMS annotations).

### Architecture

| Component | Details |
|-----------|---------|
| Input | 144-bin log-CQT spectrogram (22 050 Hz, hop 512, 24 bins/octave) |
| Encoder | 3 × ConvBlock (32→64→128 channels, freq-axis pooling) |
| Sequence | 2-layer bidirectional GRU (256 hidden) |
| Heads | Frame head + Onset head — 126 tablature classes (6 strings × 21 frets) |
| Connection | Onset→Frame (onset sigmoid concatenated into frame head input) |
| Augmentation | Gain, frequency masking, time masking (SpecAugment-lite) |
| Loss | BCE with logits, positive-class weighting, onset weight = 1.0 |

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU support install the CUDA version of PyTorch instead:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Dataset

Download GuitarSet using the included helper script:

```bash
chmod +x download_guitarset.sh
./download_guitarset.sh
```

This fetches annotation and mono-mic audio from Zenodo and extracts them into
the expected layout:

```
GuitarSet/
  annotation/*.jams
  audio_mono-mic/*.wav
```

### Train

```bash
python -m model.train --epochs 50 --batch-size 16 --lr 6e-4
# Optional: use cosine annealing scheduler
python -m model.train --epochs 50 --batch-size 16 --lr 6e-4 --scheduler cosine
# Resume from a checkpoint
python -m model.train --resume checkpoints/best_model.pt --epochs 150 --lr 6e-4
```

#### Synthetic data pretraining

Generate synthetic guitar audio (Karplus-Strong synthesis) to augment the
small GuitarSet dataset, then pretrain + fine-tune:

```bash
# 1. Generate synthetic data (e.g. 500 tracks × 30 seconds ≈ 4 hours)
python generate_synthetic.py --num-tracks 500 --duration 30

# 2. Pretrain on synthetic data
python -m model.train --synth-root SyntheticGuitar --epochs 50 --lr 6e-4

# 3. Fine-tune on real GuitarSet data
python -m model.train --resume checkpoints/best_model.pt --epochs 100 --lr 2e-4
```

Checkpoints are saved to `checkpoints/`. The best model (by test F1) is
`checkpoints/best_model.pt`.

### Predict (inference)

```bash
python -m model.predict path/to/guitar.wav -o output/transcription_ml.mid
```

### Project structure

```
detect_pitches.py           ← Iteration 1 (rule-based)
generate_synthetic.py       ← Synthetic guitar data generator (Karplus-Strong)
download_guitarset.sh       ← Downloads GuitarSet from Zenodo
model/
  constants.py              ← Shared hyperparameters
  guitarset_dataset.py      ← Primary dataset loader (CQT specs + tablature labels from JAMS)
  dataset.py                ← Legacy GAPS dataset loader (kept for compatibility)
  network.py                ← CNN + BiGRU model
  evaluate.py               ← Frame-level & note-level metrics
  train.py                  ← Training loop with checkpointing
  predict.py                ← Inference: audio → MIDI
checkpoints/                ← Saved model weights (created by train.py)
output/                     ← Generated MIDI files
```
