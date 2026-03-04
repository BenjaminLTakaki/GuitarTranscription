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

A **polyphonic** guitar transcription model trained on the
[GAPS dataset](https://arxiv.org/abs/2408.08653) (300+ solo guitar performances
with aligned audio and MIDI).

### Architecture

| Component | Details |
|-----------|---------|
| Input | 229-bin log-mel spectrogram (22 050 Hz, hop 512) |
| Encoder | 3 × ConvBlock (32→64→128 channels, freq-axis pooling) |
| Sequence | 2-layer bidirectional GRU (256 hidden) |
| Heads | Frame head + Onset head (49 pitches, E2–E6), each with hidden layer |
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

Clone or download the [GAPS repo](https://github.com/cwitkowitz/GAPS) into the
project root so the directory structure is:

```
GuitarTranscription/
  GAPS/
    audio/
    midi/
    gaps_metadata_with_splits.csv
```

### Train

```bash
python -m model.train --epochs 50 --batch-size 16 --lr 6e-4
# Optional: use cosine annealing scheduler
python -m model.train --epochs 50 --batch-size 16 --lr 6e-4 --scheduler cosine
```

Checkpoints are saved to `checkpoints/`. The best model (by test F1) is
`checkpoints/best_model.pt`.

### Predict (inference)

```bash
python -m model.predict path/to/guitar.wav -o output/transcription_ml.mid
```

### Project structure

```
detect_pitches.py       ← Iteration 1 (rule-based)
model/
  constants.py          ← Shared hyperparameters
  dataset.py            ← GAPS dataset loader (mel specs + piano rolls)
  network.py            ← CNN + BiGRU model
  evaluate.py           ← Frame-level & note-level metrics
  train.py              ← Training loop with checkpointing
  predict.py            ← Inference: audio → MIDI
checkpoints/            ← Saved model weights (created by train.py)
output/                 ← Generated MIDI files
```
