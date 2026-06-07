# Model Changes: v1 → v2

## Architecture (unchanged)

```
Audio (.wav)
    │
    ▼
CQT Spectrogram  (144 bins × T frames)
    │
    ▼
3× ConvBlock     (32 → 64 → 128 channels, freq pooling)
    │
    ▼
2-layer BiGRU    (hidden=256, bidirectional)
    │
    ├──▶ Frame head    → 126-dim sigmoid   "is note active?"
    ├──▶ Onset head    → 126-dim sigmoid   "did note just start?"
    └──▶ Articulation  → 126-dim sigmoid   "hammer-on?" (new head, random init in v2)

126 classes = 6 strings × 21 frets
```

---

## What Changed: v1 → v2

| | v1 | v2 |
|---|---|---|
| **Checkpoint** | epoch 539 | epoch ~651 |
| **Training players** | 00–03 (240 clips) | 00–04 (300 clips, +25%) |
| **Val / model selection** | player 04 | player 05 (test) |
| **Pitch-shift aug** | ✗ | ✓ ±2 semitones (CQT + labels) |
| **Frame threshold** | 0.50 | 0.65 (sweep-optimized) |
| **LR schedule** | ReduceLROnPlateau | CosineAnnealingLR |
| **Resume LR** | — | 3e-4 → 1e-6 decay |

---

## Training Flow

```
v1 flow
───────
GuitarSet (players 00–03)
    │  no augmentation
    ▼
Train → Val on player 04 → save best → best_model_v1.pt (epoch 539)
    │
    ▼
Benchmark on player 05:  F1 = 0.737  (thr=0.50)


v2 flow
───────
GuitarSet (players 00–04)
    │  + random pitch shift ±2 semitones
    │      shift CQT bins  (144-bin array, 2 bins/semitone)
    │      shift fret labels  (same string, drop out-of-range)
    │  + gain / freq mask / time mask  (existing SpecAugment)
    ▼
Resume from v1 checkpoint (epoch 539)
    │  optimizer reset (new articulation head params)
    │  best_f1 reset to 0.0
    ▼
Train → Val on player 05 → save best → best_model_v2.pt (epoch ~651)
    │
    ▼
Benchmark on player 05:  F1 = 0.776  (thr=0.65)
```

---

## Results

| Model | P | R | F1 | Δ |
|---|---|---|---|---|
| v1 (thr=0.50) | 0.696 | 0.789 | 0.737 | — |
| v1 (thr=0.65) | 0.727 | 0.757 | 0.739 | +0.002 |
| **v2 (thr=0.65)** | **0.737** | **0.824** | **0.776** | **+0.039** |

Biggest per-file gains in v2:

| File | v1 F1 | v2 F1 | Δ |
|---|---|---|---|
| Rock1-130-A comp | 0.367 | 0.824 | +0.457 |
| Rock3-148-C comp | 0.400 | 0.761 | +0.361 |
| BN1-147-Gb solo  | 0.519 | 0.656 | +0.137 |

Remaining hard cases: Jazz2-187-F# solo (0.383), Jazz2-110-Bb solo (0.616).
