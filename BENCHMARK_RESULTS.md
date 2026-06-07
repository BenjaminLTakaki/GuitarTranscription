# Benchmark Results — GuitarSet Test Split

**Test split:** Player 05 (60 files)  
**Model:** CNN + 2-layer BiGRU, 126 tablature classes (6 strings × 21 frets)  
**Metric:** Frame-level Precision / Recall / F1 against JAMS ground truth  
**Frame threshold:** 0.65 (optimized via `optimize_params.py`)

---

## Summary

| System | Checkpoint | Precision | Recall | F1 | Δ F1 |
|--------|-----------|----------:|-------:|---:|-----:|
| v1 baseline (thr=0.50) | best_model_v1.pt — epoch 539 | 0.6961 | 0.7891 | **0.7374** | — |
| v1 baseline (thr=0.65) | best_model_v1.pt — epoch 539 | 0.7272 | 0.7570 | **0.7388** | +0.0014 |
| **v2: trainval + pitch-shift aug** | best_model_v2.pt — epoch ~651 | **0.7374** | **0.8236** | **0.7761** | **+0.0387** |
| + basic-pitch soft fusion (v1) | best_model_v1.pt | 0.6908 | 0.7935 | **0.7365** | −0.0009 |

> **v2 changes:** trained on players 00–04 (trainval split, +25% data), pitch-shift augmentation ±2 semitones, frame threshold 0.65, cosine LR decay from epoch 539→651.  
> **Frame threshold 0.65** found by `optimize_params.py --frame-only`.

---

## Baseline — Per-File Results

| File | F1 |
|------|----|
| 05_BN1-129-Eb_comp | 0.774 |
| 05_BN1-129-Eb_solo | 0.855 |
| 05_BN1-147-Gb_comp | 0.834 |
| 05_BN1-147-Gb_solo | 0.519 |
| 05_BN2-131-B_comp  | 0.766 |
| 05_BN2-131-B_solo  | 0.700 |
| 05_BN2-166-Ab_comp | 0.849 |
| 05_BN2-166-Ab_solo | 0.702 |
| 05_BN3-119-G_comp  | 0.718 |
| 05_BN3-119-G_solo  | 0.677 |
| 05_BN3-154-E_comp  | 0.817 |
| 05_BN3-154-E_solo  | 0.747 |
| 05_Funk1-114-Ab_comp | 0.786 |
| 05_Funk1-114-Ab_solo | 0.687 |
| 05_Funk1-97-C_comp | 0.812 |
| 05_Funk1-97-C_solo | 0.783 |
| 05_Funk2-108-Eb_comp | 0.712 |
| 05_Funk2-108-Eb_solo | 0.746 |
| 05_Funk2-119-G_comp | 0.696 |
| 05_Funk2-119-G_solo | 0.696 |
| 05_Funk3-112-C#_comp | 0.829 |
| 05_Funk3-112-C#_solo | 0.821 |
| 05_Funk3-98-A_comp | 0.731 |
| 05_Funk3-98-A_solo | 0.823 |
| 05_Jazz1-130-D_comp | 0.725 |
| 05_Jazz1-130-D_solo | 0.741 |
| 05_Jazz1-200-B_comp | 0.730 |
| 05_Jazz1-200-B_solo | 0.750 |
| 05_Jazz2-110-Bb_comp | 0.796 |
| 05_Jazz2-110-Bb_solo | 0.630 |
| 05_Jazz2-187-F#_comp | 0.788 |
| 05_Jazz2-187-F#_solo | 0.369 |
| 05_Jazz3-137-Eb_comp | 0.867 |
| 05_Jazz3-137-Eb_solo | 0.757 |
| 05_Jazz3-150-C_comp | 0.654 |
| 05_Jazz3-150-C_solo | 0.814 |
| 05_Rock1-130-A_comp | 0.367 |
| 05_Rock1-130-A_solo | 0.829 |
| 05_Rock1-90-C#_comp | **0.948** |
| 05_Rock1-90-C#_solo | 0.789 |
| 05_Rock2-142-D_comp | 0.837 |
| 05_Rock2-142-D_solo | 0.575 |
| 05_Rock2-85-F_comp  | 0.731 |
| 05_Rock2-85-F_solo  | 0.737 |
| 05_Rock3-117-Bb_comp | 0.877 |
| 05_Rock3-117-Bb_solo | 0.880 |
| 05_Rock3-148-C_comp | 0.400 |
| 05_Rock3-148-C_solo | 0.732 |
| 05_SS1-100-C#_comp | 0.828 |
| 05_SS1-100-C#_solo | 0.799 |
| 05_SS1-68-E_comp   | 0.851 |
| 05_SS1-68-E_solo   | 0.458 |
| 05_SS2-107-Ab_comp | 0.736 |
| 05_SS2-107-Ab_solo | 0.700 |
| 05_SS2-88-F_comp   | 0.738 |
| 05_SS2-88-F_solo   | 0.677 |
| 05_SS3-84-Bb_comp  | 0.624 |
| 05_SS3-84-Bb_solo  | 0.813 |
| 05_SS3-98-C_comp   | 0.855 |
| 05_SS3-98-C_solo   | 0.759 |

**Best:** Rock1-90-C# comp (0.948) &nbsp;|&nbsp; **Worst:** Rock1-130-A comp (0.367)

---

## Notes

- **basic-pitch fusion adds nothing** (−0.09% F1). The baseline model is already strong on GuitarSet domain; basic-pitch's pitch detection is weaker than the model's implicit pitch representation learned from ground-truth annotations.
- **Low-F1 outliers** (Rock1-130-A comp 0.367, Rock3-148-C comp 0.400, Jazz2-187-F# solo 0.369) are likely fast strumming or complex chord voicings where the model's Schmitt-trigger threshold cuts too aggressively.
- YourMT3 hybrid results will be added once the overnight run completes (~9 hours).
