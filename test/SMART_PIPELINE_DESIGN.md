# Smart Guitar Transcription Pipeline — Design & Reasoning Document

> **What this file is:** A full record of every design decision, dead end, and
> conclusion reached while building `transcribe_smart.py`, including a head-to-head
> comparison against the baseline ML model.

---

## 1. The Starting Problem

The existing system has two weaknesses that the user explicitly identified:

1. **Pitch detection is imperfect.** The ML model (CQT-based CNN fine-tuned on
   GuitarSet) sometimes hallucinates notes, drops low-confidence frames, or
   stutters on sustain. `librosa.pyin` in `detect_pitches.py` is monophonic only
   and misses anything polyphonic.

2. **It does not understand guitar fingering.** A MIDI note like A4 (69) can be
   played on four different string/fret combinations on a standard-tuned guitar.
   The system outputs *a* position, not necessarily the *right* or most ergonomic
   one. A human guitarist would pick the position that flows from the previous
   note and anticipates the next.

---

## 2. First Idea: Add an LLM

The first instinct was to add a large language model (like Claude) to the pipeline
to reason about the audio and suggest better note interpretations.

### Why this sounds appealing

LLMs have absorbed enormous amounts of music theory text, tab notation, and chord
charts during training. In principle they can:

- Identify that a sequence of notes implies a key or scale.
- Suggest that a detected note is "out of place" given its neighbours.
- Reason about which guitar position makes ergonomic sense.

### Why it was rejected as the *primary* approach

| Concern | Detail |
|---|---|
| **LLMs cannot hear audio** | They process text/tokens, not waveforms. You would still need a separate pitch detector feeding them data. |
| **Latency and cost** | Every transcription would require an API round-trip, adding seconds and ongoing API cost. |
| **Non-determinism** | The same audio could produce different corrected tabs on different runs. Bad for iteration. |
| **Overkill for structured music theory** | Music theory rules are deterministic. A key has exactly 7 notes. Voice leading rules are finite. These do not need probabilistic reasoning. |
| **Better tools exist** | `music21` encodes all of this as hard logic, for free, locally. |

**Conclusion:** An LLM is better used as a *last resort* when structured approaches
fail (e.g. describing a complex chord voicing in natural language). For rule-based
music theory post-processing, `music21` is strictly superior.

---

## 3. The Chosen Approach: Music21 + Better Pitch Detection

### music21

`music21` is a MIT-developed Python library that implements music theory as
runnable code. Relevant capabilities used in this pipeline:

| Feature | How we use it |
|---|---|
| **Krumhansl-Schmuckler key detection** | Infer the tonal centre of the detected note sequence |
| **Scale membership testing** | Filter out notes that don't belong to the inferred key |
| **`chordify()`** | Collapse simultaneous or near-simultaneous notes into chord objects and name them |
| **`stream.Stream`** | A musical container that understands time, offsets, and duration — not just raw MIDI |

### Basic Pitch (Spotify)

Basic Pitch was proposed as a better pitch detector than `pyin` because:
- It is a **polyphonic** neural network (can detect multiple simultaneous notes).
- It is trained on real instrument recordings.
- It runs locally with no API dependency.

---

## 4. The Basic Pitch Dead End

### What happened

```
pip install basic-pitch
# ERROR: Could not find a version that satisfies the requirement
# tensorflow<2.15.1,>=2.4.1; platform_system != "Darwin" and python_version >= "3.11"
```

### Root cause analysis

The failure has two layers:

**Layer 1 — numpy source build failure (Python 3.13)**

`basic-pitch` versions up to 0.3.x pin `numpy<1.24`. NumPy 1.23.x has no
pre-built wheels for Python 3.13, so pip tries to build it from source. The
source build uses `pkg_resources` from an old bundled `setuptools` in pip's
isolated build environment, which calls `pkgutil.ImpImporter` — a class that was
**removed in Python 3.12** (and therefore also absent in Python 3.13).

```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

**Layer 2 — TensorFlow version ceiling (Python 3.13)**

`basic-pitch 0.4.0` (latest) relaxes the numpy constraint but adds a hard cap:

```
tensorflow<2.15.1,>=2.4.1  (for non-Mac, Python >=3.11)
```

TensorFlow added Python 3.12 support starting with version 2.16, and Python 3.13
support starting with version 2.17. No TensorFlow version below 2.15.1 ships
pre-built wheels for Python 3.12 or later. Since `basic-pitch` pins
`tensorflow>=2.4.1,<2.15.1`, every compatible TF version predates Python 3.12+
wheel availability. On this system, only TF 2.20.0+ has Python 3.13 wheels, but
basic-pitch rejects all of those (≥2.15.1).

This environment runs **Python 3.13.11** — squarely in the incompatible zone.

### What we tried

| Attempt | Result |
|---|---|
| `pip install basic-pitch` | numpy source build fails (`pkgutil.ImpImporter` removed in Python 3.12+) |
| `pip install tensorflow` first, then basic-pitch | TF 2.21.0 installed but basic-pitch rejects it (>2.15.1) |
| `pip install --no-build-isolation basic-pitch` | `numpy.distutils` deprecated / removed — build fails with distutils errors on Python 3.13 |
| `pip install basic-pitch==0.4.0` | Hard TF ceiling fails — only TF 2.20.0+ available for Python 3.13 |

### The stub

Rather than abandon the idea permanently, the code in `transcribe_smart.py`
includes a `--backend basic-pitch` flag with a `NotImplementedError` that clearly
explains the incompatibility and shows the exact code that would run once
`basic-pitch` fixes its TF dependency ceiling.

---

## 5. Why the Existing ML Model is Actually Better than Basic Pitch Here

This turned out to be the most important insight of the whole exercise.

Basic Pitch is a **general-purpose** polyphonic pitch detector trained on mixed
instrument data. The project's existing ML model is a **guitar-specific** CQT-based
CNN fine-tuned on the GuitarSet dataset — real recorded guitar performances with
per-string, per-fret ground truth annotations.

| Property | Basic Pitch | Project ML Model |
|---|---|---|
| Domain | General (mixed instruments) | Guitar only |
| Output | MIDI pitches | MIDI pitch + **string index + fret number** |
| Training data | Mixed recordings | GuitarSet (guitar-specific, F1=0.7843) |
| Onset tracking | Basic threshold | Schmitt-trigger dual-threshold (onset=0.75, sustain=0.10) |
| Polyphonic | Yes | Yes (per-string) |
| Python 3.13 compatible | No | Yes |

The ML model already knows which *string* a note is on, not just its pitch. That
is information Basic Pitch cannot provide at all — it only outputs MIDI numbers.
The fingering optimiser in `transcribe_smart.py` has to *guess* the string for
Basic Pitch output; for ML model output it already knows.

---

## 6. Pipeline Design Decisions

### 6.1 Key Detection: Why Krumhansl-Schmuckler?

The Krumhansl-Schmuckler algorithm correlates the pitch-class distribution of the
detected notes against probe tone profiles derived from listener studies. It
returns a correlation coefficient (0–1) indicating confidence.

We chose it over alternatives because:
- It is already implemented in `music21.stream.Stream.analyze('key')`.
- It works on MIDI note data, no audio needed — fits our post-processing position.
- Correlation coefficient gives a meaningful confidence signal (we log it).

### 6.2 Note Filtering: Tolerance Parameter

The key filter has a `--key-tolerance` parameter (default: 1 semitone).

- **Tolerance 0 (strict):** Only notes literally in the scale pass. Good for
  classical or diatonic pop. Aggressive — throws away chromatic passing tones.
- **Tolerance 1 (default):** Notes within 1 semitone of any scale tone also pass.
  This covers chromatic passing tones, blue notes, and minor-to-major ambiguity
  (e.g. a b3 in a major key context).
- **No filter (`--no-filter`):** Bypass entirely. Useful when the detected key is
  wrong or when transcribing atonal/modal music.

The rationale for defaulting to 1: guitar music regularly uses chromatic runs,
hammer-ons to non-scale tones, and blues-inflected bends. A strict filter would
aggressively drop legitimate notes.

### 6.3 Chord Labelling: `chordify()` Limitations

`music21.stream.Stream.chordify()` collapses notes with overlapping duration into
chord objects. On monophonic `pyin` output, this produces mostly single-note
"chords" labeled as `note`. On polyphonic ML model output, it correctly identifies
intervals and triads.

The output for `audio.wav` with the ML backend showed:
- `Perfect Fifth` — two-note power chord structure
- `Perfect Fifth with octave doublings` — power chord with octave
- `Perfect Twelfth` — compound fifth (octave + fifth)

These are characteristic of guitar playing (power chords), which validates that
the ML model's polyphonic output is being correctly interpreted.

### 6.4 Fingering Optimiser: Greedy Position Minimisation

The algorithm works as follows:

1. For each detected MIDI pitch, compute every valid (string, fret) pair on a
   standard-tuned guitar (E2=40, A2=45, D3=50, G3=55, B3=59, e4=64), frets 0–20.
2. Among the candidates, pick the one that minimises `|candidate_fret - prev_fret|`.
3. Tie-break toward lower frets (open strings preferred — they are acoustically
   different but physically easiest).
4. Update `prev_fret` and proceed to the next note.

This is a greedy heuristic, not a global optimum. A Viterbi/dynamic programming
approach would find the globally optimal sequence of positions, but the greedy
approach performs well in practice for monophonic lines and simple melodies.

For ML model output, steps 1–3 are skipped because the model already predicts the
string index. The optimiser still validates the fret is in range.

---

## 7. Performance Comparison: Smart Pipeline vs. Baseline

### Test: `audio.wav` (same file, same machine)

| Metric | Baseline ML (`model/predict.py`) | Smart ML+music21 | Smart pyin+music21 |
|---|---|---|---|
| Evaluation F1 (GuitarSet test set) | **0.7843** (epoch 539) | 0.7843 (same model) | N/A (no test-set eval) |
| Architecture | CNN (3 ConvBlocks) + BiGRU (2 layers, 256 hidden) + dual heads | Same model + music21 post-processing | librosa pyin + music21 |
| Parameters | 5,534,556 | 5,534,556 (same) | N/A |
| Output classes | 126 (6 strings × 21 frets) | 126 (same) | MIDI pitch only |
| Raw notes detected | 21 | 21 (same) | 28 |
| Notes after filtering | N/A (no filtering) | 21 (0 dropped) | 28 (0 dropped) |
| Key detection | None | **A major** (confidence: 0.783) | A major (confidence: 0.656) |
| Chord events identified | None | 29 | 36 |
| String/fret per note | Yes (model predicts string + fret) | Yes (model pass-through) | Yes (optimiser assigned) |
| Unique positions used | 11 | 11 | 14 |
| ASCII tab | No | **Yes** | **Yes** |
| MIDI output | Yes | Yes | Yes |

### Key observations

**1. The ML model feeds music21 better data**

The ML model's key confidence (0.783) is significantly higher than pyin's (0.656).
This is because the ML model detects polyphonic notes — when multiple notes sound
simultaneously (a chord or dyad), the pitch-class distribution is richer and
Krumhansl-Schmuckler has more signal to work with. pyin only tracks the dominant
(loudest) pitch per frame.

**2. Zero notes were filtered out in these tests**

Both backends detected notes that were entirely consistent with A major. This is
expected for a simple guitar recording. In a noisier recording or one with
background noise / open-string resonance, the filter would start removing false
positives. The tolerance=1 setting is appropriate here.

**3. pyin detects more "notes" but not necessarily more correct ones**

pyin found 28 notes vs. the ML model's 21. The extra 7 notes from pyin are likely
frame-level artifacts — short voiced segments at boundaries between real notes,
or sympathetic resonance being tracked as pitched events. The ML model's Schmitt-
trigger post-processing suppresses these.

**4. Chord identification quality depends on polyphony**

With pyin (monophonic), `chordify()` mostly produces single-note events labeled
`note`, with occasional `Minor Third` / `Minor Second` when pyin briefly tracks
harmonics as a separate pitch. With the ML model (polyphonic), it correctly
identifies power chord structures (`Perfect Fifth`, `Perfect Twelfth`).

**5. Fingering optimiser adds zero cost to ML output, real value to pyin output**

For ML model output, the string/fret is already known from the model's per-class
prediction. The optimiser is a pass-through. For pyin output, the optimiser must
assign fingering from scratch — and does so with a greedy heuristic that keeps
position jumps small. The pyin result used 14 unique positions vs. the ML model's
11, indicating the ML model's positions are intrinsically more clustered (it
"knows" the string).

---

## 8. What music21 Added That the Model Cannot Do

| Capability | Without music21 | With music21 |
|---|---|---|
| Key detection | None | Krumhansl-Schmuckler, with confidence score |
| Scale-based filtering | None | Drops notes outside key ± tolerance |
| Chord labelling | None | Identifies dyads and triads by name |
| Musical time model | Raw seconds | Quarter-note offsets at 120 BPM |
| ASCII tab | None | Full 6-string tab with position wrapping |

The model predicts *what* is played. music21 reasons about *why* it makes musical
sense — the interpretive layer the user described as wanting a "brain."

---

## 9. Remaining Limitations and Future Work

### Limitations

| Limitation | Impact |
|---|---|
| Key detection is global | A song that modulates (changes key mid-way) will have one key assigned to everything. Windowed key detection would fix this. |
| Greedy fingering heuristic | Does not find globally optimal hand position sequence. Viterbi over string positions would be better. |
| Chord labelling requires overlap | Monophonic input gets poor chord names. Rhythm and strumming detection would help group notes into chords. |
| Basic Pitch unavailable | Python 3.13 incompatibility (`tensorflow<2.15.1` has no Python 3.13 wheels). Revisit when basic-pitch relaxes its TF ceiling. |
| No tempo detection | Currently assumes 120 BPM for all MIDI output. `music21.tempo.MetronomeMark` analysis could infer actual BPM. |

### Future improvements

1. **Windowed key analysis** — run Krumhansl-Schmuckler in 8-bar windows to detect
   modulations and apply the correct scale filter per section.

2. **Viterbi fingering** — model the string position as a hidden state and use
   dynamic programming to find the globally cheapest sequence of hand positions.

3. **Rhythm quantisation** — snap note onsets to a musical grid (16th notes at
   detected BPM) to produce readable sheet music rather than performance-accurate
   MIDI.

4. **Basic Pitch re-evaluation** — monitor `basic-pitch` releases for Python 3.13
   support. When available, plug into `--backend basic-pitch`. It would provide
   polyphonic pitch bends (pitch contour per note) that neither pyin nor the
   current ML model outputs.

5. **LLM as final pass** — after music21 analysis, for segments where key
   confidence is low (<0.5), send the note sequence to an LLM with a structured
   prompt asking it to suggest corrections. This uses the LLM for what it does
   best: reasoning about ambiguous cases with contextual knowledge.

---

## 10. Summary Decision Tree

```
User wants "a brain" for guitar transcription
         |
         v
   Could use an LLM?
   -- No: LLMs can't hear audio, cost money, non-deterministic
         |
         v
   music21 + Better pitch detection
         |
         +----> Basic Pitch (Spotify)?
         |          -- Blocked: Python 3.13 + TF<2.15.1 incompatibility
         |          -- Stub left in code for future use
         |
         +----> Existing ML model as pitch backend
                    -- Already guitar-specific, F1=0.7843 on GuitarSet
                    -- Outputs string+fret, not just MIDI pitch
                    -- Higher key confidence than pyin (0.783 vs 0.656)
                    -- Python 3.13 compatible
                         |
                         v
                music21 adds:
                  - Key detection (Krumhansl-Schmuckler)
                  - Scale-based note filtering
                  - Chord identification
                  - Fingering optimisation
                  - ASCII tab output
```

The result is a locally-running, zero-cost, deterministic music theory layer that
gives the model-detected notes musical context and produces human-readable output
(tab, key, chords) that the baseline model never could.
