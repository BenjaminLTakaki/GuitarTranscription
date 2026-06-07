"""
External pitch detector backends for the hybrid transcription system.

Backends
--------
BasicPitchDetector  -- Spotify's basic-pitch (pip install basic-pitch).
                      Works on Windows, returns frame-level probabilities directly.
YourMT3Detector     -- Google's YourMT3+ (open-source PyTorch re-implementation).
                      Run third_party/setup_yourmt3.py once to download it.
MT3Detector         -- Legacy: read a pre-computed MIDI file from any AMT tool.

All backends return (T, NUM_PITCHES) float32 aligned to the CQT frame grid so
they can be passed directly to pitch_gate.apply_pitch_gate().
"""

from __future__ import annotations

import abc
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from model.constants import HOP_LENGTH, MIDI_MIN, MIDI_MAX, NUM_PITCHES, SAMPLE_RATE

# YourMT3 is cloned here by third_party/setup_yourmt3.py
_YOURMT3_DIR = Path(__file__).parent.parent / "third_party" / "YourMT3"

# Exact model args for the strongest public checkpoint: YPTF.MoE+Multi (noPS)
# Source: https://huggingface.co/spaces/mimbres/YourMT3/blob/main/app.py
_YOURMT3_ARGS = [
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
    "-p",   "2024",
    "-tk",  "mc13_full_plus_256",
    "-dec", "multi-t5",
    "-nl",  "26",
    "-enc", "perceiver-tf",
    "-sqr", "1",
    "-ff",  "moe",
    "-wf",  "4",
    "-nmoe","8",
    "-kmoe","2",
    "-act", "silu",
    "-epe", "rope",
    "-rp",  "1",
    "-ac",  "spec",
    "-hop", "300",
    "-atc", "1",
    "-pr",  "16",
]

# basic-pitch starts at C1 (MIDI 24); compute offsets into its output array.
_BP_MIDI_OFFSET = 24          # lowest pitch in basic-pitch's 88-bin output
_BP_GUITAR_START = MIDI_MIN - _BP_MIDI_OFFSET   # 40 - 24 = 16
_BP_GUITAR_END   = MIDI_MAX - _BP_MIDI_OFFSET + 1  # 88 - 24 + 1 = 65


class PitchDetector(abc.ABC):
    """Return frame-level pitch confidences aligned to the CQT frame grid."""

    @abc.abstractmethod
    def predict(self, audio_path: Path, n_frames: int) -> np.ndarray:
        """
        Parameters
        ----------
        audio_path : path to the audio file
        n_frames   : number of CQT frames expected in the output

        Returns
        -------
        pitch_probs : (n_frames, NUM_PITCHES) float32 in [0, 1]
                      column 0 -> MIDI_MIN (E2=40), column 48 -> MIDI_MAX (E6=88)
        """


# ---------------------------------------------------------------------------
# BasicPitch backend
# ---------------------------------------------------------------------------

class BasicPitchDetector(PitchDetector):
    """
    Uses Spotify's basic-pitch for pitch detection.

    Install:  pip install basic-pitch
    Docs:     https://github.com/spotify/basic-pitch
    """

    def __init__(self):
        # onnxruntime MUST be imported before jams (or any lib that triggers
        # basic_pitch import) — jams indirectly causes basic_pitch to cache
        # ONNX_PRESENT=False even when onnxruntime is installed.
        try:
            import onnxruntime  # noqa: F401 — ensures sys.modules entry exists
            from basic_pitch.inference import predict as _bp_predict
            from basic_pitch import ICASSP_2022_MODEL_PATH as _bp_model
        except ImportError:
            raise ImportError(
                "basic-pitch is not installed.\n"
                "Run:  pip install basic-pitch onnxruntime"
            )
        self._predict = _bp_predict
        # Prefer the ONNX model (nmp.onnx) when TensorFlow is broken/missing.
        # Pre-instantiate Model so run_inference receives an object, not a path —
        # this bypasses basic-pitch's broken TF/ONNX path-detection logic.
        from basic_pitch.inference import Model as _BPModel
        _onnx = Path(str(_bp_model) + ".onnx")
        _path = str(_onnx) if _onnx.exists() else str(_bp_model)
        self._model = _BPModel(_path)

    def predict(self, audio_path: Path, n_frames: int) -> np.ndarray:
        model_output, _, _ = self._predict(
            str(audio_path),
            self._model,
            onset_threshold=0.3,
            frame_threshold=0.2,
            minimum_note_length=58,  # ms
            melodia_trick=False,     # keep polyphonic
        )

        # model_output['note'] is (T_bp, 88), values in [0, 1]
        note_probs = model_output["note"]  # (T_bp, 88)

        # Slice to guitar MIDI range [40, 88] -> basic-pitch indices [16, 65)
        guitar_probs = note_probs[:, _BP_GUITAR_START:_BP_GUITAR_END]  # (T_bp, 49)

        return _resample_time(guitar_probs, n_frames)


# ---------------------------------------------------------------------------
# YourMT3 backend (local, PyTorch-based)
# ---------------------------------------------------------------------------

class YourMT3Detector(PitchDetector):
    """
    Uses a locally-installed YourMT3+ for pitch detection.

    Setup (one-time, ~2.8 GB download):
        python third_party/setup_yourmt3.py

    YourMT3+ is an open-source PyTorch re-implementation of Google's MT3.
    It runs in a subprocess to avoid a Python package-name collision between
    YourMT3's internal 'model/' package and this project's own 'model/' package.

    The model is heavy (~2 GB GPU RAM).  It runs on CPU if no CUDA GPU is
    available, but will be slow on long audio.
    """

    # Standalone worker script that runs YourMT3 in its own process
    _WORKER = Path(__file__).parent.parent / "third_party" / "yourmt3_transcribe.py"

    def __init__(self, yourmt3_dir: Path | None = None, device: str = "auto"):
        root = Path(yourmt3_dir) if yourmt3_dir else _YOURMT3_DIR

        if not root.exists():
            raise FileNotFoundError(
                f"YourMT3 not found at {root}\n"
                "Run first:  python third_party/setup_yourmt3.py"
            )
        if not self._WORKER.exists():
            raise FileNotFoundError(
                f"Worker script not found: {self._WORKER}\n"
                "This file should be part of the repository."
            )

        self._yourmt3_dir = root
        self._device      = device

    def predict(self, audio_path: Path, n_frames: int) -> np.ndarray:
        """Launch YourMT3 in a subprocess, return frame-level pitch probs."""
        tmpdir = Path(tempfile.mkdtemp(prefix="yourmt3_"))
        try:
            cmd = [
                sys.executable, "-X", "utf8",   # force UTF-8 stdout/stderr
                str(self._WORKER),
                "--audio",      str(audio_path.resolve()),
                "--output-dir", str(tmpdir),
            ]
            env = os.environ.copy()
            env["PYTHONUTF8"]      = "1"
            env["WANDB_DISABLED"]  = "true"   # prevent wandb from hooking stdout
            env["WANDB_MODE"]      = "disabled"

            print("Running YourMT3 (loading model + transcribing) ...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",  # subprocess uses -X utf8, parent must match
                timeout=600,   # 10-minute ceiling for long audio
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"YourMT3 subprocess failed (exit {result.returncode}):\n"
                    f"{result.stderr[-2000:]}"   # last 2000 chars of stderr
                )
            # stdout contains model config printout followed by the MIDI path on the last line
            last_line = result.stdout.strip().splitlines()[-1].strip()
            midi_path = Path(last_line)
            if not midi_path.exists():
                raise RuntimeError(
                    f"YourMT3 did not produce a MIDI file at {midi_path}\n"
                    f"stdout: {result.stdout!r}\n"
                    f"stderr: {result.stderr[-500:]!r}"
                )
            return _midi_file_to_pitch_probs(midi_path, n_frames)
        finally:
            shutil.rmtree(str(tmpdir), ignore_errors=True)


# ---------------------------------------------------------------------------
# MT3 backend (MIDI-file based, legacy)
# ---------------------------------------------------------------------------

class MT3Detector(PitchDetector):
    """
    Reads any pre-computed MIDI file and converts it to frame-level pitch
    activations.  Use this if you have output from Google's original MT3
    (JAX-based) or any other AMT tool.

    Workflow
    --------
    1. Run your AMT tool on the audio and get a .mid output file.
    2. Point this class at that file.
    3. Use --pitch-backend mt3 --mt3-midi path/to/file.mid with predict.py.
    """

    def __init__(self, midi_path: Path):
        self._midi_path = Path(midi_path)
        if not self._midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {self._midi_path}")

    def predict(self, audio_path: Path, n_frames: int) -> np.ndarray:
        # audio_path is ignored -- we already have the pre-computed MIDI
        return _midi_file_to_pitch_probs(self._midi_path, n_frames)


# ---------------------------------------------------------------------------
# MIDI -> frame activations
# ---------------------------------------------------------------------------

def _midi_file_to_pitch_probs(midi_path: Path, n_frames: int) -> np.ndarray:
    """Parse a MIDI file and return (n_frames, NUM_PITCHES) float32 activations."""
    try:
        import mido
    except ImportError:
        raise ImportError("mido is required for MIDI parsing.  pip install mido")

    midi      = mido.MidiFile(str(midi_path))
    tempo     = 500_000   # default 120 BPM in microseconds per beat
    frame_sec = HOP_LENGTH / SAMPLE_RATE

    probs: np.ndarray = np.zeros((n_frames, NUM_PITCHES), dtype=np.float32)
    active: dict[int, float] = {}   # midi_pitch -> onset_time_sec
    current_sec = 0.0

    for msg in mido.merge_tracks(midi.tracks):
        current_sec += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)

        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = current_sec
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            onset = active.pop(msg.note, None)
            if onset is not None:
                _fill_frames(probs, msg.note, onset, current_sec, frame_sec, n_frames)

    # Close notes that are still active at the end of the track
    audio_dur = n_frames * frame_sec
    for pitch, onset in active.items():
        _fill_frames(probs, pitch, onset, audio_dur, frame_sec, n_frames)

    return probs


def _fill_frames(
    probs: np.ndarray,
    midi_pitch: int,
    onset_sec: float,
    offset_sec: float,
    frame_sec: float,
    n_frames: int,
) -> None:
    if not (MIDI_MIN <= midi_pitch <= MIDI_MAX):
        return
    pitch_idx = midi_pitch - MIDI_MIN
    f_start = max(0, int(onset_sec / frame_sec))
    f_end   = min(n_frames, int(offset_sec / frame_sec) + 1)
    probs[f_start:f_end, pitch_idx] = 1.0


def _resample_time(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resample the time axis of arr to n_frames."""
    T_src = arr.shape[0]
    if T_src == n_frames:
        return arr.astype(np.float32)
    x_src  = np.linspace(0.0, 1.0, T_src)
    x_dst  = np.linspace(0.0, 1.0, n_frames)
    interp = interp1d(x_src, arr, axis=0, kind="linear", fill_value="extrapolate")
    return np.clip(interp(x_dst), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_detector(backend: str, **kwargs) -> PitchDetector:
    """
    Create a pitch detector by name.

    backend='basic_pitch'                  -> BasicPitchDetector()
    backend='yourmt3'                      -> YourMT3Detector()
    backend='yourmt3', device='cpu'        -> YourMT3Detector(device='cpu')
    backend='mt3', midi_path=<path>        -> MT3Detector(midi_path)
    """
    if backend == "basic_pitch":
        return BasicPitchDetector()
    if backend == "yourmt3":
        return YourMT3Detector(
            yourmt3_dir=kwargs.get("yourmt3_dir"),
            device=kwargs.get("device", "auto"),
        )
    if backend == "mt3":
        if "midi_path" not in kwargs:
            raise ValueError("mt3 backend requires midi_path=<path_to_midi_file>")
        return MT3Detector(kwargs["midi_path"])
    raise ValueError(
        f"Unknown backend {backend!r}. "
        "Choose 'basic_pitch', 'yourmt3', or 'mt3'."
    )
