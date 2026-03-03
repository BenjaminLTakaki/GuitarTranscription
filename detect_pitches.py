import argparse
from pathlib import Path

import librosa
import mido
import numpy as np


def frequency_to_note_name(freq: float) -> str:
    if np.isnan(freq) or freq <= 0:
        return ""
    midi = int(np.round(librosa.hz_to_midi(freq)))
    return librosa.midi_to_note(midi)


def frequency_to_midi_number(freq: float):
    if np.isnan(freq) or freq <= 0:
        return None
    return int(np.round(librosa.hz_to_midi(freq)))


def detect_pitches(audio_path: Path, frame_length: int = 2048, hop_length: int = 256):
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    fmin = librosa.note_to_hz("E2")
    fmax = librosa.note_to_hz("E6")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    frame_step_sec = float(times[1] - times[0]) if len(times) > 1 else float(hop_length / sr)

    rows = []
    for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_prob):
        note = frequency_to_note_name(freq)
        midi_number = frequency_to_midi_number(freq)
        rows.append(
            {
                "time_sec": float(t),
                "frequency_hz": "" if np.isnan(freq) else float(freq),
                "note": note,
                "midi": midi_number,
                "voiced": bool(voiced) if voiced is not None else False,
                "voiced_probability": "" if np.isnan(prob) else float(prob),
            }
        )

    return rows, frame_step_sec


def rows_to_note_segments(rows, frame_step_sec: float, min_note_duration: float):
    segments = []
    active_midi = None
    active_start = 0.0

    for row in rows:
        time_sec = float(row["time_sec"])
        is_voiced = bool(row["voiced"])
        midi_number = row["midi"]

        if is_voiced and midi_number is not None:
            if active_midi is None:
                active_midi = midi_number
                active_start = time_sec
            elif midi_number != active_midi:
                end_time = max(time_sec, active_start + frame_step_sec)
                duration = end_time - active_start
                if duration >= min_note_duration:
                    segments.append({"midi": active_midi, "start": active_start, "end": end_time})
                active_midi = midi_number
                active_start = time_sec
        else:
            if active_midi is not None:
                end_time = max(time_sec, active_start + frame_step_sec)
                duration = end_time - active_start
                if duration >= min_note_duration:
                    segments.append({"midi": active_midi, "start": active_start, "end": end_time})
                active_midi = None

    if active_midi is not None and rows:
        final_time = float(rows[-1]["time_sec"]) + frame_step_sec
        duration = final_time - active_start
        if duration >= min_note_duration:
            segments.append({"midi": active_midi, "start": active_start, "end": final_time})

    return segments


def write_midi(segments, output_path: Path, bpm: int = 120):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    current_time_sec = 0.0
    for segment in segments:
        start_sec = max(float(segment["start"]), current_time_sec)
        end_sec = max(float(segment["end"]), start_sec)
        midi_note = int(segment["midi"])

        delta_on_sec = start_sec - current_time_sec
        delta_on_ticks = int(round(mido.second2tick(delta_on_sec, midi_file.ticks_per_beat, tempo)))
        track.append(mido.Message("note_on", note=midi_note, velocity=80, time=delta_on_ticks))

        duration_sec = end_sec - start_sec
        duration_ticks = int(round(mido.second2tick(duration_sec, midi_file.ticks_per_beat, tempo)))
        track.append(mido.Message("note_off", note=midi_note, velocity=0, time=duration_ticks))

        current_time_sec = end_sec

    midi_file.save(str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Detect monophonic pitch over time from an audio file (first iteration)."
    )
    parser.add_argument("audio_file", type=Path, help="Path to input audio (wav/mp3/flac/...)" )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/detected_pitches.mid"),
        help="Output MIDI path",
    )
    parser.add_argument(
        "--frame-length",
        type=int,
        default=2048,
        help="Analysis frame length",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop length between frames",
    )
    parser.add_argument(
        "--min-note-duration",
        type=float,
        default=0.05,
        help="Minimum detected note duration in seconds",
    )

    args = parser.parse_args()

    if not args.audio_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.audio_file}")

    rows, frame_step_sec = detect_pitches(
        audio_path=args.audio_file,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )
    segments = rows_to_note_segments(
        rows=rows,
        frame_step_sec=frame_step_sec,
        min_note_duration=args.min_note_duration,
    )
    write_midi(segments, args.output)

    voiced_rows = [r for r in rows if r["voiced"] and r["note"]]
    unique_notes = sorted(set(librosa.midi_to_note(s["midi"]) for s in segments))

    print(f"Saved MIDI to: {args.output}")
    print(f"Frames analyzed: {len(rows)}")
    print(f"Voiced frames: {len(voiced_rows)}")
    print(f"MIDI notes written: {len(segments)}")
    print(f"Unique detected notes: {', '.join(unique_notes) if unique_notes else 'None'}")


if __name__ == "__main__":
    main()
