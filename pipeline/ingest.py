"""Phase 1A — Tab ingestion from DadaGP Guitar Pro files.

Parses .gp3/.gp4/.gp5/.gpx files using PyGuitarPro, extracts guitar tracks
with tempo-aware timestamp conversion, and exports each as MIDI + JAMS in
GuitarSet-compatible format.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import guitarpro
import mido

from model.constants import GUITAR_TUNING, NUM_FRETS, NUM_STRINGS


# ---------------------------------------------------------------------------
# Guitar Pro → timed note events
# ---------------------------------------------------------------------------

def _is_guitar_track(track: guitarpro.models.Track) -> bool:
    """Heuristic: keep tracks whose MIDI channel is not 9 (drums) and whose
    name does not obviously indicate bass/drums/vocals."""
    if track.channel.channel == 9:
        return False
    name = track.name.lower()
    skip = {"drum", "drums", "percussion", "bass", "vocal", "vocals", "voice", "keys", "keyboard", "piano", "synth"}
    return not any(kw in name for kw in skip)


def gp_to_events(
    gp_song: guitarpro.models.Song,
    track_idx: int,
) -> List[dict]:
    """Convert a Guitar Pro track to timed note events.

    Returns list of dicts with keys:
        onset, offset, string, fret, midi, velocity
    """
    track = gp_song.tracks[track_idx]
    events: List[dict] = []
    current_time = 0.0  # seconds

    # Build a map of tied notes to extend previous note offsets
    # Key: (string,) -> last event dict on that string (for tie handling)
    last_on_string: dict[int, dict] = {}

    for measure_idx, measure in enumerate(track.measures):
        header = gp_song.measureHeaders[measure_idx]
        bpm = _resolve_bpm(gp_song, header)
        beat_duration = 60.0 / bpm

        for voice in measure.voices:
            beat_time = current_time
            for beat in voice.beats:
                # Duration: quarter=4 → note_duration_beats = 4/4 = 1 beat
                note_duration_beats = 4.0 / beat.duration.value
                if getattr(beat.duration, "isDotted", False):
                    note_duration_beats *= 1.5
                if getattr(beat.duration, "isDoubleDotted", False):
                    note_duration_beats *= 1.75
                # Tuplet
                tuplet = getattr(beat.duration, "tuplet", None)
                enters = getattr(tuplet, "enters", 0) if tuplet is not None else 0
                if tuplet is not None and enters > 0:
                    note_duration_beats *= (
                        tuplet.times / enters
                    )
                note_duration_sec = note_duration_beats * beat_duration

                for note in beat.notes:
                    string_idx = note.string - 1  # GP uses 1-indexed
                    fret = note.value

                    # Compute MIDI pitch from track's string tuning
                    if string_idx < len(track.strings):
                        midi_pitch = track.strings[string_idx].value + fret
                    else:
                        continue

                    # Handle tied notes: extend previous note on same string
                    if note.type == guitarpro.NoteType.tie:
                        prev = last_on_string.get(string_idx)
                        if prev is not None:
                            prev["offset"] = beat_time + note_duration_sec
                        beat_time += note_duration_sec
                        continue

                    # Rest / dead notes: skip
                    if note.type == guitarpro.NoteType.rest:
                        continue

                    # Grace notes: very short duration
                    if note.effect and note.effect.grace:
                        grace_dur = 0.03  # 30ms
                        ev = {
                            "onset": beat_time,
                            "offset": beat_time + grace_dur,
                            "string": string_idx,
                            "fret": fret,
                            "midi": midi_pitch,
                            "velocity": note.velocity,
                        }
                        events.append(ev)
                        last_on_string[string_idx] = ev
                        continue

                    ev = {
                        "onset": beat_time,
                        "offset": beat_time + note_duration_sec,
                        "string": string_idx,
                        "fret": fret,
                        "midi": midi_pitch,
                        "velocity": note.velocity,
                    }
                    events.append(ev)
                    last_on_string[string_idx] = ev

                    # Let-ring: extend offset until the next note on same string
                    # (handled later in _apply_let_ring)

                beat_time += note_duration_sec

        # Advance by the measure's actual duration
        ts_num, ts_den = _resolve_time_signature(header)
        measure_duration = (ts_num / ts_den) * 4 * beat_duration
        current_time += measure_duration

    # Apply let-ring: notes with letRing extend until the next note on same string
    events = _apply_let_ring(events)
    return events


def _apply_let_ring(events: List[dict]) -> List[dict]:
    """Extend notes that ring until the next note plays on the same string."""
    from collections import defaultdict

    by_string: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_string[e["string"]].append(e)

    for string, group in by_string.items():
        group.sort(key=lambda e: e["onset"])
        for i in range(len(group) - 1):
            # If the note's offset would be shorter than the gap to next onset,
            # and it's a let-ring candidate, extend it
            next_onset = group[i + 1]["onset"]
            if group[i]["offset"] < next_onset:
                # Only extend if gap is small (< 2 seconds)
                gap = next_onset - group[i]["offset"]
                if gap < 2.0:
                    group[i]["offset"] = next_onset

    return events


def _resolve_bpm(gp_song: guitarpro.models.Song, header) -> float:
    """Resolve tempo robustly across GP versions and PyGuitarPro models."""
    tempo = getattr(header, "tempo", None)
    if tempo is not None:
        value = getattr(tempo, "value", tempo)
        try:
            bpm = float(value)
            if bpm > 0:
                return bpm
        except (TypeError, ValueError):
            pass

    # Fallbacks seen in older/variant GP structures
    song_tempo = getattr(gp_song, "tempo", None)
    if song_tempo is not None:
        value = getattr(song_tempo, "value", song_tempo)
        try:
            bpm = float(value)
            if bpm > 0:
                return bpm
        except (TypeError, ValueError):
            pass

    return 120.0


def _resolve_time_signature(header) -> tuple[float, float]:
    """Return (numerator, denominator) with safe defaults."""
    ts = getattr(header, "timeSignature", None)
    if ts is None:
        return 4.0, 4.0

    num = getattr(ts, "numerator", 4)
    den_obj = getattr(ts, "denominator", 4)
    den = getattr(den_obj, "value", den_obj)

    try:
        num_f = float(num)
        den_f = float(den)
        if num_f > 0 and den_f > 0:
            return num_f, den_f
    except (TypeError, ValueError):
        pass

    return 4.0, 4.0


# ---------------------------------------------------------------------------
# Export: events → MIDI + JAMS
# ---------------------------------------------------------------------------

def events_to_midi(events: List[dict], bpm: float = 120.0) -> mido.MidiFile:
    """Convert note events to a single-track MIDI file."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.Message("program_change", program=25, channel=0, time=0))

    raw: list[tuple[float, str, int, int]] = []
    for e in events:
        raw.append((e["onset"], "on", e["midi"], e["velocity"]))
        raw.append((e["offset"], "off", e["midi"], 0))
    raw.sort(key=lambda x: x[0])

    cur_tick = 0
    for t_sec, kind, note, vel in raw:
        abs_tick = int(mido.second2tick(t_sec, mid.ticks_per_beat, tempo))
        delta = max(0, abs_tick - cur_tick)
        if kind == "on":
            track.append(mido.Message("note_on", note=note, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=note, velocity=0, time=delta))
        cur_tick = abs_tick

    return mid


def events_to_jams(events: List[dict], duration: float, title: str = "") -> dict:
    """Build GuitarSet-compatible JAMS dict with per-string note_midi annotations."""
    annotations = []
    for string_idx in range(NUM_STRINGS):
        # Pitch contour placeholder (empty, matches GuitarSet layout)
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "aligned-pipeline", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "AlignedDataset",
                "annotation_tools": "pipeline/ingest.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "pitch_contour",
            "data": [],
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })
        # Note MIDI events for this string
        string_notes = sorted(
            [e for e in events if e["string"] == string_idx],
            key=lambda e: e["onset"],
        )
        annotations.append({
            "annotation_metadata": {
                "curator": {"name": "aligned-pipeline", "email": ""},
                "annotator": {},
                "version": "1.0",
                "corpus": "AlignedDataset",
                "annotation_tools": "pipeline/ingest.py",
                "annotation_rules": "",
                "validation": "",
                "data_source": str(string_idx),
            },
            "namespace": "note_midi",
            "data": [
                {
                    "time": e["onset"],
                    "duration": e["offset"] - e["onset"],
                    "value": float(e["midi"]),
                    "confidence": None,
                }
                for e in string_notes
            ],
            "sandbox": {},
            "time": 0,
            "duration": duration,
        })

    return {
        "annotations": annotations,
        "file_metadata": {
            "title": title,
            "artist": "",
            "release": "",
            "duration": duration,
            "identifiers": {},
            "jams_version": "0.3.4",
        },
        "sandbox": {},
    }


# ---------------------------------------------------------------------------
# Tuning detection
# ---------------------------------------------------------------------------

def get_track_tuning(track: guitarpro.models.Track) -> tuple[int, ...]:
    """Return MIDI values for each open string in the track."""
    return tuple(s.value for s in track.strings)


def is_standard_tuning(track: guitarpro.models.Track) -> bool:
    """Check if track uses standard guitar tuning."""
    tuning = get_track_tuning(track)
    return tuning == GUITAR_TUNING


# ---------------------------------------------------------------------------
# High-level: parse a directory of GP files
# ---------------------------------------------------------------------------

def ingest_directory(
    dadagp_dir: Path,
    output_dir: Path,
    max_songs: int | None = None,
    standard_tuning_only: bool = True,
) -> list[dict]:
    """Parse all Guitar Pro files in dadagp_dir, export MIDI+JAMS.

    Returns a manifest: list of dicts with metadata per exported track.
    """
    midi_dir = output_dir / "midi"
    jams_dir = output_dir / "jams"
    midi_dir.mkdir(parents=True, exist_ok=True)
    jams_dir.mkdir(parents=True, exist_ok=True)

    gp_files = []
    for ext in ("*.gp3", "*.gp4", "*.gp5", "*.gpx", "*.gp"):
        gp_files.extend(dadagp_dir.rglob(ext))
    gp_files.sort()
    if max_songs is not None:
        gp_files = gp_files[:max_songs]

    manifest: list[dict] = []
    processed = 0
    errors = 0

    for gp_path in gp_files:
        try:
            gp_song = guitarpro.parse(str(gp_path))
        except Exception as exc:
            print(f"  [SKIP] {gp_path.name}: parse error: {exc}")
            errors += 1
            continue

        title = gp_song.title or gp_path.stem
        artist = gp_song.artist or "unknown"

        for trk_idx, track in enumerate(gp_song.tracks):
            if not _is_guitar_track(track):
                continue
            if standard_tuning_only and not is_standard_tuning(track):
                continue

            events = gp_to_events(gp_song, trk_idx)
            if not events:
                continue

            # Compute duration from last event
            duration = max(e["offset"] for e in events)
            if duration < 5.0:
                continue  # skip very short tracks

            # Sanitize filename
            safe_name = (
                f"{gp_path.stem}_trk{trk_idx}"
                .replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )

            # Use robust resolver to support GP variants where header tempo may be absent
            bpm = _resolve_bpm(gp_song, gp_song.measureHeaders[0]) if gp_song.measureHeaders else 120.0

            # Export MIDI
            midi_path = midi_dir / f"{safe_name}.mid"
            mid = events_to_midi(events, bpm=bpm)
            mid.save(str(midi_path))

            # Export JAMS
            jams_path = jams_dir / f"{safe_name}.jams"
            jams = events_to_jams(events, duration, title=title)
            with open(jams_path, "w", encoding="utf-8") as f:
                json.dump(jams, f, indent=2)

            manifest.append({
                "song_title": title,
                "artist": artist,
                "gp_file": str(gp_path),
                "midi_path": str(midi_path),
                "jams_path": str(jams_path),
                "duration_sec": round(duration, 2),
                "num_notes": len(events),
                "tuning": ",".join(str(s.value) for s in track.strings),
                "track_name": track.name,
            })
            processed += 1

    print(f"Ingestion complete: {processed} tracks from {len(gp_files)} files ({errors} errors)")

    # Save manifest CSV
    if manifest:
        csv_path = output_dir / "manifest.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
            writer.writeheader()
            writer.writerows(manifest)
        print(f"Manifest saved: {csv_path}")

    return manifest
