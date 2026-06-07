#!/usr/bin/env python3
"""FastAPI web server for GuitarScribes transcription frontend."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model.constants import CHECKPOINT_DIR, DEFAULT_THRESHOLD

app = FastAPI(title="GuitarScribes")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

frontend_dir = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/checkpoints")
async def list_checkpoints():
    """Return all .pt files in the checkpoints directory with metadata."""
    import torch

    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        return {"checkpoints": []}

    allowed = {"best_model_v1.pt", "best_model_v2.pt"}
    results = []
    for pt in sorted(ckpt_dir.glob("*.pt")):
        if pt.name not in allowed:
            continue
        try:
            ckpt = torch.load(pt, map_location="cpu", weights_only=True)
            results.append({
                "name": pt.name,
                "epoch": ckpt.get("epoch"),
                "f1": round(float(ckpt["f1"]), 4) if "f1" in ckpt else None,
            })
        except Exception:
            results.append({"name": pt.name, "epoch": None, "f1": None})

    return {"checkpoints": results}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    checkpoint: str = Form("checkpoint_epoch600.pt"),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    """Accept audio upload, run model, return note events + MIDI download id."""
    import torch

    # Import heavy inference dependencies lazily so the web server can start fast
    # and reliably on Windows when autoreload/spawn is used.
    from model.predict import pianoroll_to_notes, predict, write_midi

    ckpt_path = Path(CHECKPOINT_DIR) / checkpoint
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint}")

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = Path(tmp.name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frame_prob, onset_prob, art_prob = predict(tmp_path, ckpt_path, device)

        notes = pianoroll_to_notes(
            frame_prob,
            onset_prob,
            art_prob=art_prob,
            onset_threshold=threshold,
        )

        midi_id = uuid.uuid4().hex[:10]
        midi_path = OUTPUT_DIR / f"{midi_id}.mid"
        write_midi(notes, midi_path)

        n_hammers = sum(1 for n in notes if n.get("articulation") == "hammer_on")
        duration = max((n["end"] for n in notes), default=0.0)

        return JSONResponse({
            "notes": notes,
            "midi_id": midi_id,
            "count": len(notes),
            "hammer_ons": n_hammers,
            "duration": round(duration, 2),
            "device": str(device),
        })

    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.get("/midi/{midi_id}")
async def download_midi(midi_id: str):
    """Download a previously transcribed MIDI file."""
    # Sanitize: only allow hex ids
    if not all(c in "0123456789abcdef" for c in midi_id):
        raise HTTPException(status_code=400, detail="Invalid id")
    path = OUTPUT_DIR / f"{midi_id}.mid"
    if not path.exists():
        raise HTTPException(status_code=404, detail="MIDI not found")
    return FileResponse(str(path), media_type="audio/midi", filename="transcription.mid")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    reload_enabled = os.getenv("RELOAD", "0").lower() in {"1", "true", "yes", "on"}
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)
