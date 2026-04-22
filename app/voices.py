"""PAI fork additions: FastAPI router for /embed and /voices/{name}.

These routes are additive to the upstream whisperx-asr-service. They share the
pyannote diarization pipeline already loaded by app.pipeline so no second model
enters memory (ISC-A5).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, Path as FPath, UploadFile
from fastapi.responses import JSONResponse

from app.name_validation import InvalidVoiceName, normalize_and_validate
from app.voices_config import get_settings
from app.voices_embed import (
    MODEL_VERSION,
    decode_to_mono_16k,
    extract_embedding,
    probe_clip_duration,
)
from app.voices_storage import append_voice

logger = logging.getLogger(__name__)

router = APIRouter(tags=["voices"])


def _get_pipeline():
    """Return the app-level diarization pipeline singleton.

    Isolated as its own function so tests can monkeypatch this module's
    `_get_pipeline` attribute without importing whisperx.
    """
    from app.pipeline import load_diarize_pipeline

    return load_diarize_pipeline()


def _enforce_duration(audio_bytes: bytes, max_seconds: float) -> float:
    try:
        duration = probe_clip_duration(audio_bytes)
    except Exception as exc:  # noqa: BLE001 — surfaced as 422 regardless of sub-error
        raise HTTPException(
            status_code=422,
            detail=f"uploaded bytes are not decodable audio: {exc}",
        )
    if duration > max_seconds:
        raise HTTPException(
            status_code=413,
            detail=f"clip duration {duration:.2f}s exceeds max {max_seconds:.2f}s",
        )
    return duration


async def _compute_embedding(audio_bytes: bytes) -> list[float]:
    try:
        audio = decode_to_mono_16k(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"audio decode failed: {exc}",
        )
    pipeline = _get_pipeline()
    try:
        return extract_embedding(pipeline, audio)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embedding extraction failed")
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}")


@router.post("/embed")
async def embed_clip(audio_file: UploadFile = File(...)):
    settings = get_settings()
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio_file is empty")

    _enforce_duration(audio_bytes, settings.max_clip_seconds)
    embedding = await _compute_embedding(audio_bytes)

    return JSONResponse(
        {
            "embedding": embedding,
            "dim": len(embedding),
            "model_version": MODEL_VERSION,
        }
    )


@router.post("/voices/{name}")
async def append_voice_clip(
    name: str = FPath(..., description="speaker name"),
    audio_file: UploadFile = File(...),
):
    settings = get_settings()
    try:
        safe_name = normalize_and_validate(name, settings.clips_dir)
    except InvalidVoiceName as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio_file is empty")

    _enforce_duration(audio_bytes, settings.max_clip_seconds)
    embedding = await _compute_embedding(audio_bytes)

    clip_path, ref_count = await append_voice(
        library_path=settings.library_path,
        clips_dir=settings.clips_dir,
        name=safe_name,
        audio_bytes=audio_bytes,
        embedding=embedding,
        max_refs=settings.max_references_per_name,
    )

    return JSONResponse(
        {
            "ok": True,
            "name": safe_name,
            "reference_count": ref_count,
            "stored_clip": str(clip_path),
        }
    )
