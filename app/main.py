"""
WhisperX ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints
"""

import asyncio
import os
import tempfile
import logging
import warnings
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import JSONResponse
import whisperx

from app.version import __version__
from app.pipeline import (
    DEVICE,
    COMPUTE_TYPE,
    BATCH_SIZE,
    HF_TOKEN,
    DEFAULT_MODEL,
    load_whisper_model,
    clear_gpu_memory,
    format_timestamp,
    sanitize_float_values,
    run_pipeline,
    _whisper_models as loaded_models,
)
from app.queue import run_in_queue, get_queue_metrics
from app.voices import router as voices_router
from app.voices_embed import run_startup_probe

# Suppress pyannote pooling warnings about degrees of freedom
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))
SERVE_MODE = os.getenv("SERVE_MODE", "simple")

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version=__version__
)

logger.info(f"WhisperX ASR Service v{__version__} initialized on device: {DEVICE}")
logger.info(f"Compute type: {COMPUTE_TYPE}, Batch size: {BATCH_SIZE}")
logger.info(f"Default model: {DEFAULT_MODEL}, Serve mode: {SERVE_MODE}")


# PAI fork: /health stays red until this flips true via the startup probe.
_VOICES_PROBE_PASSED: bool = False
_VOICES_PROBE_LAST_ERROR: str | None = None
_VOICES_PROBE_RETRY_TASK: asyncio.Task | None = None
_VOICES_PROBE_ATTEMPTS: int = 0


def _voices_probe_retry_delay_seconds() -> float:
    return max(1.0, float(os.getenv("PAI_VOICES_PROBE_RETRY_SEC", "30")))


async def _run_voices_probe_once() -> bool:
    """Try the embedding probe once and update shared health state."""
    global _VOICES_PROBE_PASSED, _VOICES_PROBE_LAST_ERROR, _VOICES_PROBE_ATTEMPTS

    _VOICES_PROBE_ATTEMPTS += 1
    attempt = _VOICES_PROBE_ATTEMPTS
    logger.info("[pai-voices] startup embedding probe attempt=%d", attempt)
    try:
        from app.pipeline import load_diarize_pipeline

        await asyncio.to_thread(run_startup_probe, load_diarize_pipeline)
        _VOICES_PROBE_PASSED = True
        _VOICES_PROBE_LAST_ERROR = None
        logger.info("[pai-voices] startup embedding probe marked ready attempt=%d", attempt)
        return True
    except Exception as e:  # noqa: BLE001
        _VOICES_PROBE_PASSED = False
        _VOICES_PROBE_LAST_ERROR = str(e)
        logger.error("[pai-voices] startup embedding probe FAILED attempt=%d error=%s", attempt, e)
        return False


async def _retry_voices_probe_until_ready() -> None:
    """Keep retrying the startup probe until it succeeds."""
    global _VOICES_PROBE_RETRY_TASK

    delay = _voices_probe_retry_delay_seconds()
    while not _VOICES_PROBE_PASSED:
        logger.warning("[pai-voices] retrying startup embedding probe in %.1fs", delay)
        await asyncio.sleep(delay)
        if await _run_voices_probe_once():
            break
    _VOICES_PROBE_RETRY_TASK = None


@app.on_event("startup")
async def startup_event():
    """Preload models on startup + run pai-voices embedding probe."""
    global _VOICES_PROBE_PASSED, _VOICES_PROBE_LAST_ERROR, _VOICES_PROBE_RETRY_TASK

    preload_model = os.getenv("PRELOAD_MODEL", None)
    if preload_model:
        logger.info(f"Preloading model on startup: {preload_model}")
        try:
            load_whisper_model(preload_model)
            logger.info(f"Successfully preloaded model: {preload_model}")
        except Exception as e:
            logger.error(f"Failed to preload model {preload_model}: {str(e)}")

    # PAI fork: verify the diarization pipeline can actually emit an embedding
    # before accepting /embed or /voices/{name} traffic. Gates /health to 503
    # until the probe passes. Opt out with PAI_VOICES_SKIP_PROBE=1 (used in tests
    # and in minimal CI environments without HF_TOKEN).
    if os.getenv("PAI_VOICES_SKIP_PROBE") == "1":
        logger.warning("[pai-voices] startup probe skipped (PAI_VOICES_SKIP_PROBE=1)")
        _VOICES_PROBE_PASSED = True
        _VOICES_PROBE_LAST_ERROR = None
        return
    if await _run_voices_probe_once():
        return
    if _VOICES_PROBE_RETRY_TASK is None or _VOICES_PROBE_RETRY_TASK.done():
        _VOICES_PROBE_RETRY_TASK = asyncio.create_task(_retry_voices_probe_until_ready())


@app.on_event("shutdown")
async def shutdown_event():
    """Cancel the background probe retry task when the app stops."""
    global _VOICES_PROBE_RETRY_TASK
    if _VOICES_PROBE_RETRY_TASK is not None:
        _VOICES_PROBE_RETRY_TASK.cancel()
        _VOICES_PROBE_RETRY_TASK = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "WhisperX ASR API",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "serve_mode": SERVE_MODE,
    }


@app.post("/asr")
async def transcribe_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    task: str = Query("transcribe"),
    language: Optional[str] = Query(None),
    initial_prompt: Optional[str] = Query(None),
    hotwords: Optional[str] = Query(None),
    word_timestamps: bool = Query(True),
    output_format: str = Query("json"),
    output: Optional[str] = Query(None),
    model: str = Query(DEFAULT_MODEL),
    num_speakers: Optional[int] = Query(None),
    min_speakers: Optional[int] = Query(None),
    max_speakers: Optional[int] = Query(None),
    diarize: Optional[bool] = Query(None),
    enable_diarization: Optional[bool] = Query(None),
    return_speaker_embeddings: Optional[bool] = Query(None),
):
    """
    Main ASR endpoint compatible with openai-whisper-asr-webservice

    Args:
        audio_file: Audio file to transcribe
        task: transcribe or translate
        language: Language code (e.g., 'en', 'es', 'fr')
        initial_prompt: Optional prompt to guide the model
        word_timestamps: Return word-level timestamps
        output_format: json, text, srt, vtt, or tsv
        model: WhisperX model name (tiny, base, small, medium, large-v2, large-v3)
        num_speakers: Exact number of speakers (if known, overrides min/max)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        diarize: Enable speaker diarization (compatible with whisper-asr-webservice)
        enable_diarization: Alias for diarize (deprecated, use diarize instead)
        return_speaker_embeddings: Return speaker embeddings (256-dimensional vectors)
    """
    temp_audio_path = None

    try:
        # Handle legacy parameter names
        if output is not None:
            output_format = output

        # Resolve diarization toggle
        if diarize is not None or enable_diarization is not None:
            should_diarize = (diarize is True) or (enable_diarization is True)
        else:
            should_diarize = True
        if return_speaker_embeddings is None:
            return_speaker_embeddings = False

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as temp_file:
            temp_audio_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB. "
                       f"Large files may cause out-of-memory errors."
            )

        if file_size_mb > 100:
            logger.warning(f"Processing large file ({file_size_mb:.1f}MB) - may consume significant VRAM")

        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "ASR request start client=%s file=%s size_mb=%.1f model=%s language=%s diarize=%s output=%s embeddings=%s",
            client_host,
            audio_file.filename,
            file_size_mb,
            model,
            language,
            should_diarize,
            output_format,
            return_speaker_embeddings,
        )

        # Load audio
        audio = whisperx.load_audio(temp_audio_path)

        # Run pipeline through the async queue (GPU semaphore)
        result, speaker_embeddings = await run_in_queue(
            run_pipeline,
            audio,
            model_name=model,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            word_timestamps=word_timestamps,
            should_diarize=should_diarize,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings,
        )

        detected_language = result.get("language", language or "en")

        # Format output based on requested format
        if output_format == "json":
            response_data = {
                "text": result.get("segments", []),
                "language": detected_language,
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", [])
            }

            if return_speaker_embeddings and speaker_embeddings:
                response_data["speaker_embeddings"] = sanitize_float_values(speaker_embeddings)
                logger.info(f"Including speaker embeddings in response: {list(speaker_embeddings.keys())}")

            logger.info(
                "ASR request complete client=%s file=%s segments=%d language=%s",
                client_host,
                audio_file.filename,
                len(result.get("segments", [])),
                detected_language,
            )
            return JSONResponse(content=response_data)

        elif output_format == "text":
            text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
            return {"text": text}

        elif output_format == "srt":
            srt_content = []
            for i, segment in enumerate(result.get("segments", []), 1):
                start_time = format_timestamp(segment.get("start", 0))
                end_time = format_timestamp(segment.get("end", 0))
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

            return {"srt": "\n".join(srt_content)}

        elif output_format == "vtt":
            vtt_content = ["WEBVTT\n"]
            for segment in result.get("segments", []):
                start_time = format_timestamp(segment.get("start", 0)).replace(',', '.')
                end_time = format_timestamp(segment.get("end", 0)).replace(',', '.')
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")

                if speaker:
                    text = f"[{speaker}] {text}"

                vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")

            return {"vtt": "\n".join(vtt_content)}

        elif output_format == "tsv":
            tsv_content = ["start\tend\ttext\tspeaker"]
            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "")
                tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")

            return {"tsv": "\n".join(tsv_content)}

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported output format: {output_format}")

    except Exception as e:
        logger.error(
            "ASR request failed file=%s error=%s",
            audio_file.filename,
            str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring.

    PAI fork: returns 503 until the startup embedding probe has passed.
    """
    if not _VOICES_PROBE_PASSED:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "detail": "pai-voices embedding probe has not passed yet",
                "probe_attempts": _VOICES_PROBE_ATTEMPTS,
                "probe_error": _VOICES_PROBE_LAST_ERROR,
            },
        )
    return {
        "status": "healthy",
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
        "serve_mode": SERVE_MODE,
        "probe_attempts": _VOICES_PROBE_ATTEMPTS,
    }


@app.get("/metrics")
async def metrics():
    """Queue and pipeline metrics"""
    data = {
        "serve_mode": SERVE_MODE,
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
    }
    if SERVE_MODE == "simple":
        data["queue"] = get_queue_metrics()
    return data


# Register OpenAI-compatible API routers
# Import here to avoid circular imports (openai_compat imports from this module)
from app.openai_compat import router as openai_router, models_router
app.include_router(openai_router)
app.include_router(models_router)

# PAI fork: mount /embed and /voices/{name}.
app.include_router(voices_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
