"""PAI fork additions: embedding extraction + clip duration probe.

Isolated in its own module so tests can monkeypatch `extract_embedding` without
importing the heavy pyannote/whisperx stack.
"""

from __future__ import annotations

import logging
import json
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


MODEL_VERSION = "pyannote/speaker-diarization-community-1"
TARGET_SAMPLE_RATE = 16000

# RMS amplitude below this is treated as silent (~ -50 dBFS for float32 PCM in
# [-1, 1]). Used to short-circuit /embed before invoking the diarization
# pipeline, which raises on all-silence input.
SILENCE_RMS_THRESHOLD = 0.003


def is_silent(audio_np: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """Return True if the waveform's RMS is below `threshold`.

    Operates on the same mono float32 16 kHz array produced by
    `decode_to_mono_16k`. Empty arrays count as silent.
    """
    if audio_np is None:
        return True
    arr = np.asarray(audio_np, dtype=np.float32).ravel()
    if arr.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(arr))))
    return rms < threshold


def _synthetic_probe_audio() -> np.ndarray:
    """Generate a short voiced-like clip for startup validation fallback."""
    t = np.linspace(0.0, 1.0, TARGET_SAMPLE_RATE, endpoint=False, dtype=np.float32)
    signal = (
        0.18 * np.sin(2 * np.pi * 220.0 * t)
        + 0.09 * np.sin(2 * np.pi * 440.0 * t)
        + 0.04 * np.sin(2 * np.pi * 660.0 * t)
    )
    fade = np.minimum(t / 0.05, (1.0 - t) / 0.05)
    fade = np.clip(fade, 0.0, 1.0).astype(np.float32)
    return (signal * fade).astype(np.float32)


def probe_clip_duration(audio_bytes: bytes) -> float:
    """Return clip duration in seconds without fully decoding.

    Prefer ffprobe because ffmpeg is already part of the service image.
    Fall back to soundfile when ffprobe cannot inspect the upload.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        try:
            return _probe_duration_ffprobe(tmp_path)
        except Exception as ffprobe_exc:  # noqa: BLE001
            logger.warning("ffprobe duration probe failed; falling back to soundfile: %s", ffprobe_exc)
            return _probe_duration_soundfile(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _probe_duration_ffprobe(tmp_path: str) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            tmp_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"ffprobe exit {proc.returncode}"
        raise RuntimeError(detail)
    payload = json.loads(proc.stdout or "{}")
    duration_raw = payload.get("format", {}).get("duration")
    if duration_raw in (None, ""):
        raise RuntimeError("ffprobe returned no duration")
    duration = float(duration_raw)
    if duration <= 0:
        raise RuntimeError(f"ffprobe returned non-positive duration {duration}")
    return duration


def _probe_duration_soundfile(tmp_path: str) -> float:
    import soundfile as sf  # local import: optional fallback

    info = sf.info(tmp_path)
    if not info.samplerate:
        raise RuntimeError("soundfile returned samplerate 0")
    return float(info.frames) / float(info.samplerate)


def decode_to_mono_16k(audio_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes to a mono float32 waveform at 16 kHz.

    Uses whisperx.load_audio which shells out to ffmpeg — matches /asr semantics.
    """
    import whisperx  # local import: heavy dep

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        audio = whisperx.load_audio(tmp_path)
        return np.asarray(audio, dtype=np.float32)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_embedding(pipeline, audio_np: np.ndarray) -> List[float]:
    """Run the already-loaded diarization pipeline with return_embeddings=True
    on a (presumed single-speaker) reference clip and return the first
    speaker's embedding as a flat list of floats.

    NOTE: for multi-speaker clips this returns only the first speaker's vector;
    the queue app is responsible for slicing per-speaker reference clips before
    calling /embed (parent PRD worker-stage S6).
    """
    # whisperx.diarize.DiarizationPipeline expects a raw mono waveform array.
    # It wraps this into the pyannote dict shape internally.
    diarize_out = pipeline(audio_np, return_embeddings=True)
    if not (isinstance(diarize_out, tuple) and len(diarize_out) == 2):
        raise RuntimeError("diarization pipeline did not return embeddings tuple")
    _, embeds = diarize_out
    vec = _first_embedding_vector(embeds)
    return [float(x) for x in vec.tolist()]


def _first_embedding_vector(embeds) -> np.ndarray:
    """Extract the first speaker vector from the shapes pyannote may return.

    Different pyannote / whisperx combinations may surface speaker embeddings as
    a dict-like mapping, a 2D ndarray/torch tensor, or a wrapper object with a
    `.data` ndarray. Normalize all of them here so both `/embed` and the
    startup probe share one compatibility shim.
    """
    if embeds is None:
        raise RuntimeError("diarization produced no speaker embeddings")

    if hasattr(embeds, "detach"):
        embeds = embeds.detach().cpu().numpy()

    if isinstance(embeds, Mapping):
        if not embeds:
            raise RuntimeError("diarization produced no speakers")
        first_key = next(iter(embeds))
        return np.asarray(embeds[first_key]).flatten()

    if hasattr(embeds, "data"):
        data = np.asarray(embeds.data)
        if data.size == 0:
            raise RuntimeError("diarization produced no speakers")
        return data[0].flatten() if data.ndim > 1 else data.flatten()

    data = np.asarray(embeds)
    if data.size == 0:
        raise RuntimeError("diarization produced no speakers")
    return data[0].flatten() if data.ndim > 1 else data.flatten()


def run_startup_probe(pipeline_loader, embed_fn=extract_embedding) -> Tuple[int, float]:
    """Embed 1 second of silence; assert non-trivial norm; return (dim, norm).

    Raises RuntimeError if norm is below epsilon — catches silent model-load drift.
    """
    pipeline = pipeline_loader()
    try:
        vec = embed_fn(pipeline, np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32))
        probe_kind = "silence"
    except RuntimeError as exc:
        if "no speakers" not in str(exc).lower():
            raise
        logger.warning(
            "[pai-voices] startup silence probe produced no speakers; retrying with synthetic voiced probe"
        )
        vec = embed_fn(pipeline, _synthetic_probe_audio())
        probe_kind = "synthetic"
    dim = len(vec)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        raise RuntimeError(f"startup probe: embedding norm {norm} below threshold")
    logger.info(
        "[pai-voices] startup embedding probe OK kind=%s dim=%d norm=%.4f model=%s",
        probe_kind, dim, norm, MODEL_VERSION,
    )
    return dim, norm
