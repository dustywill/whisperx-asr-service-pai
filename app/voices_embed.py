"""PAI fork additions: embedding extraction + clip duration probe.

Isolated in its own module so tests can monkeypatch `extract_embedding` without
importing the heavy pyannote/whisperx stack.
"""

from __future__ import annotations

import io
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


MODEL_VERSION = "pyannote/speaker-diarization-community-1"
TARGET_SAMPLE_RATE = 16000


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

    Uses soundfile.info() on a tempfile so the multipart upload path stays bounded.
    """
    import soundfile as sf  # local import: heavy dep

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        info = sf.info(tmp_path)
        return float(info.frames) / float(info.samplerate)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


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
