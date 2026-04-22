"""PAI fork additions: embedding extraction + clip duration probe.

Isolated in its own module so tests can monkeypatch `extract_embedding` without
importing the heavy pyannote/whisperx stack.
"""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


MODEL_VERSION = "pyannote/speaker-diarization-community-1"
TARGET_SAMPLE_RATE = 16000


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
    import torch  # local import: heavy dep

    waveform = torch.from_numpy(audio_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    diarize_out = pipeline(
        {"waveform": waveform, "sample_rate": TARGET_SAMPLE_RATE},
        return_embeddings=True,
    )
    if not (isinstance(diarize_out, tuple) and len(diarize_out) == 2):
        raise RuntimeError("diarization pipeline did not return embeddings tuple")
    _, embeds = diarize_out
    if not embeds:
        raise RuntimeError("diarization produced no speakers")
    first_key = next(iter(embeds))
    vec = np.asarray(embeds[first_key]).flatten()
    return [float(x) for x in vec.tolist()]


def run_startup_probe(pipeline_loader, embed_fn=extract_embedding) -> Tuple[int, float]:
    """Embed 1 second of silence; assert non-trivial norm; return (dim, norm).

    Raises RuntimeError if norm is below epsilon — catches silent model-load drift.
    """
    silence = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    pipeline = pipeline_loader()
    vec = embed_fn(pipeline, silence)
    dim = len(vec)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        raise RuntimeError(f"startup probe: embedding norm {norm} below threshold")
    logger.info(
        "[pai-voices] startup embedding probe OK dim=%d norm=%.4f model=%s",
        dim, norm, MODEL_VERSION,
    )
    return dim, norm
