"""PAI fork additions: runtime configuration for voice library endpoints.

Read from env at call time (not import time) so tests can patch env vars with
`monkeypatch.setenv(...)` inside fixtures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceSettings:
    library_path: Path
    clips_dir: Path
    max_clip_seconds: float
    max_references_per_name: int
    uvicorn_workers: int
    lock_strategy: str
    hf_token: str | None


def get_settings() -> VoiceSettings:
    return VoiceSettings(
        library_path=Path(os.getenv("VOICE_LIBRARY_PATH", "/data/voices.json")),
        clips_dir=Path(os.getenv("VOICE_CLIPS_DIR", "/data/voices")),
        max_clip_seconds=float(os.getenv("EMBEDDING_MAX_CLIP_SECONDS", "30")),
        max_references_per_name=int(os.getenv("MAX_REFERENCES_PER_NAME", "50")),
        uvicorn_workers=int(os.getenv("UVICORN_WORKERS", "1")),
        lock_strategy=os.getenv("VOICE_LOCK_STRATEGY", "asyncio"),
        hf_token=os.getenv("HF_TOKEN") or None,
    )


def assert_hf_token_present() -> None:
    """Fail-fast if HF_TOKEN is missing. Called from container entrypoint."""
    if not os.getenv("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN environment variable is required (pyannote model access). "
            "Refusing to start."
        )


def assert_worker_count_safe() -> None:
    """Refuse to start with workers>1 unless flock strategy explicitly chosen."""
    workers = int(os.getenv("UVICORN_WORKERS", "1"))
    strategy = os.getenv("VOICE_LOCK_STRATEGY", "asyncio")
    if workers > 1 and strategy != "flock":
        raise SystemExit(
            f"UVICORN_WORKERS={workers} with VOICE_LOCK_STRATEGY={strategy}: "
            "asyncio.Lock is per-process and does not serialize across workers. "
            "Set VOICE_LOCK_STRATEGY=flock to opt in to cross-process locking, "
            "or keep UVICORN_WORKERS=1."
        )
