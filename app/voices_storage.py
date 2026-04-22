"""PAI fork additions: atomic storage for the voice library.

Schema (matches parent PRD 20260421-202617 DESIGN.md):

    {
      "kayla": {
        "embeddings": [[0.12, ...], ...],
        "clips":      ["/data/voices/kayla/clip-2026-04-21T20-47-03.wav", ...],
        "updated_at": "2026-04-21T20:47:03Z"
      },
      ...
    }

Concurrency: an asyncio.Lock serializes writers within a single process.
Multi-worker deployments MUST set VOICE_LOCK_STRATEGY=flock (see
voices_config.assert_worker_count_safe); the lock itself is single-process.

Durability: JSON rewrites use write-to-temp + os.replace; both file and parent
directory are fsynced. Audio clip writes fsync before rename.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_LIB_LOCK = asyncio.Lock()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clip_stamp() -> str:
    # Filesystem-safe ISO-ish timestamp (colons replaced).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def load_library(path: Path) -> Dict[str, Any]:
    """Load the library; treat missing/corrupt file as empty per ISC-20."""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _fsync_dir(dir_path: Path) -> None:
    """fsync a directory so a rename is durable. No-op on platforms without O_DIRECTORY."""
    flag = getattr(os, "O_DIRECTORY", None)
    if flag is None:
        return
    fd = os.open(str(dir_path), flag)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".voices-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _write_clip(clips_dir: Path, name: str, audio_bytes: bytes) -> Path:
    name_dir = clips_dir / name
    name_dir.mkdir(parents=True, exist_ok=True)
    clip_path = name_dir / f"clip-{_clip_stamp()}.wav"
    fd, tmp = tempfile.mkstemp(prefix=".clip-", suffix=".wav.tmp", dir=str(name_dir))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, clip_path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise
    return clip_path


async def append_voice(
    *,
    library_path: Path,
    clips_dir: Path,
    name: str,
    audio_bytes: bytes,
    embedding: List[float],
    max_refs: int,
) -> Tuple[Path, int]:
    """Serialize mutation; persist clip; append embedding; enforce FIFO cap.

    Returns (clip_path, post-write reference count).
    """
    async with _LIB_LOCK:
        lib = load_library(library_path)
        entry = lib.get(name) or {"embeddings": [], "clips": [], "updated_at": ""}

        clip_path = _write_clip(clips_dir, name, audio_bytes)
        entry["embeddings"].append([float(x) for x in embedding])
        entry["clips"].append(str(clip_path))

        # FIFO cap: drop oldest embedding + its clip file.
        while len(entry["embeddings"]) > max_refs:
            entry["embeddings"].pop(0)
            dropped = entry["clips"].pop(0)
            try:
                os.unlink(dropped)
            except OSError:
                pass  # Already gone — non-fatal.

        entry["updated_at"] = _utc_iso()
        lib[name] = entry
        _atomic_write_json(library_path, lib)
        return clip_path, len(entry["embeddings"])
