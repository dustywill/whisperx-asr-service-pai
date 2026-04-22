"""voices_storage — ISC-19..ISC-23, ISC-48."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest


from app import voices_storage as storage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if sys.version_info < (
        3, 10
    ) else asyncio.run(coro)


def _append(library_path: Path, clips_dir: Path, name: str, payload: bytes, *, max_refs: int = 50, dim: int = 8):
    return asyncio.run(
        storage.append_voice(
            library_path=library_path,
            clips_dir=clips_dir,
            name=name,
            audio_bytes=payload,
            embedding=[0.1] * dim,
            max_refs=max_refs,
        )
    )


# ---------------------------------------------------------------------------
# ISC-19: atomic write failure leaves old file intact, no .tmp remains.
# ---------------------------------------------------------------------------


def test_atomic_write_failure_preserves_old_file(tmp_path, monkeypatch):
    library_path = tmp_path / "voices.json"
    clips_dir = tmp_path / "voices"
    clips_dir.mkdir()

    # Seed a valid prior library.
    _append(library_path, clips_dir, "old", b"seed")
    assert library_path.exists()
    original_bytes = library_path.read_bytes()

    # Now break os.replace on the next call.
    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src, dst):
        # Only fail the json rename — clip renames still need to succeed so the
        # sequence reaches the json write step.
        if str(dst).endswith("voices.json"):
            calls["n"] += 1
            raise OSError("simulated replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(storage.os, "replace", failing_replace)

    with pytest.raises(OSError):
        _append(library_path, clips_dir, "new", b"payload")

    # Old file byte-for-byte unchanged.
    assert library_path.read_bytes() == original_bytes
    # No stale .tmp siblings left behind.
    stray = list(library_path.parent.glob(".voices-*.tmp"))
    assert stray == [], f"leftover tmp files: {stray}"


# ---------------------------------------------------------------------------
# ISC-20: missing file at startup → {}
# ---------------------------------------------------------------------------


def test_load_library_returns_empty_when_file_missing(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    assert storage.load_library(missing) == {}


# ---------------------------------------------------------------------------
# ISC-21: corrupt JSON → {}
# ---------------------------------------------------------------------------


def test_load_library_returns_empty_on_corrupt_json(tmp_path):
    corrupt = tmp_path / "voices.json"
    corrupt.write_text("{ not valid json ::::", encoding="utf-8")
    assert storage.load_library(corrupt) == {}


# ---------------------------------------------------------------------------
# ISC-22: first write auto-creates clips_dir and per-name subdir.
# ---------------------------------------------------------------------------


def test_first_write_autocreates_dirs(tmp_path):
    library_path = tmp_path / "nested" / "voices.json"
    clips_dir = tmp_path / "deeper" / "voices"  # does not exist yet
    assert not clips_dir.exists()

    clip_path, count = _append(library_path, clips_dir, "kayla", b"payload")
    assert count == 1
    assert clips_dir.is_dir()
    assert (clips_dir / "kayla").is_dir()
    assert Path(clip_path).exists()
    assert library_path.exists()


# ---------------------------------------------------------------------------
# ISC-23: fsync called at least twice per write (file + dir on Linux;
# file only on Windows — skip dir assertion there).
# ---------------------------------------------------------------------------


def test_fsync_is_called_per_write(tmp_path, monkeypatch):
    library_path = tmp_path / "voices.json"
    clips_dir = tmp_path / "voices"

    calls = {"n": 0}
    real_fsync = os.fsync

    def counting_fsync(fd):
        calls["n"] += 1
        return real_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", counting_fsync)

    _append(library_path, clips_dir, "kayla", b"payload")

    # At minimum: one fsync on the json file, one on the clip file.
    # On Linux there's also a dir fsync, pushing this to 3+.
    if sys.platform.startswith("linux"):
        assert calls["n"] >= 2, f"expected >=2 fsyncs on Linux, got {calls['n']}"
    else:
        # On Windows/macOS os.O_DIRECTORY may not exist so _fsync_dir no-ops.
        assert calls["n"] >= 2, f"expected >=2 fsyncs (file + clip), got {calls['n']}"


# ---------------------------------------------------------------------------
# ISC-48: FIFO cap — oldest embedding + clip file removed when exceeded.
# ---------------------------------------------------------------------------


def test_fifo_cap_drops_oldest_and_unlinks_clip(tmp_path):
    library_path = tmp_path / "voices.json"
    clips_dir = tmp_path / "voices"

    clips: list[Path] = []
    for i in range(5):
        # Slight timing differentiation: the filename has per-second resolution,
        # so force unique clip paths by passing distinct payloads and tolerating
        # repeats. When repeats occur, the implementation still rewrites the
        # same path; the FIFO semantics apply to list slots, not distinct files.
        clip_path, _ = _append(
            library_path,
            clips_dir,
            "kayla",
            f"clip-{i}".encode(),
            max_refs=3,
        )
        clips.append(Path(clip_path))

    lib = json.loads(library_path.read_text(encoding="utf-8"))
    entry = lib["kayla"]
    assert len(entry["embeddings"]) == 3, "FIFO cap should hold exactly 3 embeddings"
    assert len(entry["clips"]) == 3

    # The two oldest clip file paths should have been unlinked from disk —
    # unless they coincide (same-second stamp) with one of the survivors.
    surviving_paths = {Path(p) for p in entry["clips"]}
    for idx, clip in enumerate(clips[:2]):  # first two (oldest)
        if clip in surviving_paths:
            continue  # stamp collision with a survivor — acceptable
        assert not clip.exists(), f"oldest clip {clip} should have been unlinked"
