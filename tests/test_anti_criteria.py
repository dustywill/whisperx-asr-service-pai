"""Anti-criteria — ISC-A1..ISC-A6.

These tests assert what the fork must NOT do. They are deliberately static
(source-grep + route-inspection) so they cannot be satisfied by runtime behavior
alone; the code has to actually stay clean.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest


APP_DIR = Path(__file__).resolve().parent.parent / "app"
FORK_FILES = [
    APP_DIR / "voices.py",
    APP_DIR / "voices_storage.py",
    APP_DIR / "voices_config.py",
    APP_DIR / "voices_embed.py",
    APP_DIR / "name_validation.py",
]


# ---------------------------------------------------------------------------
# ISC-A1: no GET or DELETE route under /voices
# ---------------------------------------------------------------------------


def test_no_get_or_delete_voices_routes(client):
    """Fork scope is APPEND-ONLY: no read or delete endpoints."""
    from app.main import app

    for r in app.routes:
        path = getattr(r, "path", "") or ""
        methods = getattr(r, "methods", None) or set()
        if path.startswith("/voices"):
            assert "GET" not in methods, f"forbidden GET on {path}: {methods}"
            assert "DELETE" not in methods, f"forbidden DELETE on {path}: {methods}"


# ---------------------------------------------------------------------------
# ISC-A2: no cosine-similarity math in app/voices.py
# ---------------------------------------------------------------------------


def test_voices_py_does_not_compute_similarity():
    src = (APP_DIR / "voices.py").read_text(encoding="utf-8").lower()
    assert "cosine" not in src, "voices.py must not contain cosine similarity logic"
    assert "similarity" not in src, "voices.py must not compute similarity scores"


# ---------------------------------------------------------------------------
# ISC-A4: no SQLite in fork-added files
# ---------------------------------------------------------------------------


def test_no_sqlite_in_fork_files():
    for f in FORK_FILES:
        src = f.read_text(encoding="utf-8").lower()
        assert "sqlite" not in src, f"{f.name} must not reference sqlite"


# ---------------------------------------------------------------------------
# ISC-A5: no additional pyannote model imports in voices*.py
# (app/pipeline.py is the sole loader.)
# ---------------------------------------------------------------------------


def test_voices_modules_do_not_import_pyannote_models():
    pattern = re.compile(r"^\s*(from|import)\s+pyannote", re.MULTILINE)
    for f in FORK_FILES:
        if not f.name.startswith("voices"):
            continue
        src = f.read_text(encoding="utf-8")
        matches = pattern.findall(src)
        assert not matches, f"{f.name} must not import pyannote directly: {matches}"


# ---------------------------------------------------------------------------
# ISC-A6: _LIB_LOCK is the sole asyncio write gate.
# ---------------------------------------------------------------------------


def test_lib_lock_is_an_asyncio_lock():
    """Docstring: the voices_storage._LIB_LOCK serializes all library mutations
    within a single process and is the sole write gate (per the module docstring).
    This asserts its type so a future refactor can't silently swap it for a
    non-async lock and drop concurrency guarantees."""
    from app import voices_storage as storage

    assert isinstance(storage._LIB_LOCK, asyncio.Lock)
