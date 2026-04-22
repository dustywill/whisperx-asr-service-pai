"""voices_config — ISC-24..ISC-27, ISC-49."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import voices_config as cfg


# ---------------------------------------------------------------------------
# ISC-24: defaults
# ---------------------------------------------------------------------------


def test_defaults_when_no_env_set(monkeypatch):
    for var in (
        "VOICE_LIBRARY_PATH",
        "VOICE_CLIPS_DIR",
        "EMBEDDING_MAX_CLIP_SECONDS",
        "MAX_REFERENCES_PER_NAME",
        "UVICORN_WORKERS",
        "VOICE_LOCK_STRATEGY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)

    s = cfg.get_settings()
    assert s.library_path == Path("/data/voices.json")
    assert s.clips_dir == Path("/data/voices")
    assert s.max_clip_seconds == 30.0
    assert s.max_references_per_name == 50
    assert s.uvicorn_workers == 1
    assert s.lock_strategy == "asyncio"
    assert s.hf_token is None


# ---------------------------------------------------------------------------
# ISC-25: env vars override defaults
# ---------------------------------------------------------------------------


def test_env_overrides_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICE_LIBRARY_PATH", str(tmp_path / "lib.json"))
    monkeypatch.setenv("VOICE_CLIPS_DIR", str(tmp_path / "clips"))
    monkeypatch.setenv("EMBEDDING_MAX_CLIP_SECONDS", "7.5")
    monkeypatch.setenv("MAX_REFERENCES_PER_NAME", "12")
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    monkeypatch.setenv("VOICE_LOCK_STRATEGY", "flock")
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")

    s = cfg.get_settings()
    assert s.library_path == tmp_path / "lib.json"
    assert s.clips_dir == tmp_path / "clips"
    assert s.max_clip_seconds == 7.5
    assert s.max_references_per_name == 12
    assert s.uvicorn_workers == 4
    assert s.lock_strategy == "flock"
    assert s.hf_token == "hf_xxx"


# ---------------------------------------------------------------------------
# ISC-26: assert_hf_token_present raises SystemExit when unset
# ---------------------------------------------------------------------------


def test_assert_hf_token_present_raises_when_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        cfg.assert_hf_token_present()


def test_assert_hf_token_present_ok_when_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    # No exception expected.
    cfg.assert_hf_token_present()


# ---------------------------------------------------------------------------
# ISC-27/ISC-49: assert_worker_count_safe
# ---------------------------------------------------------------------------


def test_assert_worker_count_safe_raises_on_asyncio_with_multi_workers(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("VOICE_LOCK_STRATEGY", "asyncio")
    with pytest.raises(SystemExit):
        cfg.assert_worker_count_safe()


def test_assert_worker_count_safe_allows_flock_with_multi_workers(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "2")
    monkeypatch.setenv("VOICE_LOCK_STRATEGY", "flock")
    # No exception expected.
    cfg.assert_worker_count_safe()


def test_assert_worker_count_safe_allows_single_worker_default(monkeypatch):
    monkeypatch.setenv("UVICORN_WORKERS", "1")
    monkeypatch.delenv("VOICE_LOCK_STRATEGY", raising=False)
    cfg.assert_worker_count_safe()
