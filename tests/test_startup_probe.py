"""Startup probe + /health gating — ISC-28, ISC-30, ISC-50."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# ISC-28: passing probe returns (dim, norm) with norm > 0
# ---------------------------------------------------------------------------


def test_run_startup_probe_passes_with_nontrivial_vector():
    from app.voices_embed import run_startup_probe

    loader = MagicMock(return_value=object())
    fake_embedder = lambda pipeline, audio: [0.1] * 256

    dim, norm = run_startup_probe(loader, embed_fn=fake_embedder)
    assert dim == 256
    assert norm > 0
    assert loader.call_count == 1


# ---------------------------------------------------------------------------
# ISC-30: failing probe (zero vector) raises RuntimeError
# ---------------------------------------------------------------------------


def test_run_startup_probe_fails_on_zero_vector():
    from app.voices_embed import run_startup_probe

    loader = MagicMock(return_value=object())
    zero_embedder = lambda pipeline, audio: [0.0] * 256

    with pytest.raises(RuntimeError):
        run_startup_probe(loader, embed_fn=zero_embedder)


# ---------------------------------------------------------------------------
# ISC-50: pipeline loader called exactly once per probe
# ---------------------------------------------------------------------------


def test_run_startup_probe_loads_pipeline_once():
    from app.voices_embed import run_startup_probe

    loader = MagicMock(return_value=object())
    embedder = lambda pipeline, audio: [0.2] * 16

    run_startup_probe(loader, embed_fn=embedder)
    assert loader.call_count == 1


# ---------------------------------------------------------------------------
# /health gating
# ---------------------------------------------------------------------------


def test_health_returns_200_when_probe_is_skipped(client):
    """With PAI_VOICES_SKIP_PROBE=1 (set in conftest), /health should be 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "healthy"


def test_health_returns_503_when_probe_has_not_passed(monkeypatch, tmp_voice_dirs, fake_pipeline, fake_embedder):
    """With PAI_VOICES_SKIP_PROBE=0 and _VOICES_PROBE_PASSED=False, /health is 503."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PAI_VOICES_SKIP_PROBE", "0")

    # Force a fresh import of app.main so startup re-runs with the new env
    # *and* fails — by making the pipeline loader raise inside the probe.
    if "app.main" in sys.modules:
        del sys.modules["app.main"]

    # Make the probe fail deterministically: loader raises.
    from app import pipeline as pipeline_mod

    def boom():
        raise RuntimeError("no pyannote in test env")

    monkeypatch.setattr(pipeline_mod, "load_diarize_pipeline", boom, raising=False)

    from app.main import app as fresh_app
    import app.main as main_mod

    with TestClient(fresh_app) as tc:
        # Sanity: probe should have failed, leaving flag False.
        assert main_mod._VOICES_PROBE_PASSED is False
        resp = tc.get("/health")
        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body.get("status") == "starting"
