"""Pytest fixtures for the pai-voices fork.

Design notes
------------
- The upstream app pulls in ``whisperx``, ``torch``, and ``pyannote.audio`` at
  import time via ``app.pipeline``, ``app.main``, and ``app.openai_compat``.
  Tests are intended to run on a developer laptop without CUDA / pyannote, so
  this conftest installs ``MagicMock`` stand-ins in ``sys.modules`` **before**
  any ``app.*`` import occurs.
- ``PAI_VOICES_SKIP_PROBE=1`` is set at session start so FastAPI startup does
  not try to load the real diarization pipeline.
- The ``client`` fixture defers ``from app.main import app`` until after the
  stubs are in place.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Session-level import shielding — must run before any `app.*` import.
# ---------------------------------------------------------------------------

def _install_heavy_stubs() -> None:
    """Inject MagicMock modules for heavy deps so `import app.main` succeeds."""

    # Skip the real startup embedding probe.
    os.environ.setdefault("PAI_VOICES_SKIP_PROBE", "1")
    # Upstream reads these at import time; supply harmless defaults.
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("COMPUTE_TYPE", "int8")
    os.environ.setdefault("HF_TOKEN", "test-token")

    # whisperx + submodule
    if "whisperx" not in sys.modules:
        whisperx_stub = MagicMock(name="whisperx")
        whisperx_stub.load_audio = MagicMock(return_value=None)
        sys.modules["whisperx"] = whisperx_stub
    if "whisperx.diarize" not in sys.modules:
        diarize_stub = MagicMock(name="whisperx.diarize")
        diarize_stub.DiarizationPipeline = MagicMock()
        sys.modules["whisperx.diarize"] = diarize_stub

    # torch — upstream reads torch.cuda.is_available() at import time.
    if "torch" not in sys.modules:
        torch_stub = MagicMock(name="torch")
        torch_stub.cuda = MagicMock()
        torch_stub.cuda.is_available = MagicMock(return_value=False)
        sys.modules["torch"] = torch_stub

    # pyannote.audio (not directly imported by the fork but defensive).
    if "pyannote" not in sys.modules:
        sys.modules["pyannote"] = ModuleType("pyannote")
    if "pyannote.audio" not in sys.modules:
        sys.modules["pyannote.audio"] = MagicMock(name="pyannote.audio")

    # soundfile — imported lazily inside voices_embed.probe_clip_duration, but
    # stub anyway so any accidental real call fails loudly.
    if "soundfile" not in sys.modules:
        sys.modules["soundfile"] = MagicMock(name="soundfile")


_install_heavy_stubs()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_voice_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point VOICE_LIBRARY_PATH and VOICE_CLIPS_DIR at a tmp subtree."""
    library_path = tmp_path / "voices.json"
    clips_dir = tmp_path / "voices"
    clips_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOICE_LIBRARY_PATH", str(library_path))
    monkeypatch.setenv("VOICE_CLIPS_DIR", str(clips_dir))
    return {
        "tmp_path": tmp_path,
        "library_path": library_path,
        "clips_dir": clips_dir,
    }


@pytest.fixture()
def fake_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Replace `app.voices._get_pipeline` with a cheap sentinel + call counter.

    Returns the MagicMock so tests can assert ``.call_count``.
    """
    from app import voices as voices_mod

    sentinel = object()
    mock = MagicMock(return_value=sentinel)
    monkeypatch.setattr(voices_mod, "_get_pipeline", mock)
    return mock


@pytest.fixture()
def fake_embedder(monkeypatch: pytest.MonkeyPatch):
    """Stub out the heavy embedding / decoding helpers.

    - ``extract_embedding`` → deterministic 256-float vector.
    - ``decode_to_mono_16k`` → 1 second of float32 silence.
    - ``probe_clip_duration`` → 1.0 (safely under the 30s default cap).
    """
    import numpy as np

    from app import voices_embed as emb_mod

    vec = [0.01] * 256
    monkeypatch.setattr(emb_mod, "extract_embedding", lambda pipeline, audio: list(vec))
    monkeypatch.setattr(
        emb_mod,
        "decode_to_mono_16k",
        lambda audio_bytes: np.full(16000, 0.1, dtype=np.float32),
    )
    monkeypatch.setattr(emb_mod, "probe_clip_duration", lambda audio_bytes: 1.0)

    # The voices module imports these names at module-load time, so patch the
    # binding in that module too.
    from app import voices as voices_mod

    monkeypatch.setattr(
        voices_mod,
        "extract_embedding",
        lambda pipeline, audio: list(vec),
    )
    monkeypatch.setattr(
        voices_mod,
        "decode_to_mono_16k",
        lambda audio_bytes: np.full(16000, 0.1, dtype=np.float32),
    )
    monkeypatch.setattr(voices_mod, "probe_clip_duration", lambda audio_bytes: 1.0)

    return {"vector": list(vec), "dim": 256}


@pytest.fixture()
def client(tmp_voice_dirs, fake_pipeline, fake_embedder, monkeypatch):
    """Return a TestClient for the real FastAPI app.

    Env must be configured and heavy modules stubbed BEFORE `from app.main
    import app`, so the import happens lazily inside this fixture.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PAI_VOICES_SKIP_PROBE", "1")

    # Force re-import to pick up fresh env / monkeypatches where relevant.
    # We only force re-import for app.main (the app instance); sub-modules that
    # expose state (voices_storage._LIB_LOCK) should persist across tests.
    if "app.main" in sys.modules:
        del sys.modules["app.main"]

    from app.main import app  # noqa: WPS433 — intentional late import

    with TestClient(app) as tc:
        yield tc
