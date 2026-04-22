"""Optional live integration smoke for the deployed PAI fork.

This test is intentionally skipped by default. It is meant for TutelarAlien or
another live host with a real HF token, real pyannote weights, and the startup
probe enabled (`PAI_VOICES_SKIP_PROBE=0`).
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import uuid
import urllib.error
import urllib.request
from pathlib import Path

import pytest


RUN_E2E = os.getenv("RUN_E2E") == "1"
BASE_URL = os.getenv("PAI_LIVE_BASE_URL", "http://localhost:9000").rstrip("/")
LIVE_AUDIO_FILE = os.getenv("PAI_LIVE_AUDIO_FILE")

pytestmark = pytest.mark.skipif(
    not RUN_E2E,
    reason="set RUN_E2E=1 to run live integration smoke tests",
)


def _require_audio_file() -> Path:
    if not LIVE_AUDIO_FILE:
        pytest.skip("set PAI_LIVE_AUDIO_FILE to a real reference clip")
    path = Path(LIVE_AUDIO_FILE)
    if not path.exists():
        pytest.skip(f"live audio file not found: {path}")
    return path


def _multipart_body(field_name: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"pai-live-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    content = file_path.read_bytes()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + content + tail, boundary


def _get_json(url: str) -> tuple[int, dict]:
    with urllib.request.urlopen(url, timeout=30) as response:
        body = response.read().decode("utf-8")
        return response.status, json.loads(body)


def _post_embed(file_path: Path) -> tuple[int, dict]:
    body, boundary = _multipart_body("audio_file", file_path)
    request = urllib.request.Request(
        f"{BASE_URL}/embed",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        return exc.code, payload


def test_live_health_is_green():
    status, body = _get_json(f"{BASE_URL}/health")
    assert status == 200
    assert body.get("status") == "healthy"


def test_live_embed_returns_nontrivial_vector():
    file_path = _require_audio_file()
    status, body = _post_embed(file_path)

    assert status == 200, body
    assert body["dim"] > 0
    assert len(body["embedding"]) == body["dim"]
    assert all(isinstance(x, float) for x in body["embedding"])

    norm = math.sqrt(sum((x * x) for x in body["embedding"]))
    assert norm > 1e-6
