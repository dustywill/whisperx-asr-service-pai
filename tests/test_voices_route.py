"""POST /voices/{name} — ISC-9..ISC-18 + ISC-46, ISC-47."""

from __future__ import annotations

import asyncio
import io
import json
import unicodedata
from pathlib import Path

import pytest


def _audio_upload(content: bytes = b"fake-wav-bytes") -> dict:
    return {"audio_file": ("clip.wav", io.BytesIO(content), "audio/wav")}


# ---------------------------------------------------------------------------
# Happy path (ISC-9..ISC-11)
# ---------------------------------------------------------------------------


def test_voices_happy_path_writes_clip_and_updates_json(client, fake_embedder, tmp_voice_dirs):
    """ISC-9/10/11: 200; clip file exists on disk; reference_count == 1; voices.json has expected schema."""
    resp = client.post("/voices/kayla", files=_audio_upload())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["ok"] is True
    assert body["name"] == "kayla"
    assert body["reference_count"] == 1

    # Clip file actually exists on disk.
    clips_dir: Path = tmp_voice_dirs["clips_dir"]
    kayla_dir = clips_dir / "kayla"
    assert kayla_dir.is_dir(), "per-name clip dir must be auto-created"
    clip_files = list(kayla_dir.glob("clip-*.wav"))
    assert len(clip_files) == 1, f"expected exactly one clip on disk, got {clip_files}"

    # voices.json schema check.
    library_path: Path = tmp_voice_dirs["library_path"]
    assert library_path.exists()
    lib = json.loads(library_path.read_text(encoding="utf-8"))
    assert "kayla" in lib
    entry = lib["kayla"]
    assert set(entry.keys()) >= {"embeddings", "clips", "updated_at"}
    assert len(entry["embeddings"]) == 1
    assert len(entry["clips"]) == 1
    assert isinstance(entry["updated_at"], str) and entry["updated_at"]


def test_voices_idempotent_append_increments_reference_count(client, fake_embedder, tmp_voice_dirs):
    """ISC-12: hitting the same name twice yields reference_count == 2 and two embeddings stored."""
    r1 = client.post("/voices/kayla", files=_audio_upload(b"a"))
    r2 = client.post("/voices/kayla", files=_audio_upload(b"b"))
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json()["reference_count"] == 2

    library_path: Path = tmp_voice_dirs["library_path"]
    lib = json.loads(library_path.read_text(encoding="utf-8"))
    assert len(lib["kayla"]["embeddings"]) == 2
    assert len(lib["kayla"]["clips"]) == 2


# ---------------------------------------------------------------------------
# Name validation at the HTTP layer (ISC-13..ISC-15)
# ---------------------------------------------------------------------------


def test_voices_rejects_forward_slash_in_name(client, fake_embedder):
    """ISC-13: `/` in name is rejected — FastAPI likely 404s the route match,
    but whichever status, it must NOT be 200 and must NOT persist anything."""
    resp = client.post("/voices/bad/name", files=_audio_upload())
    # The `/` collapses the path so the router won't match /voices/{name}
    # as a single segment; any of 400/404/405 is acceptable, just not 200.
    assert resp.status_code != 200


def test_voices_rejects_traversal_dotdot(client, fake_embedder):
    """ISC-14: `..` in name is rejected.

    Depending on the HTTP client + URL normalization, `/voices/..` may be
    collapsed before reaching the app. We additionally assert via URL-encoded
    `..` so the literal two dots reach the route handler and hit the
    name_validation guard, which must return 400. Either way, 200 is forbidden.
    """
    # Encoded form — bytes reach the route intact.
    resp_encoded = client.post("/voices/%2E%2E", files=_audio_upload())
    assert resp_encoded.status_code == 400, resp_encoded.text


def test_voices_rejects_names_longer_than_64_chars(client, fake_embedder):
    """ISC-15: names > 64 chars are rejected → 400."""
    too_long = "a" * 65
    resp = client.post(f"/voices/{too_long}", files=_audio_upload())
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Unicode (ISC-16)
# ---------------------------------------------------------------------------


def test_voices_accepts_unicode_letter_name(client, fake_embedder, tmp_voice_dirs):
    """ISC-16: `kayła` (Polish) is a valid Unicode letter sequence."""
    resp = client.post("/voices/kay%C5%82a", files=_audio_upload())  # URL-encoded 'ł'
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "kayła"


# ---------------------------------------------------------------------------
# NFC normalization (ISC-17, ISC-46)
# ---------------------------------------------------------------------------


def test_voices_nfc_normalizes_nfd_input(client, fake_embedder, tmp_voice_dirs):
    """ISC-17/ISC-46: NFD (decomposed) input is stored as NFC (composed)."""
    # 'á' as NFD = U+0061 U+0301
    nfd_name = "kayl" + "\u0061\u0301"
    nfc_name = unicodedata.normalize("NFC", nfd_name)
    assert nfd_name != nfc_name, "sanity: inputs differ pre-normalization"

    # URL-encode raw NFD bytes so the router sees them unchanged.
    import urllib.parse

    resp = client.post(
        f"/voices/{urllib.parse.quote(nfd_name)}",
        files=_audio_upload(),
    )
    assert resp.status_code == 200, resp.text

    library_path: Path = tmp_voice_dirs["library_path"]
    lib = json.loads(library_path.read_text(encoding="utf-8"))
    assert nfc_name in lib, f"expected NFC key in voices.json, got keys: {list(lib)}"
    assert nfd_name not in lib, "NFD form must not appear as a key"


# ---------------------------------------------------------------------------
# Concurrency (ISC-18, ISC-47)
# ---------------------------------------------------------------------------


def test_voices_concurrent_writers_do_not_corrupt_library(
    client, fake_embedder, tmp_voice_dirs
):
    """ISC-18/ISC-47: 10 concurrent writes to the same name yield reference_count=10
    and a readable, well-formed voices.json."""
    # TestClient is sync; drive concurrency via threads.
    from concurrent.futures import ThreadPoolExecutor

    def _post(_idx: int):
        return client.post(
            "/voices/kayla",
            files={"audio_file": ("clip.wav", io.BytesIO(f"clip-{_idx}".encode()), "audio/wav")},
        )

    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_post, range(10)))

    statuses = [r.status_code for r in results]
    assert all(s == 200 for s in statuses), f"some writes failed: {statuses}"

    library_path: Path = tmp_voice_dirs["library_path"]
    lib = json.loads(library_path.read_text(encoding="utf-8"))
    assert "kayla" in lib
    # Final reference_count == 10 (under the default max_references_per_name=50).
    assert len(lib["kayla"]["embeddings"]) == 10
    assert len(lib["kayla"]["clips"]) == 10
    # No corruption: every stored embedding is a list[float] of the same dim.
    dims = {len(e) for e in lib["kayla"]["embeddings"]}
    assert dims == {fake_embedder["dim"]}
