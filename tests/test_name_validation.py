"""name_validation.normalize_and_validate — unit tests."""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

from app.name_validation import InvalidVoiceName, normalize_and_validate


@pytest.fixture()
def base(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "kayla",
        "byron_123",
        "kayła",   # Polish ł (Latin Extended-A)
        "田中",    # CJK unified ideograph
        "a-b_c",
    ],
)
def test_accepts_valid_names(base, name):
    result = normalize_and_validate(name, base)
    assert result == unicodedata.normalize("NFC", name)


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/",
        "\\",
        "..",
        "\x00",
        "a/b",
        "../a",
        "a" * 65,          # > 64 chars post-NFC
        "\u00E0" * 256,   # 256-char NFD-explosion attempt, tripped by the 256 guard
    ],
)
def test_rejects_invalid_names(base, name):
    with pytest.raises(InvalidVoiceName):
        normalize_and_validate(name, base)


# ---------------------------------------------------------------------------
# NFC round-trip
# ---------------------------------------------------------------------------


def test_nfd_input_returns_nfc_output(base):
    nfd = "kayl" + "\u0061\u0301"  # a + combining acute
    nfc = unicodedata.normalize("NFC", nfd)
    assert nfd != nfc  # sanity

    result = normalize_and_validate(nfd, base)
    assert result == nfc


# ---------------------------------------------------------------------------
# Containment check (POSIX + Windows)
# ---------------------------------------------------------------------------


def test_containment_blocks_absolute_path_names(base):
    """An absolute path as the speaker name must be rejected either via the
    path-separator guard (contains '/' or '\\\\') or the containment check."""

    if sys.platform.startswith("win"):
        evil = "C:\\windows\\system32"
    else:
        evil = "/etc/passwd"

    with pytest.raises(InvalidVoiceName):
        normalize_and_validate(evil, base)
