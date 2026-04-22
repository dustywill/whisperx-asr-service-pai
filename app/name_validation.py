"""PAI fork additions: speaker-name validation.

Rules enforced:
  - NFC-normalize before anything else.
  - Reject path separators (/ and \\), null bytes, and `..` traversal.
  - Allow-list regex ^[\\p{L}\\p{N}_-]{1,64}$ (Unicode letters/digits, underscore, hyphen).
  - commonpath containment check against the configured base directory.

Intentional behavior:
  - Unicode homoglyph risk (kayla vs kаyla with Cyrillic а) is accepted at the
    storage layer; the queue app handles human review per parent PRD Dec-18.
    We do NOT reject mixed-script names here to avoid false-negatives on real
    non-ASCII speaker names.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path


class InvalidVoiceName(ValueError):
    pass


try:
    import regex as _regex  # Unicode-property-aware regex (pip install regex)

    _NAME_RE = _regex.compile(r"^[\p{L}\p{N}_-]{1,64}$")

    def _allowed(name: str) -> bool:
        return bool(_NAME_RE.match(name))

except ImportError:  # pragma: no cover
    # Fallback: use unicodedata category checks (stdlib only).
    def _allowed(name: str) -> bool:
        if not (1 <= len(name) <= 64):
            return False
        for ch in name:
            if ch in ("_", "-"):
                continue
            cat = unicodedata.category(ch)
            if not (cat.startswith("L") or cat.startswith("N")):
                return False
        return True


def normalize_and_validate(name: str, base_dir: Path) -> str:
    """Return NFC-normalized, containment-checked name. Raise InvalidVoiceName otherwise."""
    if not isinstance(name, str) or not name:
        raise InvalidVoiceName("empty name")
    if len(name) > 256:
        # Guard against extreme inputs before NFC normalization (which can grow length).
        raise InvalidVoiceName("name too long")

    nfc = unicodedata.normalize("NFC", name)

    if "/" in nfc or "\\" in nfc:
        raise InvalidVoiceName("name contains path separator")
    if ".." in nfc:
        raise InvalidVoiceName("name contains traversal sequence")
    if "\x00" in nfc:
        raise InvalidVoiceName("name contains null byte")

    if not _allowed(nfc):
        raise InvalidVoiceName("name failed allow-list")

    base_resolved = base_dir.resolve()
    joined = (base_dir / nfc).resolve()
    try:
        joined.relative_to(base_resolved)
    except ValueError as exc:
        raise InvalidVoiceName("resolved path escapes base dir") from exc

    return nfc
