"""Upstream route preservation — ISC-31..ISC-35."""

from __future__ import annotations


def _paths_and_methods(app):
    """Return a set of (path, frozenset(methods)) tuples for all registered routes."""
    out = set()
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None) or set()
        if path is None:
            continue
        out.add((path, frozenset(methods)))
    return out


def test_upstream_routes_all_present(client):
    """ISC-31..ISC-35: /asr, /health, /metrics, /v1/audio/transcriptions, /v1/models, / all registered."""
    from app.main import app

    paths = {p for p, _ in _paths_and_methods(app)}
    for required in (
        "/",
        "/asr",
        "/health",
        "/metrics",
        "/v1/audio/transcriptions",
        "/v1/models",
    ):
        assert required in paths, f"upstream route missing: {required}"


def test_fork_routes_are_present(client):
    """New fork routes /embed and /voices/{name} are also registered."""
    from app.main import app

    paths = {p for p, _ in _paths_and_methods(app)}
    assert "/embed" in paths
    assert "/voices/{name}" in paths


def test_asr_route_supports_post(client):
    """Upstream ASR still accepts POST (we haven't regressed method wiring)."""
    from app.main import app

    for path, methods in _paths_and_methods(app):
        if path == "/asr":
            assert "POST" in methods
            return
    raise AssertionError("/asr route not found")
