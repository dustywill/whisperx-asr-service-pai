#!/usr/bin/env bash
# PAI fork entrypoint: enforce HF_TOKEN presence and worker-count safety before
# handing control to the upstream entrypoint.
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "FATAL: HF_TOKEN environment variable is required (pyannote model access). Refusing to start." >&2
  exit 1
fi

WORKERS="${UVICORN_WORKERS:-1}"
STRATEGY="${VOICE_LOCK_STRATEGY:-asyncio}"
if [ "$WORKERS" -gt 1 ] && [ "$STRATEGY" != "flock" ]; then
  echo "FATAL: UVICORN_WORKERS=$WORKERS with VOICE_LOCK_STRATEGY=$STRATEGY: asyncio.Lock is per-process." >&2
  echo "       Set VOICE_LOCK_STRATEGY=flock to opt in to cross-process locking, or keep UVICORN_WORKERS=1." >&2
  exit 1
fi

# Ensure voice data directories exist on first boot (ISC-21 / ISC-22).
mkdir -p "${VOICE_CLIPS_DIR:-/data/voices}"
LIB_DIR="$(dirname "${VOICE_LIBRARY_PATH:-/data/voices.json}")"
mkdir -p "$LIB_DIR"

exec /workspace/entrypoint.sh "$@"
