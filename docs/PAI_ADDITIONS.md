# PAI Fork Additions

This fork of [murtaza-nasir/whisperx-asr-service](https://github.com/murtaza-nasir/whisperx-asr-service) adds two FastAPI routes to support speaker resolution for the PAI transcription queue (parent PRD `20260421-202617_transcription-queue-server-plus-lightweight-client`, worker stages S6 and S8).

**Deploy target:** TutelarAlien (Linux + CUDA). Windows developers can author and run unit tests locally, but the container itself requires CUDA drivers matching `nvidia/cuda:12.3.2`.

## What this fork adds

| Route | Purpose |
|---|---|
| `POST /embed` | Return a speaker embedding for an uploaded audio clip. |
| `POST /voices/{name}` | Persist a reference clip + its embedding under `name`. |

Everything else — `/asr`, `/health`, `/metrics`, `/v1/audio/transcriptions`, `/v1/models`, `/` — is upstream behavior, unchanged.

## What it deliberately does NOT do

- No GET or DELETE routes under `/voices`. Read surface belongs to the queue app.
- No cosine-similarity collision detection on `POST /voices/{name}`. The queue app's `voice_submissions` table and `/review/voices/*` UI own human review.
- No second ML model load. The existing pyannote diarization pipeline provides the embedding (via `return_embeddings=True`).
- No queue-app bleed-through (no jobs table, no review UI, no SQLite).

## Storage contract

- `/data/voices.json` — canonical library. Schema matches parent PRD DESIGN.md:

  ```json
  {
    "kayla": {
      "embeddings": [[0.12, ...], ...],
      "clips":      ["/data/voices/kayla/clip-2026-04-21T20-47-03.wav", ...],
      "updated_at": "2026-04-21T20:47:03Z"
    }
  }
  ```

- `/data/voices/{name}/clip-{iso8601}.wav` — raw reference clips, retained so the whole library can be re-embedded if pyannote is upgraded.

- Writes to `voices.json` use write-to-temp → `fsync` → `os.replace` → `fsync(parent-dir)`.
- An `asyncio.Lock` serializes writers within the single uvicorn process. Multi-worker deployments must set `VOICE_LOCK_STRATEGY=flock` (the entrypoint refuses to start otherwise).

## Environment variables added

| Var | Default | Purpose |
|---|---|---|
| `VOICE_LIBRARY_PATH` | `/data/voices.json` | Library JSON path. |
| `VOICE_CLIPS_DIR` | `/data/voices` | Root for `{name}/clip-*.wav` files. |
| `EMBEDDING_MAX_CLIP_SECONDS` | `30` | Reject clips longer than this with 413. |
| `MAX_REFERENCES_PER_NAME` | `50` | FIFO cap on stored references per speaker. |
| `UVICORN_WORKERS` | `1` | Multi-worker requires `VOICE_LOCK_STRATEGY=flock`. |
| `VOICE_LOCK_STRATEGY` | `asyncio` | `asyncio` (default) or `flock` (multi-worker opt-in). |
| `PAI_VOICES_SKIP_PROBE` | *(unset)* | Set to `1` in CI to bypass the startup embedding probe. |

Upstream env vars (`HF_TOKEN`, `DEVICE`, `COMPUTE_TYPE`, `DEFAULT_MODEL`, `SERVE_MODE`, `PRELOAD_MODEL`, `MAX_FILE_SIZE_MB`) are preserved.

## Startup probe

On boot, the service embeds 1 second of silence through the diarization pipeline and verifies the resulting vector has `norm > 1e-6`. `/health` returns `503` until this probe passes. Log line to grep for:

```
[pai-voices] startup embedding probe OK dim=<N> norm=<M> model=pyannote/speaker-diarization-community-1
```

## Name validation

Accepted names:

- NFC-normalized before storage and lookup.
- Allow-list: `^[\p{L}\p{N}_-]{1,64}$` (Unicode letters + digits + underscore + hyphen, 1–64 chars).
- Explicitly rejected: `/`, `\`, `..`, null byte, and anything that escapes `VOICE_CLIPS_DIR` under `Path.resolve()`.

## Upstream sync policy

Our changes live on branch `pai-main` (not `pai/main` — slashes in branch names create tooling ambiguity). We rebase, not merge, against `upstream/main`:

```bash
./scripts/rebase-upstream.sh
```

Then push with `--force-with-lease` (never `--force`).

## Build metrics

Captured on each SemVer release via `docker images --format "{{.Size}}"` on the build host.

| Tag | Compressed image size | Built on | Build host |
|---|---|---|---|
| `0.1.0` | _TBD (first build on TutelarAlien pending)_ | _pending_ | tutelaralien |

## Release policy

Every deploy-worthy build gets a SemVer tag first (`dustywill/whisperx-asr-service-pai:0.x.y`). `:latest` is only re-pointed after the integration smoke test passes on that tag. Build metrics (compressed layer size) are captured in the main README under **Build metrics**.

## Deploy to TutelarAlien

1. Build on the build host (or TutelarAlien itself):
   ```bash
   docker build -t dustywill/whisperx-asr-service-pai:0.1.0 .
   ```
2. Push both tags:
   ```bash
   docker push dustywill/whisperx-asr-service-pai:0.1.0
   docker tag dustywill/whisperx-asr-service-pai:0.1.0 dustywill/whisperx-asr-service-pai:latest
   docker push dustywill/whisperx-asr-service-pai:latest
   ```
3. Merge `docker-compose.pai.fragment.yml` into TutelarAlien's stack; ensure the host `/srv/pai/voice-data` (or whatever host path you choose) is the same mount point the queue app uses.
4. `docker compose up -d whisperx-asr`
5. `curl http://tutelaralien:9000/health` — expect 503 for a few seconds, then 200.
