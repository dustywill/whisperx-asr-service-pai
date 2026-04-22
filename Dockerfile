# WhisperX ASR API Service Dockerfile
# Based on NVIDIA CUDA for GPU support

FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA support (includes bundled cuDNN 9.8)
RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Set library path to prefer PyTorch's bundled cuDNN over system cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Install WhisperX from sealambda's pyannote-audio-4 compatible branch
# Credit: https://github.com/sealambda/whisperX/tree/feat/pyannote-audio-4
RUN pip3 install --no-cache-dir git+https://github.com/sealambda/whisperX.git@feat/pyannote-audio-4

# Patch WhisperX diarize.py to use 'token=' instead of 'use_token=' for pyannote.audio 4.0
# This handles both single-line and multi-line formatting
RUN sed -i 's/use_token=/token=/g' \
    /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py

# Install latest pyannote.audio for community-1 model support
RUN pip3 install --no-cache-dir --upgrade pyannote.audio

# Install API dependencies
RUN pip3 install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    "ray[serve]>=2.9"

# PAI fork additions: Unicode-property-aware regex for name validation.
RUN pip3 install --no-cache-dir regex==2024.11.6

# Pre-download NLTK data for timestamp alignment (enables offline use)
RUN python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/.cache/nltk_data')"
ENV NLTK_DATA=/.cache/nltk_data

# Create cache directory
RUN mkdir -p /.cache && chmod 777 /.cache

# Copy application code
COPY app /workspace/app

# Copy entrypoint scripts (upstream + PAI wrapper)
COPY entrypoint.sh /workspace/entrypoint.sh
COPY docker-entrypoint-pai.sh /workspace/docker-entrypoint-pai.sh
RUN chmod +x /workspace/entrypoint.sh /workspace/docker-entrypoint-pai.sh

# PAI fork: non-root user matching parent-PRD volume ownership convention.
# /data owns the voice library; /.cache and /workspace must stay writable.
RUN groupadd -g 1000 pai && useradd -u 1000 -g 1000 -m -s /bin/bash pai \
 && mkdir -p /data && chown -R 1000:1000 /data /.cache /workspace
USER 1000:1000

# Expose API port (9000) and Ray dashboard (8265)
EXPOSE 9000 8265

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:9000/health',timeout=2).status==200 else sys.exit(1)" || exit 1

# Default: simple mode (uvicorn). Set SERVE_MODE=ray for Ray Serve.
ENV SERVE_MODE=simple

# PAI fork: entrypoint validates HF_TOKEN + worker count, then hands off to upstream.
CMD ["/workspace/docker-entrypoint-pai.sh"]
