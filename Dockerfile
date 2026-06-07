# ============================================================
# Skin Classification Service — Dockerfile
# ============================================================
# Base: python:3.10-slim (Debian Bookworm, no CUDA)
# Model: EfficientNet-B3 — CPU-only inference
# ============================================================

FROM python:3.10-slim

# Reproducible, non-buffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── System deps (libgomp needed by PyTorch CPU) ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────
# CPU torch wheels from PyTorch index (avoid pulling the 2.5 GB CUDA build)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ── Application code ───────────────────────────────────────
COPY app/ ./app/
COPY .env .

# ── Model weights (bind-mount at runtime, not baked in) ───
# docker run -v $(pwd)/weights:/app/weights ...
# If you want to bake the weights into the image instead, uncomment:
# COPY weights/ ./weights/

# Create the weights directory so the app doesn't error on startup
RUN mkdir -p weights

EXPOSE 8000

# ── Entrypoint ─────────────────────────────────────────────
# Run with uvicorn directly — app/main.py's __main__ block
# is only for local dev (python app/main.py).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
