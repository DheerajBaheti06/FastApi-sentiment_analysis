# syntax=docker/dockerfile:1.7

# ---------- Builder: install wheels to a separate prefix ----------
FROM python:3.10-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    # Prefer official CPU wheels for PyTorch to avoid source builds
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements.txt ./

# Build tools only in builder, then purge
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ---------- Runtime: minimal libs + app code ----------
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    fonts-dejavu-core \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy site-packages and binaries from builder stage
COPY --from=builder /install /usr/local

# Copy source code
COPY app ./app
COPY predictor.py ./predictor.py

# Optional: download models at build time via args (recommended for Railway)
ARG FINBERT_URL=""
ARG SVM_URL=""
ARG TFIDF_URL=""
RUN set -eux; \
    if [ -n "$FINBERT_URL" ]; then curl -L "$FINBERT_URL" -o FINBERT_FINAL.BIN; fi; \
    if [ -n "$SVM_URL" ]; then curl -L "$SVM_URL" -o SVM_FINAL.PKL; fi; \
        if [ -n "$TFIDF_URL" ]; then curl -L "$TFIDF_URL" -o TFIDF_VECTORIZER_FINAL.PKL; fi; \
        # Sanity check: if FINBERT is present but tiny, likely an HTML page was downloaded
        if [ -f FINBERT_FINAL.BIN ]; then \
            sz=$(wc -c < FINBERT_FINAL.BIN); \
            if [ "$sz" -lt 10000000 ]; then echo "ERROR: FINBERT_FINAL.BIN too small ($sz bytes). Check FINBERT_URL is a direct asset link." >&2; exit 1; fi; \
        fi

# Service config (let platform provide PORT)
ENV HOST=0.0.0.0
EXPOSE 8080

# Start FastAPI with uvicorn (bind to $PORT if provided by platform)
CMD ["/bin/sh", "-lc", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --access-log --log-level info"]
