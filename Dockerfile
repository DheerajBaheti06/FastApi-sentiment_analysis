# syntax=docker/dockerfile:1.7

FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: build basics + libgomp for numpy, font libs for wordcloud
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app ./app
COPY predictor.py ./predictor.py
COPY FINBERT_FINAL.BIN ./FINBERT_FINAL.BIN
COPY SVM_FINAL.PKL ./SVM_FINAL.PKL
COPY TFIDF_VECTORIZER_FINAL.PKL ./TFIDF_VECTORIZER_FINAL.PKL

# Default host/port via env; can be overridden at runtime
ENV HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Start FastAPI with uvicorn (bind to $PORT if provided by platform)
CMD ["/bin/sh", "-lc", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
