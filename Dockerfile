# Note: sentencepiece-lite publishes x86_64 wheels only (build/run for linux/amd64).
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install runtime dependencies only from uv.lock (Zero-Torch: no torch, no transformers)
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    LITE_MODEL="Chottokun/ruri-v3-30m-lite" \
    HOST="0.0.0.0" \
    PORT="8080" \
    HF_HOME="/data/huggingface"

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /sbin/nologin app \
    && mkdir -p /data/huggingface \
    && chown -R app:app /data/huggingface

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app src ./src

USER app

VOLUME ["/data/huggingface"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD ["python", "-c", "import os, urllib.request; port = os.getenv('PORT', '8080'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)"]

CMD ["python", "-m", "app.main"]
