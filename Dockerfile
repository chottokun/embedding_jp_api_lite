FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LITE_MODEL=Chottokun/ruri-v3-30m-lite \
    HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/data/huggingface

WORKDIR /app

# Install runtime dependencies only (Zero-Torch: no torch, no transformers)
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn>=0.32.0" \
    "onnxruntime>=1.19.0" \
    "numpy>=1.26.0" \
    "huggingface-hub>=0.26.0" \
    "https://github.com/chottokun/embedding-sentencepiece-lite/releases/download/v0.1.1/sentencepiece_lite-0.1.1-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

COPY src ./src
ENV PYTHONPATH=/app/src

VOLUME ["/data/huggingface"]
EXPOSE 8080

CMD ["python", "-m", "app.main"]
