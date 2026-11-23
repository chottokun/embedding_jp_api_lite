"""FastAPI application for embedding_jp_api_lite."""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated

import anyio
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request

from app.auth import verify_api_key
from app.config import get_host, get_lite_model, get_lite_model_revision, get_port
from app.loader import LiteEmbeddingEngine
from app.schemas import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    ModelCard,
    ModelList,
    Usage,
)

logger = logging.getLogger("uvicorn.error")


def get_engine(request: Request) -> LiteEmbeddingEngine:
    engine: LiteEmbeddingEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return engine


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    model_id = get_lite_model()
    revision = get_lite_model_revision()
    logger.info(
        "[embedding_jp_api_lite] Loading model: %s (revision=%s)",
        model_id,
        revision or "unpinned",
    )
    engine = LiteEmbeddingEngine(model_id=model_id, revision=revision)
    app_instance.state.engine = engine
    torch_loaded = "torch" in sys.modules or "transformers" in sys.modules
    logger.info(
        "[embedding_jp_api_lite] Ready: model=%s, dim=%s, torch_or_transformers_loaded=%s",
        engine.model_id,
        engine.embedding_dim,
        torch_loaded,
    )
    yield
    app_instance.state.engine = None


app = FastAPI(
    title="embedding_jp_api_lite",
    version="0.1.0",
    description=(
        "OpenAI-compatible lightweight embedding server "
        "(sentencepiece_lite + ONNX Runtime, Zero-Torch)."
    ),
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health(engine: Annotated[LiteEmbeddingEngine, Depends(get_engine)]):
    """Health check returning loaded model ID and embedding dimension."""
    return HealthResponse(
        status="ok",
        model=engine.model_id,
        embedding_dim=engine.embedding_dim,
    )


@app.get(
    "/v1/models",
    response_model=ModelList,
    dependencies=[Depends(verify_api_key)],
    responses={401: {"model": ErrorResponse}},
    tags=["Models"],
)
async def list_models(engine: Annotated[LiteEmbeddingEngine, Depends(get_engine)]):
    """List the currently loaded model in OpenAI-compatible format."""
    return ModelList(
        data=[
            ModelCard(
                id=engine.model_id,
                created=engine.created_at,
                owned_by="local",
            )
        ]
    )


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingResponse,
    dependencies=[Depends(verify_api_key)],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    tags=["Embeddings"],
)
async def create_embeddings(
    request: EmbeddingRequest,
    engine: Annotated[LiteEmbeddingEngine, Depends(get_engine)],
):
    """Create embeddings for input string or list of strings."""
    if request.dimensions is not None and request.dimensions > engine.embedding_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Requested dimensions ({request.dimensions}) exceed model embedding "
                f"dimension ({engine.embedding_dim})."
            ),
        )

    if isinstance(request.input, str):
        is_single_str = True
        raw_texts = [request.input]
    else:
        is_single_str = False
        raw_texts = list(request.input)

    prefix = engine.resolve_prefix(
        input_type=request.input_type,
        apply_ruri_prefix=request.apply_ruri_prefix,
        is_single_str=is_single_str,
    )
    processed_texts = engine.apply_prefix(raw_texts, prefix)

    formatted_embs, token_count = await anyio.to_thread.run_sync(
        engine.encode_and_format,
        processed_texts,
        request.dimensions,
        request.encoding_format,
    )

    return EmbeddingResponse(
        data=[EmbeddingData(embedding=emb, index=i) for i, emb in enumerate(formatted_embs)],
        model=request.model or engine.model_id,
        usage=Usage(prompt_tokens=token_count, total_tokens=token_count),
    )


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=get_host(),
        port=get_port(),
        workers=1,
    )


if __name__ == "__main__":
    main()
