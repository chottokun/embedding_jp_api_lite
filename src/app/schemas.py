"""Pydantic schemas for OpenAI-compatible embedding API."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator

from app.config import MAX_INPUT_ITEMS, MAX_INPUT_LENGTH

LimitedString = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_INPUT_LENGTH),
]

InputType = Literal[
    "query",
    "document",
    "passage",
    "classification",
    "clustering",
    "sts",
]


class EmbeddingRequest(BaseModel):
    input: (
        LimitedString
        | Annotated[list[LimitedString], Field(min_length=1, max_length=MAX_INPUT_ITEMS)]
    ) = Field(description="Input text or list of texts to embed.")
    model: str | None = Field(
        default=None,
        description="Model identifier (defaults to the currently loaded model).",
    )
    encoding_format: Literal["float", "base64"] = Field(
        default="float",
        description="Format to return embeddings in ('float' or 'base64').",
    )
    dimensions: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional output embedding dimensions (Matryoshka truncation + L2 re-normalization)."
        ),
    )
    input_type: InputType | None = Field(
        default=None,
        description=(
            "Input task type: 'query', 'document', 'passage', "
            "'classification', 'clustering', 'sts'."
        ),
    )
    apply_ruri_prefix: bool = Field(
        default=False,
        description=(
            "Automatically apply query/passage prefix based on input shape "
            "when input_type is omitted."
        ),
    )
    user: str | None = Field(
        default=None,
        description="Optional end-user identifier.",
    )

    @field_validator("input_type", mode="before")
    @classmethod
    def _normalize_input_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float] | str
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    embedding_dim: int


class ErrorResponse(BaseModel):
    detail: str
