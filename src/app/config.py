"""Configuration loaded from environment variables."""

import os


def _get_int(
    name: str,
    default: int,
    min_value: int = 1,
    max_value: int | None = None,
) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer >= {min_value}, got {raw!r}") from exc
    if value < min_value or (max_value is not None and value > max_value):
        if max_value is not None:
            raise ValueError(
                f"{name} must be an integer between {min_value} and {max_value}, got {raw!r}"
            )
        raise ValueError(f"{name} must be an integer >= {min_value}, got {raw!r}")
    return value


def get_lite_model() -> str:
    return os.getenv("LITE_MODEL", "Chottokun/ruri-v3-30m-lite")


def get_lite_model_revision() -> str | None:
    rev = os.getenv("LITE_MODEL_REVISION", "").strip()
    return rev or None


def get_host() -> str:
    return os.getenv("HOST", "0.0.0.0")


def get_port() -> int:
    return _get_int("PORT", 8080, min_value=1, max_value=65535)


def get_api_key() -> str:
    return os.getenv("API_KEY", "").strip()


def get_max_length() -> int:
    return _get_int("LITE_MAX_LENGTH", 512, min_value=1)


def get_batch_size() -> int:
    return _get_int("LITE_BATCH_SIZE", 64, min_value=1)


def get_max_input_length() -> int:
    return _get_int("MAX_INPUT_LENGTH", 65536, min_value=1)


def get_max_input_items() -> int:
    return _get_int("MAX_INPUT_ITEMS", 256, min_value=1)


# Read at import time so Pydantic field constraints in schemas.py can bind them.
MAX_INPUT_LENGTH = get_max_input_length()
MAX_INPUT_ITEMS = get_max_input_items()
