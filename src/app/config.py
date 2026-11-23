"""Configuration loaded from environment variables."""

import os


def get_lite_model() -> str:
    return os.getenv("LITE_MODEL", "Chottokun/ruri-v3-30m-lite")


def get_host() -> str:
    return os.getenv("HOST", "0.0.0.0")


def get_port() -> int:
    return int(os.getenv("PORT", "8080"))


def get_api_key() -> str:
    return os.getenv("API_KEY", "").strip()


def get_max_length() -> int:
    return int(os.getenv("LITE_MAX_LENGTH", "512"))


def get_batch_size() -> int:
    return int(os.getenv("LITE_BATCH_SIZE", "64"))


MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "65536"))
MAX_INPUT_ITEMS = int(os.getenv("MAX_INPUT_ITEMS", "256"))
