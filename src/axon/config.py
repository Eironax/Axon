"""Application configuration loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AxonSettings(BaseSettings):
    """Runtime settings for model inference, steering defaults, and storage."""

    model_config = SettingsConfigDict(
        env_prefix="AXON_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    device: str = "cuda"
    default_layer: int = 14

    postgres_dsn: str = "postgresql://axon:axon@localhost:5432/axon"
    redis_url: str = "redis://localhost:6379/0"


@lru_cache(maxsize=1)
def get_settings() -> AxonSettings:
    """Return cached AxonSettings instance."""
    return AxonSettings()
