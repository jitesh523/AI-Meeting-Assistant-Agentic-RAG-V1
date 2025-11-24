from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@postgres:5432/meeting_assistant"
    redis_url: str = "redis://redis:6379"
    cors_allow_origins: List[str] = []
    # ASR implementation: 'openai' (default) or 'faster'
    asr_impl: str = "openai"
    # Model size for faster-whisper or openai-whisper (e.g., 'small', 'base')
    asr_model_size: str = "small"
    # Device for faster-whisper: 'cpu' or 'cuda'
    asr_device: str = "cpu"
    # Compute type for faster-whisper: 'int8', 'int8_float16', 'float16', 'float32'
    asr_compute_type: str = "int8"

    model_config = SettingsConfigDict(env_file="../../.env", env_file_encoding="utf-8", case_sensitive=False, secrets_dir="/run/secrets")

    @classmethod
    def model_validate_environment(cls) -> "Settings":
        settings = cls()
        # Support comma-separated CORS_ALLOW_ORIGINS
        import os
        cors_env = os.getenv("CORS_ALLOW_ORIGINS")
        if cors_env:
            settings.cors_allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        return settings


settings = Settings.model_validate_environment()
