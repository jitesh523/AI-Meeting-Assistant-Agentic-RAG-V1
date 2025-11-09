from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@postgres:5432/meeting_assistant"
    redis_url: str = "redis://redis:6379"
    openai_api_key: Optional[str] = None
    require_openai: bool = False
    cors_allow_origins: List[str] = []

    model_config = SettingsConfigDict(env_file="../../.env", env_file_encoding="utf-8", case_sensitive=False, secrets_dir="/run/secrets")

    @classmethod
    def model_validate_environment(cls) -> "Settings":
        settings = cls()  # load from env
        # Support comma-separated CORS_ALLOW_ORIGINS
        import os
        cors_env = os.getenv("CORS_ALLOW_ORIGINS")
        if cors_env:
            settings.cors_allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        return settings


settings = Settings.model_validate_environment()


