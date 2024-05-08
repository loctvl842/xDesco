import os
from typing import Literal

from pydantic_settings import BaseSettings


class CoreSettings(BaseSettings):
    ENV: Literal["development", "production"] = "development"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 5001
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class RedisSettings(BaseSettings):
    REDIS_URL: str = "redis://127.0.0.1:6379/0"


class Settings(CoreSettings, RedisSettings): ...


class DevelopmentSettings(Settings): ...


class ProductionSettings(Settings):
    DEBUG: bool = False


def get_settings() -> Settings:
    source = {"_env_file": ".env", "_env_file_encoding": "utf-8"}
    env = os.getenv("ENV", "development")
    setting_types = {
        "development": DevelopmentSettings(**source),
        "production": ProductionSettings(**source),
    }
    return setting_types[env]


settings = get_settings()
