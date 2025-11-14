
# FILE 3: config.py
# Save in project root

import os
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./dengue.db")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    api_title: str = "Dengue Prediction API"
    api_version: str = "1.0.0"
    model_path: str = os.getenv("MODEL_PATH", "./models/model.h5")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", 8000))
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
