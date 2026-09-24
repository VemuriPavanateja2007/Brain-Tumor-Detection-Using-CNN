import os
from pydantic_settings import BaseSettings

class Settings:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.65"))
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/applet/model/neuro_mri_cnn.keras")

settings = Settings()
