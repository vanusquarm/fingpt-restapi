from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "FinGPT Sentiment Analysis Service"
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration
    # Using the Llama2-7b LoRA version as a standard FinGPT example
    # For local CPU testing without high RAM, one might use 'ProsusAI/finbert'
    # but we stick to the requested FinGPT architecture.
    MODEL_NAME: str = "FinGPT/fingpt-sentiment_llama2-7b_lora"
    BASE_MODEL: str = "nousresearch/llama-2-7b-hf"
    
    # Device configuration
    DEVICE: str = "cuda"  # or "cpu"
    LOAD_IN_8BIT: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
