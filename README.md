# FinGPT Sentiment Analysis Service

This project is a production-ready FastAPI service that wraps the FinGPT model for financial sentiment analysis. It uses LLM-based sentiment extraction (LLaMA-2 with LoRA adapters) to provide high-accuracy sentiment labels for financial news.

## Features
- **FastAPI Framework**: High-performance asynchronous API.
- **FinGPT Integration**: Uses LLaMA-2-7b with FinGPT LoRA weights.
- **Quantized Inference**: Supports 8-bit quantization for reduced VRAM usage.
- **Robust Error Handling**: Comprehensive validation and logging.
- **Dockerized**: Ready for containerized deployment with GPU support.

## Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with 12GB+ VRAM (for default configuration)
- NVIDIA Container Toolkit installed (for GPU pass-through)

## Installation & Setup

1. **Clone the repository**:
    git clone <repository-url>
    cd fingpt-sentiment-service

2. **Configuration**:
    The service can be configured via environment variables in `app/core/config.py` or a `.env` file.
    - `MODEL_NAME`: The HuggingFace LoRA model path.
    - `BASE_MODEL`: The HuggingFace base LLM path.

3. **Run with Docker Compose**:
    docker-compose up --build

## API Reference

### 1. Analyze Sentiment
- **Endpoint**: `POST /api/v1/analyze`
- **Request Body**:
    {
      "text": ["Apple reports record-breaking revenue.", "Market uncertainty grows."]
    }
- **Response**:
    {
      "results": [
        {"text": "...", "label": "positive", "score": 1.0},
        {"text": "...", "label": "negative", "score": 1.0}
      ],
      "model_used": "FinGPT/fingpt-sentiment_llama2-7b_lora"
    }

### 2. Health Check
- **Endpoint**: `GET /api/v1/health`
- **Description**: Returns the status of the service and whether the model has finished loading.

## Running Tests
To run unit tests (using mocks to avoid model loading):
    pytest

## Project Structure
- `app/api/`: Route definitions.
- `app/core/`: Configuration and logging setup.
- `app/schemas/`: Pydantic models for validation.
- `app/services/`: Core logic for FinGPT model inference.
- `tests/`: Pytest suite.

## Known Limitations
- **Cold Start**: The model takes ~60-90 seconds to load into VRAM. Use the `/health` endpoint to verify readiness.
- **Hardware**: While it can run on CPU, it is extremely slow. An NVIDIA GPU is highly recommended.
