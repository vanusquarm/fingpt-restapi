from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.logging_config import LoggingMiddleware, logger
from app.services.sentiment_service import analyzer
import threading

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A production-ready API for financial sentiment analysis using FinGPT",
    version="1.0.0"
)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up service...")
    # Load model in a separate thread to not block the server startup
    # Note: In production, you might want to wait for it, but for FastAPI 
    # health checks to work immediately, background loading is often preferred.
    thread = threading.Thread(target=analyzer.load_model)
    thread.start()

@app.get("/")
async def root():
    return {
        "message": "Welcome to FinGPT Sentiment Analysis API",
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
