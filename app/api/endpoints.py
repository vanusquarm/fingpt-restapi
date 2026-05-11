from fastapi import APIRouter, HTTPException, Depends
from app.schemas.sentiment import SentimentRequest, SentimentResponse, SentimentResult
from app.services.sentiment_service import analyzer
from app.core.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    if analyzer.ready:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "loading", "model_loaded": False}

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    if not analyzer.ready:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to initialize")
    
    try:
        # Normalize input to list
        texts = [request.text] if isinstance(request.text, str) else request.text
        
        if not texts:
            raise HTTPException(status_code=400, detail="Empty text input")

        predictions = analyzer.predict(texts)
        
        results = [
            SentimentResult(text=p["text"], label=p["label"], score=p["score"])
            for p in predictions
        ]
        
        return SentimentResponse(
            results=results,
            model_used=settings.MODEL_NAME
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
