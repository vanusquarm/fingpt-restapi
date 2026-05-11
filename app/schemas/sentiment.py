from pydantic import BaseModel, Field
from typing import List, Union, Optional

class SentimentRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="A single financial headline or a list of headlines.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": ["Goldman Sachs reports record profits for Q3.", "Markets tumble amid inflation fears."]
            }
        }
    }

class SentimentResult(BaseModel):
    text: str
    label: str = Field(..., description="Sentiment label (positive, negative, neutral)")
    score: Optional[float] = Field(None, description="Confidence score if available")

class SentimentResponse(BaseModel):
    results: List[SentimentResult]
    model_used: str
