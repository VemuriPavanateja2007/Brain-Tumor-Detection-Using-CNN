from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional
from app.services.gemini_service import GeminiExplanationService

router = APIRouter()
gemini_service = GeminiExplanationService()

class ExplainRequest(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    low_confidence: bool
    fileName: Optional[str] = "scan.jpg"

@router.post("/explain")
def explain_mri(req: ExplainRequest):
    return gemini_service.generate_explanation(
        prediction=req.prediction,
        confidence=req.confidence,
        probabilities=req.probabilities,
        low_confidence=req.low_confidence,
        file_name=req.fileName
    )
