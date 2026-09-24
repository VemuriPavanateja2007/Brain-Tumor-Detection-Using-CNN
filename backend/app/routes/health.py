from fastapi import APIRouter
from app.services.cnn_model import CNNModelService

router = APIRouter()

@router.get("/health")
def health_check():
    cnn = CNNModelService.get_instance()
    return {
        "status": "healthy",
        "model_loaded": cnn.is_loaded,
        "model_name": "Neuro MRI CNN",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "input_shape": [128, 128, 3],
        "preprocessing": "1.0 / 255.0 normalization"
    }
