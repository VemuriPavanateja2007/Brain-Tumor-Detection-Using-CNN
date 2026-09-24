import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.preprocessing.pipeline import load_and_preprocess_image
from app.services.cnn_model import CNNModelService
from app.services.gradcam import compute_gradcam

router = APIRouter()

@router.post("/predict")
async def predict_mri(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        content = await file.read()
        preprocessed = load_and_preprocess_image(content)

        cnn = CNNModelService.get_instance()
        pred_class, confidence, probs, pred_idx = cnn.predict(preprocessed)

        # Grad-CAM computation on layer conv2d_2
        heatmap2d = []
        gradcam_available = False
        if cnn._model is not None:
            try:
                heatmap2d = compute_gradcam(cnn._model, preprocessed, pred_idx, layer_name="conv2d_2")
                gradcam_available = True
            except Exception as e:
                print(f"Grad-CAM error: {e}")

        inference_time_ms = int((time.time() - t0) * 1000)

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "display_confidence": f"{confidence * 100:.1f}%",
            "low_confidence": confidence < 0.65,
            "model_name": "Neuro MRI CNN",
            "inference_time_ms": inference_time_ms,
            "gradcam_available": gradcam_available,
            "probabilities": probs,
            "heatmap2D": heatmap2d
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
