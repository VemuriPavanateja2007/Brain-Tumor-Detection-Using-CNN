import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.routes.explain import router as explain_router

app = FastAPI(
    title="Neuro MRI AI Backend",
    description="FastAPI service for CNN-based Brain MRI classification and Grad-CAM interpretability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(predict_router, prefix="/api", tags=["Inference"])
app.include_router(explain_router, prefix="/api", tags=["Explanation"])

@app.get("/")
def root():
    return {
        "message": "Neuro MRI AI CNN Inference Service is operational",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
