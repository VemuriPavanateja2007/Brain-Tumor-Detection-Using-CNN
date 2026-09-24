# Neuro MRI AI - Backend API (FastAPI + TensorFlow)

FastAPI backend service running the trained Convolutional Neural Network (CNN) for brain MRI analysis and Grad-CAM interpretability.

## Requirements
- Python 3.10+
- TensorFlow 2.15+
- FastAPI & Uvicorn

## Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Server

```bash
# Start FastAPI backend on http://localhost:8000
python -m app.main
# Or via uvicorn directly:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /api/health` - Check model status and loaded class architecture
- `POST /api/predict` - Upload brain MRI scan (multipart/form-data) to obtain CNN classification and Grad-CAM 28x28 matrix
- `POST /api/explain` - Synthesize AI-assisted clinical research explanation using Gemini API
