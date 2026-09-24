# AI-Powered Neuro MRI Analysis Using CNN

> **Short Brand:** Neuro MRI AI  
> **Tagline:** *AI-assisted analysis of brain MRI images using deep learning.*  
> **Foundation Repository:** [VemuriPavanateja2007/Brain-Tumor-Detection-Using-CNN](https://github.com/VemuriPavanateja2007/Brain-Tumor-Detection-Using-CNN)

---

## ⚠️ Medical & Clinical Disclaimer

**Neuro MRI AI is an AI-assisted screening and research prototype. It is not a clinically validated diagnostic system and should not be used as a substitute for evaluation by a qualified healthcare professional, neuroradiologist, or physician. This software does not provide medical diagnoses or treatment recommendations.**

---

## 1. Project Overview

This project converts the CNN-based Brain Tumor Detection model developed by Vemuri Pavanateja into a production-grade, human-designed web application for **AI-assisted neuroimaging screening**.

### User Workflow
```
Landing Page
    ↓
Upload MRI Image or Select Test Scan
    ↓
Preview & Validate Image (Resolution, Aspect Ratio)
    ↓
Exact CNN Preprocessing (128×128 RGB, 1/255 Rescaling)
    ↓
Trained CNN Model Forward Pass
    ↓
Prediction & Softmax Confidence Score
    ↓
Layer-Level Grad-CAM Heatmap Visualization (conv2d_2)
    ↓
AI-Assisted Clinical Research Explanation (Gemini)
    ↓
Comprehensive Result Dashboard & Report
```

---

## 2. CNN Architecture & Specifications

Replicates the exact deep learning architecture from `brain-tumor.ipynb`:

| Layer Name | Type & Configuration | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| `rescaling` | `Rescaling(1./255)` | `(None, 128, 128, 3)` | 0 |
| `conv2d` | `Conv2D(32, (3,3), relu)` | `(None, 126, 126, 32)` | 896 |
| `max_pooling2d` | `MaxPooling2D(2,2)` | `(None, 63, 63, 32)` | 0 |
| `conv2d_1` | `Conv2D(64, (3,3), relu)` | `(None, 61, 61, 64)` | 18,496 |
| `max_pooling2d_1` | `MaxPooling2D(2,2)` | `(None, 30, 30, 64)` | 0 |
| `conv2d_2` | `Conv2D(128, (3,3), relu)` *(Grad-CAM)* | `(None, 28, 28, 128)` | 73,856 |
| `max_pooling2d_2` | `MaxPooling2D(2,2)` | `(None, 14, 14, 128)` | 0 |
| `flatten` | `Flatten` | `(None, 25088)` | 0 |
| `dense` | `Dense(128, relu)` | `(None, 128)` | 3,211,392 |
| `dense_1` | `Dense(4, softmax)` | `(None, 4)` | 516 |

- **Total Trainable Parameters:** 3,305,156 (~12.6 MB)
- **Input Dimensions:** `128 × 128 × 3` (RGB)
- **Normalization:** `1.0 / 255.0`
- **Trained Classes:** `['glioma', 'meningioma', 'notumor', 'pituitary']`
- **Validation Accuracy:** 88.94% (Test cohort of 1,600 balanced scans)

---

## 3. Technology Stack

- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons
- **Backend (Full-Stack Mode):** Node.js / Express with TypeScript running on port 3000
- **Backend (Python Standalone Mode):** FastAPI, Uvicorn, TensorFlow/Keras, Pillow
- **AI Interpretability:** Gradient-weighted Class Activation Mapping (Grad-CAM) on layer `conv2d_2`
- **Explanatory Assistant:** Google Gemini API (`@google/genai` with `gemini-2.5-flash`) for post-inference clinical synthesis

---

## 4. Running the Application

### Option A: Standard Full-Stack (Dev Server)
```bash
npm install
npm run dev
```
The server will start at `http://localhost:3000` with the complete API and UI.

### Option B: Standalone Python FastAPI Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5. API Endpoints

- `GET /api/health` - Check model readiness and architecture specs
- `POST /api/predict` - Upload MRI scan (JPEG, PNG, WebP) for CNN forward pass & Grad-CAM heatmap
- `POST /api/explain` - Synthesize structured clinical explanation using Gemini
