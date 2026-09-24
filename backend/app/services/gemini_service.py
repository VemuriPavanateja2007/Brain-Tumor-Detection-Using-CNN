import os
import json
from google import genai

class GeminiExplanationService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Gemini client: {e}")

    def generate_explanation(
        self,
        prediction: str,
        confidence: float,
        probabilities: dict,
        low_confidence: bool,
        file_name: str
    ) -> dict:
        conf_percent = f"{confidence * 100:.1f}%"

        fallback = {
            "summary": f"The custom convolutional neural network classified this neuroimaging slice as {prediction} with {conf_percent} confidence.",
            "patternAnalysis": f"The network analyzed spatial pixel textures and detected structural patterns characteristic of the {prediction} class cohort.",
            "uncertainties": "Single-slice 2D MRI scans cannot capture full 3D volumetric morphology, DWI diffusion parameters, or patient clinical history." if not low_confidence else f"The model confidence ({conf_percent}) is borderline, indicating potential ambiguity in regional feature signals.",
            "recommendedNextSteps": "Comprehensive multi-sequence MRI evaluation by a certified neuroradiologist.",
            "clinicalDisclaimer": "Neuro MRI AI is an AI-assisted screening and research prototype. It does not provide medical diagnoses.",
            "source": "template"
        }

        if not self.client:
            return fallback

        try:
            prompt = f"""
You are a clinical neuroimaging AI research assistant for the "Neuro MRI AI" screening prototype.
The CNN has already run its classification:
- Prediction: {prediction}
- Confidence: {conf_percent}
- Class Probabilities: {json.dumps(probabilities)}
- Low Confidence Flag: {low_confidence}

STRICT SAFETY RULES:
1. Do NOT make a diagnosis.
2. NEVER say "You have a tumor" or "The patient is healthy".
3. Frame everything as: "The CNN model classified this image as...", "The network identified patterns associated with...".
4. Clarify that the CNN operates only on 2D visual patterns from its training dataset.
5. Emphasize research prototype status.

Return ONLY a JSON object:
{{
  "summary": "...",
  "patternAnalysis": "...",
  "uncertainties": "...",
  "recommendedNextSteps": "...",
  "clinicalDisclaimer": "..."
}}
"""
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            data = json.loads(response.text)
            data["source"] = "gemini"
            return data
        except Exception as e:
            print(f"Gemini API generation exception: {e}")
            return fallback
