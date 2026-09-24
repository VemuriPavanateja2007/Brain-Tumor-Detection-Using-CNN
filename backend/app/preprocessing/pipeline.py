import io
import numpy as np
from PIL import Image

TARGET_SIZE = (128, 128)

def load_and_preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Preprocesses uploaded image bytes into the exact tensor format expected by the CNN:
    1. Read and decode image into RGB
    2. Resize to (128, 128)
    3. Normalization: Rescaling by 1./255 is performed inside the Keras model's Rescaling layer,
       or here if direct array inference is run.
    Returns:
        np.ndarray of shape (1, 128, 128, 3) with float32 values [0.0, 255.0]
        (the model's initial Rescaling(1./255) layer maps this to [0.0, 1.0]).
    """
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = image.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    image_array = np.array(image, dtype=np.float32)
    # Expand dims to batch shape (1, 128, 128, 3)
    batch_array = np.expand_dims(image_array, axis=0)
    return batch_array
