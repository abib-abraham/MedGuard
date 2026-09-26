@'
"""
Stage 2 — Authenticity Classification (Inference)
------------------------------------------------
Loads the CNN trained in Colab (authenticity_cnn.h5) and uses it to classify
a medicine package photo as genuine or suspect.

This file does NOT train anything — training happens in Colab. This is the
lightweight inference side that runs locally in VS Code.

Usage (standalone test):
    py -3.12 src/stage2_authenticity.py path/to/image.jpg

Usage (as a module, from app.py or fusion.py):
    from stage2_authenticity import run_authenticity_check
    result = run_authenticity_check("path/to/image.jpg")
"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/authenticity_cnn.h5"
IMG_SIZE = (224, 224)  # must match the size used during training in Colab

# ---------------------------------------------------------------------------
# The model is expensive to load, so we load it ONCE at module import time
# and reuse it — never reload it per-request inside the Streamlit app.
# ---------------------------------------------------------------------------
_model = None


def get_model():
    """Lazily load and cache the trained authenticity model."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_path):
    """
    Load an image from disk and prepare it exactly the way the model
    expects: resized to 224x224 and pixel values scaled to 0-1
    (matching the rescale=1./255 used during training in Colab).
    """
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # model expects a batch
    return img_array


def run_authenticity_check(image_path):
    """
    Main entry point for Stage 2.

    Returns a dict:
        {
            "raw_score": float,      # model's raw output, 0.0-1.0
            "label": str,            # "Genuine" or "Suspect"
            "confidence": float,     # how confident the model is in its label, 0.0-1.0
        }

    Recall from training: 0 = Fake, 1 = Real (see class_mode/classes in
    the Colab training script), so a raw_score close to 1.0 means genuine,
    close to 0.0 means suspect.
    """
    model = get_model()
    img_array = preprocess_image(image_path)

    raw_score = float(model.predict(img_array, verbose=0)[0][0])

    if raw_score >= 0.5:
        label = "Genuine"
        confidence = raw_score
    else:
        label = "Suspect"
        confidence = 1.0 - raw_score

    return {
        "raw_score": raw_score,
        "label": label,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Standalone test runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.12 src/stage2_authenticity.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Running authenticity check on: {image_path}\n")

    result = run_authenticity_check(image_path)

    print("Raw model score (0=Fake, 1=Real):", round(result["raw_score"], 4))
    print("Label:                           ", result["label"])
    print("Confidence:                      ", f"{result['confidence']*100:.1f}%")

'@ | Out-File -FilePath src\stage2_authenticity.py -Encoding utf8