
"""Stage 2: experimental CNN inference for MedGuard."""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "authenticity_cnn.h5")
IMG_SIZE = (224, 224)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    array = image.img_to_array(img).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)


def run_authenticity_check(image_path):
    model = get_model()
    array = preprocess_image(image_path)
    score = float(model.predict(array, verbose=0).reshape(-1)[0])

    # The model output's class mapping must be checked against training code.
    # This provisional mapping preserves the behavior of the original file.
    label = "Genuine" if score >= 0.5 else "Suspect"
    confidence = score if score >= 0.5 else 1.0 - score

    return {
        "raw_score": score,
        "label": label,
        "confidence": confidence,
        "validated": False,
        "warning": (
            "Experimental prediction only. Training labels and real-world "
            "performance have not been independently validated."
        ),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/stage2_authenticity.py path/to/image.jpg")
        raise SystemExit(1)

    print(run_authenticity_check(sys.argv[1]))
