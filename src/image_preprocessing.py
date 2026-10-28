
import cv2
import numpy as np


def prepare_image_for_ocr(image_path):
    """
    Prepare a medicine photo for OCR.
    Uses a conservative central crop, contrast enhancement,
    and upscaling. Falls back to the full image if needed.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot open image: {image_path}")

    h, w = image.shape[:2]

    # Limit huge phone images while retaining readable detail.
    longest = max(h, w)
    if longest > 1800:
        scale = 1800 / longest
        image = cv2.resize(
            image, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

    h, w = image.shape[:2]

    # Try a central crop, but avoid aggressive cropping.
    # The full image is retained when the crop is too small.
    x1, y1 = int(w * 0.05), int(h * 0.05)
    x2, y2 = int(w * 0.95), int(h * 0.95)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 100 or crop.shape[1] < 100:
        crop = image

    # Improve contrast for printed text.
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )

    # Upscale small text.
    h, w = enhanced.shape[:2]
    if max(h, w) < 1600:
        scale = min(1.5, 1600 / max(h, w))
        enhanced = cv2.resize(
            enhanced,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    return enhanced
