
"""
MedGuard Phase 3
Enhanced Medicine Image Preprocessing

Features:
1. Full-image preservation
2. Image resizing
3. CLAHE contrast enhancement
4. Grayscale preprocessing
5. Adaptive threshold preprocessing
6. Otsu threshold preprocessing
7. Multiple image variants for OCR

Compatibility:
prepare_image_for_ocr(image_path) remains available
for the existing application and OCR modules.
"""

import cv2
import numpy as np


# ============================================================
# 1. IMAGE LOADING
# ============================================================

def load_medicine_image(image_path):
    """
    Load a medicine package image safely.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Cannot open image: {image_path}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel color image."
        )

    return image


# ============================================================
# 2. IMAGE RESIZING
# ============================================================

def resize_medicine_image(image, max_dimension=1800):
    """
    Resize large images while preserving aspect ratio.
    """

    height, width = image.shape[:2]

    longest = max(height, width)

    if longest <= max_dimension:
        return image.copy()

    scale = max_dimension / longest

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized


# ============================================================
# 3. CONTRAST ENHANCEMENT
# ============================================================

def enhance_contrast(image):
    """
    Apply CLAHE to the luminance channel.

    This improves local contrast while retaining
    the original color information.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR,
    )


# ============================================================
# 4. UPSCALING
# ============================================================

def upscale_image(image, target_dimension=1600):
    """
    Upscale smaller images to help OCR read fine text.
    """

    height, width = image.shape[:2]

    longest = max(height, width)

    if longest >= target_dimension:
        return image.copy()

    scale = target_dimension / longest

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


# ============================================================
# 5. GRAYSCALE PREPROCESSING
# ============================================================

def create_grayscale_variant(image):
    """
    Convert the image into grayscale and improve contrast.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_gray = clahe.apply(gray)

    return cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2BGR,
    )


# ============================================================
# 6. ADAPTIVE THRESHOLD PREPROCESSING
# ============================================================

def create_adaptive_threshold_variant(image):
    """
    Separate text from the background using adaptive
    thresholding.

    Useful when lighting is uneven.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR,
    )


# ============================================================
# 7. OTSU THRESHOLD PREPROCESSING
# ============================================================

def create_otsu_variant(image):
    """
    Apply Otsu thresholding to produce another
    text recognition candidate.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR,
    )


# ============================================================
# 8. MAIN PREPROCESSING FUNCTION
# ============================================================

def prepare_image_for_ocr(image_path):
    """
    Backward-compatible preprocessing function.

    Returns one contrast-enhanced full-frame image,
    as expected by the existing OCR pipeline.
    """

    image = load_medicine_image(image_path)

    image = resize_medicine_image(
        image,
        max_dimension=1800,
    )

    enhanced = enhance_contrast(image)

    enhanced = upscale_image(
        enhanced,
        target_dimension=1600,
    )

    return enhanced


# ============================================================
# 9. MULTIPLE PREPROCESSING VARIANTS
# ============================================================

def prepare_image_variants(image_path):
    """
    Generate multiple full-frame preprocessing variants.

    Returns:
        Dictionary containing named OpenCV images.

    These variants will be evaluated by the upgraded
    EasyOCR pipeline.
    """

    image = load_medicine_image(image_path)

    image = resize_medicine_image(
        image,
        max_dimension=1800,
    )

    # Original full-frame image
    original = upscale_image(
        image,
        target_dimension=1600,
    )

    # Contrast-enhanced color image
    enhanced = enhance_contrast(image)

    enhanced = upscale_image(
        enhanced,
        target_dimension=1600,
    )

    # Grayscale image
    grayscale = create_grayscale_variant(
        enhanced
    )

    # Adaptive threshold image
    adaptive = create_adaptive_threshold_variant(
        enhanced
    )

    # Otsu threshold image
    otsu = create_otsu_variant(
        enhanced
    )

    variants = {
        "original": original,
        "enhanced": enhanced,
        "grayscale": grayscale,
        "adaptive_threshold": adaptive,
        "otsu_threshold": otsu,
    }

    return variants


# ============================================================
# 10. QUICK PREPROCESSING TEST
# ============================================================

if __name__ == "__main__":

    import os
    import sys

    if len(sys.argv) < 2:

        print(
            "Usage: python src/image_preprocessing.py "
            "path/to/medicine_image.jpg"
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    variants = prepare_image_variants(
        image_path
    )

    output_dir = "tests/preprocessing_output"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("\nMEDGUARD PREPROCESSING TEST")
    print("-" * 50)

    for name, image in variants.items():

        output_path = os.path.join(
            output_dir,
            f"{name}.jpg",
        )

        cv2.imwrite(
            output_path,
            image,
        )

        print(
            f"{name}: {image.shape} -> {output_path}"
        )

    print("-" * 50)
    print("Preprocessing completed successfully.")
