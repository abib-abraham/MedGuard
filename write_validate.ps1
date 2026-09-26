@'
"""
Batch Validation — Primary Dataset (Real-World Photos)
--------------------------------------------------------
Runs every image in tests/sample_images/primary_validation/ through the
trained Stage 2 authenticity model and prints a summary report.

Since these are genuine medicines you photographed yourself, the expected
label for all of them is "Genuine" — this script reports how many the
model got right, and flags any it got wrong or was unsure about, which is
exactly what you want for your validation results section.

Usage:
    py -3.12 src/validate_primary.py
"""

import os
import sys
import glob

# Reuse the existing Stage 2 inference module
sys.path.insert(0, os.path.dirname(__file__))
from stage2_authenticity import run_authenticity_check

VALIDATION_DIR = "tests/sample_images/primary_validation"
EXPECTED_LABEL = "Genuine"  # change this if some images represent suspect cases


def main():
    image_paths = sorted(
        glob.glob(os.path.join(VALIDATION_DIR, "*.jpg")) +
        glob.glob(os.path.join(VALIDATION_DIR, "*.jpeg")) +
        glob.glob(os.path.join(VALIDATION_DIR, "*.png"))
    )

    if not image_paths:
        print(f"No images found in {VALIDATION_DIR}")
        print("Check the folder path and that your photos were saved there.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Running validation...\n")
    print(f"{'Image':<35} {'Predicted':<12} {'Confidence':<12} {'Correct?'}")
    print("-" * 75)

    correct = 0
    results = []

    for path in image_paths:
        filename = os.path.basename(path)
        try:
            result = run_authenticity_check(path)
            is_correct = (result["label"] == EXPECTED_LABEL)
            if is_correct:
                correct += 1
            results.append((filename, result["label"], result["confidence"], is_correct))
            mark = "YES" if is_correct else "NO"
            print(f"{filename:<35} {result['label']:<12} {result['confidence']*100:>6.1f}%      {mark}")
        except Exception as e:
            print(f"{filename:<35} ERROR: {e}")
            results.append((filename, "ERROR", 0, False))

    total = len(image_paths)
    accuracy = (correct / total) * 100 if total else 0

    print("-" * 75)
    print(f"\nSUMMARY")
    print(f"Total images tested:     {total}")
    print(f"Correctly classified:    {correct}")
    print(f"Validation accuracy:     {accuracy:.1f}%")

    misclassified = [r for r in results if not r[3]]
    if misclassified:
        print(f"\nMisclassified / flagged images ({len(misclassified)}):")
        for filename, label, conf, _ in misclassified:
            print(f"  - {filename}: predicted {label} ({conf*100:.1f}% confidence)")
    else:
        print("\nAll images correctly classified as Genuine.")


if __name__ == "__main__":
    main()

'@ | Out-File -FilePath src\validate_primary.py -Encoding utf8