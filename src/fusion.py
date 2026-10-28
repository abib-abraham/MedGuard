
"""Stage 4: explainable fusion of MedGuard screening signals."""


def combine_results(screening, authenticity, ocr):
    if not screening.get("is_medicine_candidate", False):
        return {
            "verdict": "Unable to verify image",
            "explanation": (
                "The preliminary screen did not find enough medicine-package "
                "text. Upload a clear, close-up package photo."
            ),
        }

    batch_found = bool(ocr.get("batch_no"))
    expiry_found = ocr.get("expiry_parsed") is not None

    if batch_found and expiry_found:
        details = "Batch number and expiry date were extracted."
    elif batch_found:
        details = "Batch number was extracted, but expiry was not confirmed."
    elif expiry_found:
        details = "Expiry date was extracted, but batch number was not detected."
    else:
        details = (
            "Batch number and expiry date could not be reliably extracted. "
            "This does not mean they are absent from the package."
        )

    model_note = (
        " The CNN prediction is experimental and is not proof of authenticity."
        if authenticity is not None
        else " The CNN was not run because the image did not pass preliminary screening."
    )

    return {
        "verdict": "Preliminary screening — manual verification needed",
        "explanation": details + model_note,
    }
