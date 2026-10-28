
import re

# Known medicine and ingredient terms used for preliminary screening.
SPECIFIC_TERMS = {
    "omnigel",
    "flexon",
    "vazidol",
    "pentixol",
    "diclofenac",
    "ibuprofen",
    "paracetamol",
    "varenicline",
    "flupentixol",
    "melitracen",
    "cipla",
    "consern",
    "aristo",
}

GENERAL_TERMS = {
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "syrup",
    "gel",
    "ointment",
    "composition",
    "dosage",
    "prescription",
    "batch",
    "expiry",
    "mfg",
    "mfd",
    "manufacture",
    "manufacturer",
    "pharmaceutical",
    "mg",
    "ml",
    "ingredients",
    "warning",
    "storage",
    "varenicline",
    "diclofenac",
    "ibuprofen",
    "paracetamol",
}


def screen_medicine_image(ocr_text):
    """
    Preliminary text-based medicine screening.

    This is not an image classifier and cannot authenticate
    medicines. OCR text may be incomplete or incorrect.
    """
    text = (ocr_text or "").lower()
    text = re.sub(r"[^a-z0-9./ -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    specific_found = sorted(
        term for term in SPECIFIC_TERMS if term in text
    )

    general_found = sorted(
        term for term in GENERAL_TERMS if term in text
    )

    # One recognized medicine/ingredient term is enough to
    # pass the preliminary text screen.
    accepted = bool(specific_found or general_found)

    indicators = sorted(set(specific_found + general_found))

    if accepted:
        reason = (
            "Medicine-related text indicators were detected. "
            "This does not confirm the product's identity or authenticity."
        )
    else:
        reason = (
            "OCR did not identify enough recognizable medicine-related "
            "text. The image may still show a medicine package."
        )

    return {
        "is_medicine_candidate": accepted,
        "matched_indicators": indicators,
        "reason": reason,
    }
