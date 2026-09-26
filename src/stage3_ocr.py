"""
Stage 3 — Batch/Expiry OCR Cross-Verification
------------------------------------------------
Reads printed text off a medicine package image (batch number, expiry date,
manufacturing date) using EasyOCR, then extracts structured fields using
pattern matching.

Usage (standalone test):
    py -3.12 src/stage3_ocr.py path/to/image.jpg

Usage (as a module, from app.py or fusion.py):
    from stage3_ocr import run_ocr_check
    result = run_ocr_check("path/to/image.jpg")
"""

import re
import sys
from datetime import datetime

import easyocr

# ---------------------------------------------------------------------------
# EasyOCR reader is expensive to initialize (loads detection + recognition
# models), so we create it ONCE at module load and reuse it — never create
# a new easyocr.Reader() inside a loop or per-request in the Streamlit app.
# ---------------------------------------------------------------------------
_reader = None


def get_reader():
    """Lazily initialize and cache the EasyOCR reader (English only, CPU)."""
    global _reader
    if _reader is None:
        # gpu=False because most dev/demo machines won't have a CUDA GPU set up.
        # Set gpu=True later if you train/run on a CUDA-enabled machine.
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


# ---------------------------------------------------------------------------
# Regex patterns for common Indian medicine packaging text formats.
# Real packaging is inconsistent, so these are deliberately loose/multiple
# patterns rather than one strict format. Extend this list as you test more
# real packages from your tablet_pack dataset.
# ---------------------------------------------------------------------------
BATCH_PATTERNS = [
    # Requires "Batch"/"B.No"/"Lot" as a whole word BEFORE the captured value,
    # so it can no longer match a substring of the label word itself (the
    # "atch" bug). \b enforces a real word boundary.
    r"\bBatch\s*No\.?\s*[:\-]?\s*([A-Z0-9]{4,12})\b",
    r"\bB\.?\s?No\.?\s*[:\-]?\s*([A-Z0-9]{4,12})\b",
    r"\bBatch\s*[:\-]\s*([A-Z0-9]{4,12})\b",
    r"\bLot\s*(?:No\.?)?\s*[:\-]?\s*([A-Z0-9]{4,12})\b",
]

# Matches dates like: 05/2027, 05-2027, MAY 2027, MAY2027, 05/27,
# and OCR-mangled spaced-out versions like "OC T 23" -> we normalize
# whitespace inside month abbreviations before matching (see _normalize_text).
EXPIRY_PATTERNS = [
    r"Exp(?:iry)?\.?\s*(?:Date)?\s*[:\-]?\s*(\d{1,2}[\/\-]\d{2,4})",
    r"Exp(?:iry)?\.?\s*(?:Date)?\s*[:\-]?\s*([A-Za-z]{3,9}\s?\d{2,4})",
    r"EXP\s*[:\-]?\s*(\d{1,2}[\/\-]\d{2,4})",
    r"EXP\s*[:\-]?\s*([A-Za-z]{3,9}\s?\d{2,4})",
    # fallback: bare "MONTH YY/YYYY" appearing anywhere near the word Exp
    r"Exp[^A-Za-z0-9]{0,15}([A-Za-z]{3,9}\s?\d{2,4})",
]

MFG_PATTERNS = [
    r"Mfg\.?\s*(?:Date)?\s*[:\-]?\s*(\d{1,2}[\/\-]\d{2,4})",
    r"Mfg\.?\s*(?:Date)?\s*[:\-]?\s*([A-Za-z]{3,9}\s?\d{2,4})",
    r"Mfg[^A-Za-z0-9]{0,15}([A-Za-z]{3,9}\s?\d{2,4})",
]


def _normalize_text(text):
    """
    Clean up common OCR artifacts before running regex matching:
    - Collapses letter-by-letter spacing inside month abbreviations,
      e.g. "OC T 23" -> "OCT 23", "SE P 25" -> "SEP 25".
    - Collapses multiple spaces into one.
    This is intentionally simple; extend it as you see more real OCR output.
    """
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    cleaned = text
    for m in months:
        # Rejoin a 3-letter month even if OCR split it with spaces, e.g. "O C T" or "OC T"
        spaced = r"\s*".join(list(m))
        cleaned = re.sub(spaced, m, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Generic "any date-like token" pattern, used as a last-resort fallback when
# a label (Mfg/Exp) and its value get separated by OCR's reading order —
# common when EasyOCR reads text by spatial position rather than logical flow.
GENERIC_DATE_PATTERN = r"\b(\d{1,2}[\/\-]\d{2,4}|[A-Za-z]{3,9}\s?\d{2,4})\b"


def _find_all_dates(text):
    """Return every date-like token found in the text, in order of appearance."""
    matches = re.findall(GENERIC_DATE_PATTERN, text, re.IGNORECASE)
    # Filter out obvious false positives that are too short to be a real year
    return [m for m in matches if any(ch.isdigit() for ch in m)]


def _first_match(patterns, text):
    """Return the first regex match found across a list of patterns, or None."""
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _try_parse_date(date_str):
    """
    Try a handful of common formats to parse an extracted date string.
    Returns a datetime object if successful, else None.
    Not exhaustive — real packaging text is messy, so failures are expected
    and handled gracefully rather than crashing.
    """
    if not date_str:
        return None

    formats_to_try = [
        "%m/%Y", "%m-%Y", "%m/%y", "%m-%y",
        "%b %Y", "%B %Y", "%b%Y", "%B%Y",
        "%b %y", "%B %y", "%b%y", "%B%y",
        "%d/%m/%Y", "%d-%m-%Y",
    ]
    cleaned = date_str.strip().upper().replace("  ", " ")
    for fmt in formats_to_try:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def extract_text(image_path):
    """
    Run EasyOCR on the image and return the raw detected text, joined into
    a single string (easier for regex matching than the raw box-by-box list).
    """
    reader = get_reader()
    results = reader.readtext(image_path, detail=0)  # detail=0 -> just strings
    return " ".join(results)


def run_ocr_check(image_path):
    """
    Main entry point for Stage 3.

    Returns a dict:
        {
            "raw_text": str,          # everything OCR detected, for debugging
            "batch_no": str | None,
            "expiry_raw": str | None,
            "expiry_parsed": datetime | None,
            "expired": bool | None,   # True/False if we could parse a date, else None
            "confidence": float,      # 0.0-1.0, how much we actually found
        }
    """
    raw_text = extract_text(image_path)
    normalized_text = _normalize_text(raw_text)

    batch_no = _first_match(BATCH_PATTERNS, normalized_text)
    expiry_raw = _first_match(EXPIRY_PATTERNS, normalized_text)
    expiry_parsed = _try_parse_date(expiry_raw)
    used_fallback = False

    # Fallback: if we couldn't find a labeled expiry, scan for any date-like
    # tokens in the whole text. Packaging conventionally lists Mfg date
    # before Exp date, so if we find two or more dates, treat the LAST one
    # as the likely expiry (safer assumption than the first, since expiry
    # is usually printed after/below manufacturing date).
    if expiry_parsed is None:
        all_dates = _find_all_dates(normalized_text)
        if all_dates:
            candidate = all_dates[-1]
            parsed_candidate = _try_parse_date(candidate)
            if parsed_candidate is not None:
                expiry_raw = candidate
                expiry_parsed = parsed_candidate
                used_fallback = True

    expired = None
    if expiry_parsed is not None:
        expired = expiry_parsed < datetime.now()

    # Simple confidence heuristic: found batch + found & parsed expiry = 1.0,
    # found one of them = 0.5, found neither = 0.0.
    # This is intentionally simple for the prototype — refine later if needed.
    found_count = sum([batch_no is not None, expiry_parsed is not None])
    confidence = found_count / 2.0

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "batch_no": batch_no,
        "expiry_raw": expiry_raw,
        "expiry_parsed": expiry_parsed,
        "expiry_used_fallback": used_fallback,
        "expired": expired,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Standalone test runner — lets you check this file works on its own,
# without needing the Streamlit app or the other two stages built yet.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.12 src/stage3_ocr.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Running OCR on: {image_path}\n")

    result = run_ocr_check(image_path)

    print("Raw OCR text detected:")
    print(" ", result["raw_text"])
    print("Normalized text (used for matching):")
    print(" ", result["normalized_text"])
    print()
    print("Batch number found:  ", result["batch_no"])
    print("Expiry (raw match):  ", result["expiry_raw"])
    print("Expiry (parsed):     ", result["expiry_parsed"])
    print("Used fallback scan?  ", result["expiry_used_fallback"])
    print("Expired?             ", result["expired"])
    print("Confidence score:    ", result["confidence"])