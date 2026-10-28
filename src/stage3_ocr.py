
import re
import calendar
import time
from datetime import date, datetime
from functools import lru_cache
from image_preprocessing import prepare_image_for_ocr

import cv2
import easyocr
import torch


# ============================================================
# MEDGUARD OCR ENGINE
# ============================================================

# Reader is cached so it is not recreated on every analysis.
# GPU is used only when the installed PyTorch supports CUDA.
@lru_cache(maxsize=1)
def get_reader():
    use_gpu = torch.cuda.is_available()

    print(
        f"[MedGuard OCR] Initializing EasyOCR "
        f"(GPU={'enabled' if use_gpu else 'disabled'})..."
    )

    start = time.perf_counter()

    reader = easyocr.Reader(
        ["en"],
        gpu=use_gpu,
        verbose=False,
    )

    print(
        f"[MedGuard OCR] Reader initialized in "
        f"{time.perf_counter() - start:.1f}s"
    )

    return reader


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """Read and validate the uploaded image."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Unable to open image: {image_path}"
        )

    if len(image.shape) != 3:
        raise ValueError("The uploaded image is not a valid color image.")

    return image


def resize_image(image, max_dimension=1400):
    """Resize large images without changing their aspect ratio."""
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest <= max_dimension:
        return image

    scale = max_dimension / longest

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# OCR QUALITY ESTIMATION
# ============================================================

MEDICINE_TERMS = (
    "omnigel",
    "cipla",
    "diclofenac",
    "paracetamol",
    "ibuprofen",
    "menthol",
    "salicylate",
    "tablet",
    "capsule",
    "syrup",
    "gel",
    "ointment",
    "batch",
    "expiry",
    "manufacture",
    "mfg",
    "mfd",
    "composition",
    "dosage",
    "pharmaceutical",
)


def ocr_quality(text):
    """
    Heuristic for choosing between OCR attempts.
    This is not an OCR confidence score.
    """
    if not text:
        return 0

    normalized = re.sub(
        r"[^A-Z0-9 ]",
        " ",
        text.upper(),
    )

    words = normalized.split()

    if not words:
        return 0

    useful_words = sum(
        1 for word in words
        if len(word) >= 3 and word.isalpha()
    )

    medicine_matches = sum(
        1 for term in MEDICINE_TERMS
        if term in normalized.lower()
    )

    return (
        useful_words
        + medicine_matches * 8
    )


# ============================================================
# OCR TEXT EXTRACTION
# ============================================================

def extract_text(image_path):
    """
    Run OCR on a resized original image.

    If the result appears weak, make one fallback attempt
    using a clockwise-rotated image. The EasyOCR reader is cached.
    """
    reader = get_reader()
    image = prepare_image_for_ocr(image_path)
    image = resize_image(image, max_dimension=1400)

    def recognize(img):
        results = reader.readtext(
            img,
            detail=0,
            paragraph=False,
            batch_size=1,
            workers=0,
            canvas_size=1400,
            mag_ratio=1.0,
            min_size=12,
        )

        lines = []
        seen = set()

        for result in results:
            line = " ".join(str(result).split()).strip()

            if line and line.casefold() not in seen:
                seen.add(line.casefold())
                lines.append(line)

        return "\n".join(lines)

    start = time.perf_counter()

    # First attempt: original orientation.
    text = recognize(image)

    print(
        f"[MedGuard OCR] First pass: "
        f"{time.perf_counter() - start:.1f}s"
    )

    # Retry only if the original result looks weak.
    if ocr_quality(text) < 12:
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

        fallback_start = time.perf_counter()
        fallback_text = recognize(rotated)

        print(
            f"[MedGuard OCR] Rotation fallback: "
            f"{time.perf_counter() - fallback_start:.1f}s"
        )

        if ocr_quality(fallback_text) > ocr_quality(text):
            text = fallback_text

    return text


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize text for searching and field extraction."""
    text = (text or "").upper()

    replacements = {
        "EXP1RY": "EXPIRY",
        "EXP1R": "EXPIR",
        "MANUFACTUREO": "MANUFACTURED",
        "MANUFACTURE0": "MANUFACTURED",
        "BATCH.": "BATCH",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ============================================================
# DATE PARSING
# ============================================================

DATE_TOKEN = (
    r"(?:"
    r"(?:0?[1-9]|1[0-2])\s*[/.-]\s*(?:\d{2}|\d{4})"
    r"|"
    r"(?:0?[1-9]|[12]\d|3[01])\s*[/.-]\s*"
    r"(?:0?[1-9]|1[0-2])\s*[/.-]\s*(?:\d{2}|\d{4})"
    r"|"
    r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"[A-Z]*\s*[- /.]?\s*(?:\d{2}|\d{4})"
    r")"
)


def parse_date(value):
    """
    Parse common medicine-package dates.

    Month/year dates represent the final day of that month.
    Numeric day/month/year dates are interpreted as DD/MM/YYYY.
    """
    if not value:
        return None

    cleaned = re.sub(r"\s+", "", value.upper())
    cleaned = cleaned.replace(".", "/").replace("-", "/")

    formats = [
        ("%m/%Y", True),
        ("%m/%y", True),
        ("%d/%m/%Y", False),
        ("%d/%m/%y", False),
        ("%b/%Y", True),
        ("%B/%Y", True),
        ("%b/%y", True),
        ("%B/%y", True),
        ("%b%Y", True),
        ("%B%Y", True),
        ("%b%y", True),
        ("%B%y", True),
    ]

    for fmt, month_only in formats:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()

            if month_only:
                last_day = calendar.monthrange(
                    parsed.year,
                    parsed.month,
                )[1]

                return date(
                    parsed.year,
                    parsed.month,
                    last_day,
                )

            return parsed

        except ValueError:
            continue

    return None


def find_date_tokens(text):
    """Find distinct date-like strings in OCR text."""
    matches = re.findall(
        DATE_TOKEN,
        text or "",
        flags=re.IGNORECASE,
    )

    result = []
    seen = set()

    for match in matches:
        key = re.sub(r"\s+", "", match.upper())

        if key not in seen:
            seen.add(key)
            result.append(match.strip())

    return result


# ============================================================
# BATCH NUMBER EXTRACTION
# ============================================================

BATCH_PATTERNS = [
    r"\b(?:BATCH|LOT)\s*(?:NO|NUMBER)?\s*[:#.-]\s*"
    r"([A-Z0-9][A-Z0-9/-]{2,19})\b",

    r"\bB\s*NO\s*[:#.-]\s*"
    r"([A-Z0-9][A-Z0-9/-]{2,19})\b",
]


def extract_batch(text):
    """Extract a batch number only when a label is present."""
    for pattern in BATCH_PATTERNS:
        match = re.search(
            pattern,
            text or "",
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = match.group(1).strip(" .:-/#")
        upper_value = value.upper()

        if upper_value in {
            "MONTH",
            "YEAR",
            "NUMBER",
            "MANUFACTURE",
            "MANUFACTURING",
            "DATE",
            "EXPIRY",
        }:
            continue

        return value

    return None


# ============================================================
# EXPIRY AND MANUFACTURING DATES
# ============================================================

EXPIRY_PATTERNS = [
    r"\b(?:EXPIRY|EXPIRATION|EXP)\s*(?:DATE)?\s*"
    r"[:#.-]?\s*(" + DATE_TOKEN + r")",

    r"\bUSE\s*BEFORE\s*[:#.-]?\s*(" + DATE_TOKEN + r")",
]


MFG_PATTERNS = [
    r"\b(?:MFG|MFD)\s*(?:DATE)?\s*[:#.-]?\s*"
    r"(" + DATE_TOKEN + r")",

    r"\b(?:MANUFACTURING\s*DATE|DATE\s*OF\s*MANUFACTURE)"
    r"\s*[:#.-]?\s*(" + DATE_TOKEN + r")",
]


def extract_labeled_date(text, patterns):
    """Extract a date only when it follows an appropriate label."""
    for pattern in patterns:
        match = re.search(
            pattern,
            text or "",
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        raw = match.group(1).strip()
        parsed = parse_date(raw)

        if parsed:
            return raw, parsed

    return None, None


def extract_expiry_rule(text):
    """
    Find statements such as:
    EXPIRY DATE: 24 MONTHS FROM THE MONTH OF MANUFACTURE
    """
    patterns = [
        r"\bEXPIRY\s*(?:DATE)?\s*[:#.-]?\s*"
        r"(\d{1,3})\s*MONTHS?\s*FROM\s*"
        r"(?:THE\s*)?MONTH\s*OF\s*MANUFACTURE\b",

        r"\bEXPIRY\s*(?:DATE)?\s*[:#.-]?\s*"
        r"(\d{1,3})\s*MONTHS?\s*FROM\s*"
        r"(?:THE\s*)?(?:DATE\s*OF\s*)?MANUFACTURE\b",

        r"\b(\d{1,3})\s*MONTHS?\s*FROM\s*"
        r"(?:THE\s*)?MONTH\s*OF\s*MANUFACTURE\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text or "",
            flags=re.IGNORECASE,
        )

        if match:
            months = int(match.group(1))

            if 1 <= months <= 120:
                return months

    return None


def extract_mfg_date_near_label(text):
    """
    Search near explicit manufacturing-date labels.
    Do not treat 'Manufactured by' as a date label.
    """
    if not text:
        return None, None

    patterns = [
        r"\bMONTH\s*AND\s*YEAR\s*OF\s*MANUFACTURE\b",
        r"\bDATE\s*OF\s*MANUFACTURE\b",
        r"\bMANUFACTURING\s*DATE\b",
        r"\bMFG\b",
        r"\bMFD\b",
    ]

    today = date.today()

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            nearby = text[match.end():match.end() + 45]

            # Avoid interpreting a duration rule as a date.
            if re.search(
                r"\b\d{1,3}\s*MONTHS?\s*FROM\b",
                nearby,
                flags=re.IGNORECASE,
            ):
                continue

            for candidate in find_date_tokens(nearby):
                parsed = parse_date(candidate)

                # Reject future dates and implausibly old dates.
                if parsed and date(1980, 1, 1) <= parsed <= today:
                    return candidate, parsed

    return None, None


# ============================================================
# EXPIRY CALCULATION
# ============================================================

def add_months_to_date(start_date, months):
    """Add months while respecting the target month's last day."""
    month_index = start_date.month - 1 + months
    year = start_date.year + month_index // 12
    month = month_index % 12 + 1

    last_day = calendar.monthrange(year, month)[1]
    day = min(start_date.day, last_day)

    return date(year, month, day)


# ============================================================
# MAIN OCR PIPELINE
# ============================================================

def run_ocr_check(image_path):
    """Extract visible package text and return structured fields."""
    raw_text = extract_text(image_path)
    normalized = normalize_text(raw_text)

    batch_no = extract_batch(normalized)

    expiry_raw, expiry_parsed = extract_labeled_date(
        normalized,
        EXPIRY_PATTERNS,
    )

    mfg_raw, mfg_parsed = extract_labeled_date(
        normalized,
        MFG_PATTERNS,
    )

    if not mfg_parsed:
        mfg_raw, mfg_parsed = extract_mfg_date_near_label(
            normalized
        )

    expiry_rule_months = extract_expiry_rule(normalized)

    expiry_used_calculation = False
    expiry_calculation_note = None

    if (
        not expiry_parsed
        and expiry_rule_months
        and mfg_parsed
    ):
        calculated_expiry = add_months_to_date(
            mfg_parsed,
            expiry_rule_months,
        )

        expiry_parsed = calculated_expiry
        expiry_raw = None
        expiry_used_calculation = True

        expiry_calculation_note = (
            f"Calculated using {expiry_rule_months} months "
            "from the detected manufacturing date. "
            "Verify against the printed package."
        )

    if (
        expiry_rule_months
        and not mfg_parsed
        and not expiry_parsed
    ):
        expiry_calculation_note = (
            f"The label states {expiry_rule_months} months "
            "from manufacture, but the manufacturing date "
            "could not be identified reliably."
        )

    today = date.today()
    expired = None
    expiry_status = "Not detected"

    if expiry_parsed:
        expired = expiry_parsed < today

        if expired:
            expiry_status = "expired"
        else:
            expiry_status = "not_expired"

    fields_found = sum([
        bool(batch_no),
        bool(expiry_parsed),
    ])

    # This is field presence, not OCR accuracy.
    field_indicator = fields_found / 2

    return {
        "raw_text": raw_text,
        "normalized_text": normalized,
        "batch_no": batch_no,
        "expiry_raw": expiry_raw,
        "expiry_parsed": expiry_parsed,
        "mfg_raw": mfg_raw,
        "mfg_parsed": mfg_parsed,
        "expiry_rule_months": expiry_rule_months,
        "expiry_used_calculation": expiry_used_calculation,
        "expiry_calculation_note": expiry_calculation_note,
        "expiry_used_fallback": False,
        "expired": expired,
        "expiry_status": expiry_status,
        "confidence": field_indicator,
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            'Usage: python src/stage3_ocr.py "path/to/image.jpg"'
        )
        raise SystemExit(1)

    result = run_ocr_check(sys.argv[1])

    for key, value in result.items():
        print(f"{key}: {value}")
