
"""
MedGuard AI - Stage 3 OCR Engine

Field-focused OCR for medicine blister packs.
- Tests four orientations.
- Locates the purple batch/date imprint.
- Crops the printed imprint instead of the whole package.
- Extracts batch number, MFD and EXP conservatively.
- Preserves existing app-facing output keys.

OCR output is preliminary. Manual verification is required.
"""

import re
import time
import calendar
from datetime import date
from functools import lru_cache

import cv2
import easyocr
import numpy as np
import torch


# ============================================================
# 1. OCR READER
# ============================================================

@lru_cache(maxsize=1)
def get_reader():
    use_gpu = bool(torch.cuda.is_available())

    print(
        f"[MedGuard OCR] Initializing EasyOCR "
        f"(GPU={'enabled' if use_gpu else 'disabled'})..."
    )

    start = time.perf_counter()

    reader = easyocr.Reader(
        ["en"],
        gpu=use_gpu,
        verbose=False
    )

    print(
        f"[MedGuard OCR] Reader initialized in "
        f"{time.perf_counter() - start:.1f}s"
    )

    return reader


# ============================================================
# 2. IMAGE UTILITIES
# ============================================================

def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Cannot open image: {image_path}")

    return image


def resize_image(image, max_dimension=1800):
    height, width = image.shape[:2]

    scale = min(
        1.0,
        max_dimension / float(max(height, width))
    )

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )


def rotate_image(image, orientation):
    if orientation == "clockwise":
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if orientation == "counter_clockwise":
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    if orientation == "180":
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    return image


def create_ocr_variants(image_path):
    """Legacy-compatible preprocessing helper."""

    original = resize_image(load_image(image_path))

    lab = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_l, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR
    )

    gray = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return {
        "original": original,
        "enhanced": enhanced,
        "grayscale": cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        ),
        "adaptive_threshold": cv2.cvtColor(
            adaptive,
            cv2.COLOR_GRAY2BGR
        )
    }


# ============================================================
# 3. OCR RECOGNITION
# ============================================================

def recognize_text(reader, image):
    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        batch_size=1,
        workers=0,
        canvas_size=1800,
        mag_ratio=1.5,
        min_size=8,
        text_threshold=0.55,
        low_text=0.30,
        link_threshold=0.35
    )

    lines = []
    confidences = []
    detections = []

    for item in results:
        if len(item) < 3:
            continue

        bbox, value, confidence = item

        value = " ".join(
            str(value).split()
        ).strip()

        if not value:
            continue

        xs = [float(point[0]) for point in bbox]
        ys = [float(point[1]) for point in bbox]

        confidence = float(confidence)

        lines.append(value)
        confidences.append(confidence)

        detections.append({
            "text": value,
            "confidence": confidence,
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys)
        })

    average_confidence = (
        float(np.mean(confidences))
        if confidences else 0.0
    )

    return (
        "\n".join(lines),
        average_confidence,
        detections
    )


# ============================================================
# 4. OCR SCORING
# ============================================================

def orientation_score(text, confidence=0.0):
    upper = (text or "").upper()

    score = confidence * 10

    terms = {
        "VAZIDOL": 5,
        "VARENICLINE": 3,
        "TABLET": 2,
        "BNO": 18,
        "MFD": 15,
        "MFG": 15,
        "EXP": 18,
        "MRP": 2
    }

    for term, points in terms.items():
        if term in upper:
            score += points

    if re.search(r"20\d{2}", upper):
        score += 4

    return score


# ============================================================
# 5. FULL IMAGE OCR
# ============================================================

def extract_text(image_path):
    reader = get_reader()

    image = resize_image(
        load_image(image_path)
    )

    best = {
        "raw_text": "",
        "ocr_score": -1.0,
        "ocr_confidence": 0.0,
        "ocr_variant": "original",
        "ocr_orientation": None,
        "detections": [],
        "image": image
    }

    orientations = [
        "original",
        "clockwise",
        "counter_clockwise",
        "180"
    ]

    for orientation in orientations:
        candidate = rotate_image(
            image,
            orientation
        )

        start = time.perf_counter()

        text, confidence, detections = recognize_text(
            reader,
            candidate
        )

        score = orientation_score(
            text,
            confidence
        )

        elapsed = time.perf_counter() - start

        print(
            f"[MedGuard OCR] {orientation}: "
            f"score={score:.2f}, "
            f"confidence={confidence:.3f}, "
            f"time={elapsed:.1f}s"
        )

        if score > best["ocr_score"]:
            best.update({
                "raw_text": text,
                "ocr_score": score,
                "ocr_confidence": confidence,
                "ocr_orientation": orientation,
                "detections": detections,
                "image": candidate
            })

    print(
        "\n[MedGuard OCR] Selected orientation:",
        best["ocr_orientation"]
    )

    return best


# ============================================================
# 6. PURPLE IMPRINT DETECTION
# ============================================================

def locate_purple_imprint(image):
    """
    Locate the purple batch/date imprint.

    Uses HSV colour segmentation and groups purple components
    that share approximately the same horizontal baseline.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_purple = np.array([125, 80, 40])
    upper_purple = np.array([165, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_purple,
        upper_purple
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8)
    )

    joined = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (13, 3)
        ),
        iterations=1
    )

    contours, _ = cv2.findContours(
        joined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fragments = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        ratio = w / max(h, 1)

        if (
            w >= 12
            and h >= 8
            and w * h >= 150
            and ratio >= 1.15
            and h <= 100
        ):
            fragments.append((
                x,
                y,
                x + w,
                y + h,
                w * h
            ))

    if len(fragments) < 2:
        return None, None

    fragments.sort(
        key=lambda item: (item[1] + item[3]) / 2
    )

    groups = []

    for fragment in fragments:
        center_y = (
            fragment[1] + fragment[3]
        ) / 2

        best_group = None
        best_distance = float("inf")

        for group in groups:
            group_y = np.mean([
                (item[1] + item[3]) / 2
                for item in group
            ])

            distance = abs(
                center_y - group_y
            )

            if distance < 42 and distance < best_distance:
                best_group = group
                best_distance = distance

        if best_group is None:
            groups.append([fragment])
        else:
            best_group.append(fragment)

    ranked = []

    for group in groups:
        x1 = min(item[0] for item in group)
        y1 = min(item[1] for item in group)
        x2 = max(item[2] for item in group)
        y2 = max(item[3] for item in group)

        span = x2 - x1
        ink_area = sum(item[4] for item in group)

        if (
            len(group) >= 2
            and span >= image.shape[1] * 0.22
        ):
            score = (
                span * min(len(group), 12)
                + ink_area * 0.03
            )

            ranked.append((
                score,
                x1,
                y1,
                x2,
                y2
            ))

    if not ranked:
        return None, None

    _, x1, y1, x2, y2 = max(ranked)

    height, width = image.shape[:2]

    pad_x = max(
        80,
        int((x2 - x1) * 0.06)
    )

    pad_y = 24

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)

    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


# ============================================================
# 7. IMPRINT PREPROCESSING
# ============================================================

def crop_variants(crop):
    enlarged = cv2.resize(
        crop,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    return {
        "imprint_color": enlarged,
        "imprint_gray": cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2RGB
        ),
        "imprint_adaptive": cv2.cvtColor(
            adaptive,
            cv2.COLOR_GRAY2RGB
        )
    }


# ============================================================
# 8. FIELD PARSING
# ============================================================

MONTHS = {
    month.upper(): index
    for index, month in enumerate(calendar.month_abbr)
    if month
}

MONTHS.update({
    "SEPT": 9,
    "OCH": 10,
    "0CT": 10
})


def normalize_text(text):
    text = (text or "").upper()

    text = text.replace("SEPT.", "SEP")
    text = text.replace("SEPT ", "SEP ")

    text = re.sub(
        r"[\t ]+",
        " ",
        text
    )

    return text.strip()


def parse_month_year(value):
    """Parse month-year values in OCR and consensus formats."""

    if not value:
        return None

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper()
    )

    # Consensus format: YYYY-MM
    match = re.fullmatch(
        r"(20\d{2})(0[1-9]|1[0-2])",
        compact
    )

    if match:
        year = int(match.group(1))
        month = int(match.group(2))

        return date(
            year,
            month,
            calendar.monthrange(year, month)[1]
        )

    # Original OCR format: OCT2024
    match = re.fullmatch(
        r"([A-Z]{3,4})(\d{4})",
        compact
    )

    if not match:
        return None

    month_token, year_token = match.groups()

    month = MONTHS.get(month_token)
    year = int(year_token)

    if not month or not (1980 <= year <= 2100):
        return None

    return date(
        year,
        month,
        calendar.monthrange(year, month)[1]
    )


def _normalize_ocr_month(token):
    replacements = str.maketrans({
        "0": "O", "Q": "O", "D": "O",
        "G": "C",
        "H": "T", "7": "T", "1": "T", "L": "T", "I": "T",
        "8": "E", "3": "E",
        "2": "P", "R": "P",
        "5": "S"
    })

    token = str(token).upper().translate(replacements)
    return token if token in MONTHS else None


def _normalize_ocr_year(token):
    replacements = str.maketrans({
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1", "|": "1",
        "Z": "2", "H": "2",
        "S": "5", "G": "6", "B": "8"
    })

    token = str(token).upper().translate(replacements)

    if re.fullmatch(r"(?:19|20|21)\d{2}", token):
        return token

    return None


def extract_labeled_month_date(text, label_pattern):
    """Extract a labelled month/year with conservative OCR correction."""

    source = (text or "").upper()

    match = re.search(
        label_pattern,
        source,
        flags=re.IGNORECASE
    )

    if not match:
        return None, None

    tail = source[match.end():]

    stop = re.search(
        r"\b(?:MFD|MFG|MANUFACTURED|EXP|EXPIRY|EXPIRATION|MRP)",
        tail,
        flags=re.IGNORECASE
    )

    if stop:
        tail = tail[:stop.start()]

    compact = re.sub(r"[^A-Z0-9]", "", tail)

    for index in range(max(0, len(compact) - 6)):

        month_token = _normalize_ocr_month(
            compact[index:index + 3]
        )

        year_token = _normalize_ocr_year(
            compact[index + 3:index + 7]
        )

        if month_token and year_token:

            month_number = MONTHS[month_token]

            parsed = date(
                int(year_token),
                month_number,
                calendar.monthrange(
                    int(year_token),
                    month_number
                )[1]
            )

            return month_token + year_token, parsed

    return None, None


def extract_expiry_date(text):
    return extract_labeled_month_date(
        text,
        r"\b(?:EXP|EXPIRY|EXPIRATION)"
    )


def extract_mfg_date(text):
    return extract_labeled_month_date(
        text,
        r"\b(?:MFD|MFG|MANUFACTURED)"
    )


def extract_batch(text):
    """
    Extract a labelled batch number while handling common
    OCR character substitutions.
    """

    if not text:
        return None

    source = str(text).upper()

    patterns = [
        r"\bB\s*\.?\s*NO\s*\.?\s*[:#.,-]?\s*"
        r"([A-Z0-9][A-Z0-9/-]{5,19})",

        r"\b(?:BATCH|LOT)\s*"
        r"(?:NO\.?|NUMBER)?\s*[:#.,-]?\s*"
        r"([A-Z0-9][A-Z0-9/-]{5,19})"
    ]

    for pattern in patterns:
        match = re.search(pattern, source)

        if not match:
            continue

        candidate = match.group(1).strip(
            " .:-/#,"
        )

        candidate = re.sub(
            r"[^A-Z0-9/-]",
            "",
            candidate
        )

        if len(candidate) >= 6:
            return candidate

    return None


def extract_expiry_rule(text):
    patterns = [
        (
            r"\b(\d{1,3})\s*MONTHS?\s*FROM\s*"
            r"(?:THE\s*)?MONTH\s*OF\s*MANUFACTURE\b"
        ),
        (
            r"\b(\d{1,3})\s*MONTHS?\s*FROM\s*"
            r"(?:THE\s*)?(?:DATE\s*OF\s*)?MANUFACTURE\b"
        )
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text or "",
            flags=re.IGNORECASE
        )

        if match:
            months = int(match.group(1))

            if 1 <= months <= 120:
                return months

    return None


def add_months_to_date(start_date, months):
    month_index = start_date.month - 1 + months

    year = (
        start_date.year
        + month_index // 12
    )

    month = month_index % 12 + 1

    last_day = calendar.monthrange(
        year,
        month
    )[1]

    day = min(
        start_date.day,
        last_day
    )

    return date(
        year,
        month,
        day
    )


# ============================================================
# 9. TARGETED OCR
# ============================================================


# ============================================================
# 9. TARGETED FIELD-BY-FIELD OCR
# ============================================================

def recognize_field(reader, image, field_type):
    """
    Recognize one isolated medicine-package field.

    Batch:
        Letters, numbers, slash and hyphen.

    MFD / EXP:
        Letters, numbers and date separators.
    """

    if field_type == "batch":
        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-"
    else:
        allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./-"

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        batch_size=1,
        workers=0,
        canvas_size=1200,
        mag_ratio=2.0,
        min_size=5,
        allowlist=allowlist
    )

    candidates = []

    for result in results:
        if len(result) < 3:
            continue

        text = str(result[1]).strip()
        confidence = float(result[2])

        if text:
            candidates.append({
                "text": text,
                "confidence": confidence
            })

    candidates.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    return candidates


# ------------------------------------------------------------
# 9.1 FIELD IMAGE PREPROCESSING
# ------------------------------------------------------------

def prepare_field_crop(crop):
    """
    Enlarge a small field and create OCR variants.
    """

    enlarged = cv2.resize(
        crop,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return {
        "color": cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2RGB
        ),

        "enhanced": cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2RGB
        )
    }


# ------------------------------------------------------------
# 9.2 SPLIT PRINTED IMPRINT INTO FIELDS
# ------------------------------------------------------------

def split_imprint_fields(crop):
    """
    Split the first purple imprint row into Batch, MFD and EXP.
    Coordinates are based on the actual horizontal imprint layout.
    """

    height, width = crop.shape[:2]

    # Retain the first printed row and exclude most of the MRP row.
    top_row = crop[:int(height * 0.62), :]

    row_height, row_width = top_row.shape[:2]

    coordinates = {
        "batch": (
            int(row_width * 0.04),
            0,
            int(row_width * 0.35),
            row_height
        ),

        "mfg": (
            int(row_width * 0.34),
            0,
            int(row_width * 0.63),
            row_height
        ),

        "expiry": (
            int(row_width * 0.61),
            0,
            int(row_width * 0.97),
            row_height
        )
    }

    fields = {}

    for name, (x1, y1, x2, y2) in coordinates.items():
        field = top_row[y1:y2, x1:x2]

        if field.size > 0:
            fields[name] = field

    return fields


# ------------------------------------------------------------
# 9.3 FIELD CANDIDATE SELECTION
# ------------------------------------------------------------

def get_best_field_candidate(reader, field_crop, field_name):
    """
    Run OCR on two preprocessing variants.

    Keep the strongest candidate, but do not treat its
    confidence as proof that the field is correct.
    """

    variants = prepare_field_crop(field_crop)

    best_text = ""
    best_confidence = 0.0
    best_variant = None

    for variant_name, variant_image in variants.items():

        try:
            candidates = recognize_field(
                reader,
                variant_image,
                field_name
            )

            for candidate in candidates:

                text = candidate["text"]
                confidence = candidate["confidence"]

                if confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence
                    best_variant = variant_name

        except Exception as error:
            print(
                f"[MedGuard OCR] {field_name} "
                f"{variant_name} failed: {error}"
            )

    return {
        "text": best_text,
        "confidence": best_confidence,
        "variant": best_variant
    }





# ============================================================
# 9. TARGETED OCR - CONSERVATIVE FIELD CONSENSUS
# ============================================================

def normalize_ocr_candidate(text):
    """Normalize a candidate for exact comparison."""
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text or "").upper()
    )


def find_consensus(votes):
    """
    Accept a candidate only when at least two
    independent images agree.

    Multiple preprocessing variants from the same
    image count as one image vote.
    """

    grouped = {}

    for vote in votes:
        value = vote.get("value")
        source = vote.get("source", "")

        if not value or not source:
            continue

        normalized = normalize_ocr_candidate(value)

        if len(normalized) < 4:
            continue

        # Extract image identity from source.
        # Example: image_1_imprint_color -> image_1
        image_id = source.split("_imprint_")[0]

        grouped.setdefault(normalized, {}).setdefault(
            image_id, []
        ).append(vote)

    accepted = [
        (key, image_votes)
        for key, image_votes in grouped.items()
        if len(image_votes) >= 2
    ]

    if len(accepted) != 1:
        return None

    return next(iter(accepted[0][1].values()))[0]["value"]

def parse_isolated_date(text):
    """
    Parse a month and four-digit year from an isolated OCR field.
    """

    if not text:
        return None

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )

    month_map = {
        "0": "O",
        "Q": "O",
        "D": "O",
        "6": "G",
        "H": "T",
        "7": "T",
        "1": "I",
        "L": "I",
        "5": "S",
        "3": "E",
        "2": "Z"
    }

    year_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8"
    }

    for index in range(max(0, len(compact) - 6)):

        month_token = compact[index:index + 3]
        year_token = compact[index + 3:index + 7]

        corrected_month = month_token.translate(
            str.maketrans(month_map)
        )

        corrected_year = year_token.translate(
            str.maketrans(year_map)
        )

        month_number = MONTHS.get(corrected_month)

        if (
            month_number
            and re.fullmatch(r"20\d{2}", corrected_year)
        ):
            year = int(corrected_year)

            return date(
                year,
                month_number,
                calendar.monthrange(year, month_number)[1]
            )

    return None

def run_targeted_ocr(image, detections=None):
    """
    OCR the purple imprint and collect independent candidates.

    Returns raw OCR text, diagnostics and field consensus.
    """
    reader = get_reader()

    crop, box = locate_purple_imprint(image)

    if crop is None:
        print("[MedGuard OCR] Purple imprint not found.")

        return {
            "text": "",
            "diagnostics": [],
            "consensus": {
                "batch": None,
                "mfg": None,
                "expiry": None
            }
        }

    print(f"[MedGuard OCR] Imprint crop: {box}")

    variants = crop_variants(crop)

    all_texts = []
    diagnostics = []

    votes = {
        "batch": [],
        "mfg": [],
        "expiry": []
    }

    for variant_name, variant_image in variants.items():
        try:
            text, confidence, _ = recognize_text(
                reader,
                variant_image
            )

            print(
                f"[MedGuard OCR] {variant_name}: "
                f"confidence={confidence:.3f}"
            )
            print("OCR:", text)

            all_texts.append(text)

            batch = extract_batch(text)

            mfg_raw, mfg_date = extract_mfg_date(text)

            expiry_raw, expiry_date = extract_expiry_date(text)

            candidates = {
                "batch": batch,
                "mfg": (
                    mfg_date.strftime("%Y-%m")
                    if mfg_date else None
                ),
                "expiry": (
                    expiry_date.strftime("%Y-%m")
                    if expiry_date else None
                )
            }

            for field, value in candidates.items():
                if value:
                    votes[field].append({
                        "source": variant_name,
                        "value": value
                    })

            diagnostics.append({
                "label": "complete_purple_imprint",
                "text": text,
                "confidence": confidence,
                "variant": variant_name,
                "box": box,
                "candidates": candidates
            })

        except Exception as error:
            print(
                f"[MedGuard OCR] {variant_name} failed: {error}"
            )

    consensus = {
        field: find_consensus(field_votes)
        for field, field_votes in votes.items()
    }

    print("\n[MedGuard OCR] FIELD CONSENSUS")

    for field in ("batch", "mfg", "expiry"):
        print(f"{field}: {votes[field]}")
        print(f"{field} accepted: {consensus[field]}")

    return {
        "text": "\n".join(
            text for text in all_texts if text
        ),
        "diagnostics": diagnostics,
        "consensus": consensus
    }

    # --------------------------------------------------------
    # ISOLATED FIELD RECOGNITION
    # --------------------------------------------------------

    fields = split_imprint_fields(crop)

    for field_name, field_crop in fields.items():

        result = get_best_field_candidate(
            reader,
            field_crop,
            field_name
        )

        field_text = result["text"]
        field_confidence = result["confidence"]
        field_variant = result["variant"]

        field_value = None

        if field_text:

            if field_name == "batch":

                compact = re.sub(
                    r"[^A-Z0-9/-]",
                    "",
                    field_text.upper()
                )

                compact = re.sub(
                    r"^(?:BNO|BATCH|LOT|NO)",
                    "",
                    compact
                )

                match = re.search(
                    r"[A-Z0-9][A-Z0-9/-]{5,19}",
                    compact
                )

                if match:
                    field_value = match.group(0)

            elif field_name in ("mfg", "expiry"):

                parsed_date = parse_isolated_date(field_text)

                if parsed_date:
                    field_value = parsed_date.strftime("%Y-%m")

        print(
            f"[MedGuard OCR] Isolated {field_name}: "
            f"{field_text!r} -> {field_value} "
            f"(confidence={field_confidence:.3f})"
        )

        if field_value:

            votes[field_name].append({
                "source": f"isolated_{field_name}",
                "value": field_value
            })

        diagnostics.append({
            "label": f"isolated_{field_name}_field",
            "text": field_text,
            "confidence": field_confidence,
            "variant": field_variant,
            "box": box,
            "candidates": {
                "batch": field_value if field_name == "batch" else None,
                "mfg": field_value if field_name == "mfg" else None,
                "expiry": field_value if field_name == "expiry" else None
            }
        })

    consensus = {
        field: find_consensus(field_votes)
        for field, field_votes in votes.items()
    }

    print("\n[MedGuard OCR] FIELD CONSENSUS")

    for field in ("batch", "mfg", "expiry"):
        print(f"{field}: {votes[field]}")
        print(f"{field} accepted: {consensus[field]}")

    return {
        "text": "\n".join(
            text for text in all_texts if text
        ),
        "diagnostics": diagnostics,
        "consensus": consensus
    }

    # --------------------------------------------------------
    # FIELD-FOCUSED OCR
    # The top imprint row contains Batch, MFD and EXP. OCR each
    # field separately to keep the lower MRP row out of the read.
    # --------------------------------------------------------
    field_crops = split_imprint_fields(crop)
    month_year_pattern = re.compile(
        r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC|OCH|0CT)"
        r"\s*[-/. ]?\s*(20\d{2})",
        flags=re.IGNORECASE
    )

    for field_name in ("batch", "mfg", "expiry"):
        field_crop = field_crops.get(field_name)
        if field_crop is None or field_crop.size == 0:
            continue

        try:
            field_result = get_best_field_candidate(
                reader, field_crop, field_name
            )
            field_text = field_result["text"].strip()
            field_value = None

            if field_text:
                if field_name == "batch":
                    compact = re.sub(r"[^A-Z0-9/-]", "", field_text.upper())
                    compact = re.sub(r"^(?:BNO|BATCH|LOT|NO)", "", compact)
                    match = re.search(r"[A-Z0-9][A-Z0-9/-]{5,19}", compact)
                    if match:
                        field_value = match.group(0).upper()
                else:
                    date_match = month_year_pattern.search(field_text.upper())
                    if date_match:
                        month_token, year_token = date_match.groups()
                        month_token = month_token.upper()
                        if month_token in ("OCH", "0CT"):
                            month_token = "OCT"
                        parsed_date = parse_month_year(month_token + year_token)
                        if parsed_date:
                            field_value = parsed_date.strftime("%Y-%m")

            field_candidates = {
                "batch": field_value if field_name == "batch" else None,
                "mfg": field_value if field_name == "mfg" else None,
                "expiry": field_value if field_name == "expiry" else None
            }

            diagnostics.append({
                "label": f"isolated_{field_name}_field",
                "text": field_text,
                "confidence": field_result["confidence"],
                "variant": f"isolated_{field_name}_{field_result['variant']}",
                "box": box,
                "candidates": field_candidates
            })

            if field_value:
                votes[field_name].append({
                    "source": f"isolated_{field_name}",
                    "value": field_value
                })

            print(
                f"[MedGuard OCR] Isolated {field_name}: "
                f"{field_text!r} -> {field_value!r} "
                f"(confidence={field_result['confidence']:.3f})"
            )
        except Exception as error:
            print(f"[MedGuard OCR] Isolated {field_name} OCR failed: {error}")

    consensus = {
        field: find_consensus(field_votes)
        for field, field_votes in votes.items()
    }

    print("\n[MedGuard OCR] FIELD CONSENSUS")

    for field in ("batch", "mfg", "expiry"):
        print(f"{field}: {votes[field]}")
        print(f"{field} accepted: {consensus[field]}")

    return {
        "text": "\n".join(
            text for text in all_texts if text
        ),
        "diagnostics": diagnostics,
        "consensus": consensus
    }


# ============================================================
# 10. MAIN OCR PIPELINE - CONSENSUS-BASED RESULTS
# ============================================================

def run_ocr_check(image_path):
    """
    Process one image or multiple images.

    A field is accepted only if at least two distinct
    preprocessing variants agree on its exact value.

    For multiple images, votes are combined, but the source
    names remain image-specific.
    """

    if isinstance(image_path, (list, tuple)):
        image_paths = list(image_path)
    else:
        image_paths = [image_path]

    if not image_paths:
        raise ValueError("At least one image is required.")

    all_raw_text = []
    all_targeted_text = []
    all_diagnostics = []

    votes = {
        "batch": [],
        "mfg": [],
        "expiry": []
    }

    best_full_result = None

    for image_index, current_path in enumerate(image_paths):
        print("\n" + "=" * 60)
        print(f"[MedGuard OCR] IMAGE {image_index + 1}")
        print(current_path)
        print("=" * 60)

        full_result = extract_text(current_path)

        if (
            best_full_result is None
            or full_result["ocr_score"] > best_full_result["ocr_score"]
        ):
            best_full_result = full_result

        raw_text = full_result["raw_text"] or ""
        all_raw_text.append(raw_text)

        targeted = run_targeted_ocr(
            full_result["image"],
            full_result["detections"]
        )

        all_targeted_text.append(targeted["text"])

        for item in targeted["diagnostics"]:
            diagnostic = dict(item)

            diagnostic["image_index"] = image_index + 1

            all_diagnostics.append(diagnostic)

        # Keep each image/variant as an independent vote.
        for item in targeted["diagnostics"]:
            candidates = item.get("candidates", {})

            for field in ("batch", "mfg", "expiry"):
                value = candidates.get(field)

                if value:
                    votes[field].append({
                        "source": (
                            f"image_{image_index + 1}_"
                            f"{item['variant']}"
                        ),
                        "value": value
                    })

    # --------------------------------------------------------
    # FINAL CONSENSUS
    # --------------------------------------------------------

    accepted = {
        field: find_consensus(field_votes)
        for field, field_votes in votes.items()
    }

    print("\n" + "=" * 60)
    print("[MedGuard OCR] FINAL CONSENSUS")
    print("=" * 60)

    for field in ("batch", "mfg", "expiry"):
        print(f"{field} votes: {votes[field]}")
        print(f"{field} accepted: {accepted[field]}")

    # --------------------------------------------------------
    # ACCEPT ONLY CONSENSUS VALUES
    # --------------------------------------------------------

    batch_no = accepted["batch"]

    mfg_raw = accepted["mfg"]
    expiry_raw = accepted["expiry"]

    # The consensus values are normalized YYYY-MM strings.
    # Convert them into dates using the existing date parser.
    mfg_parsed = None
    expiry_parsed = None

    if mfg_raw:
        mfg_parsed = parse_month_year(
            mfg_raw.replace("-", "")
        )

    if expiry_raw:
        expiry_parsed = parse_month_year(
            expiry_raw.replace("-", "")
        )

    # Do not calculate expiry from uncertain OCR text.
    # A printed expiry date must be independently verified.
    expiry_rule_months = None
    expiry_used_calculation = False
    expiry_calculation_note = None

    expired = (
        expiry_parsed < date.today()
        if expiry_parsed else None
    )

    expiry_status = (
        "expired" if expired else "not_expired"
    ) if expiry_parsed else "Not detected"

    fields_found = (
        int(bool(batch_no))
        + int(bool(expiry_parsed))
    )

    combined_raw_text = "\n".join(
        part for part in all_raw_text if part
    )

    targeted_text = "\n".join(
        part for part in all_targeted_text if part
    )

    normalized = normalize_text(
        combined_raw_text + "\n" + targeted_text
    )

    # Keep the candidate lists for debugging.
    expiry_candidates = [
        vote["value"]
        for vote in votes["expiry"]
    ]

    mfg_candidates = [
        vote["value"]
        for vote in votes["mfg"]
    ]

    return {
        # ----------------------------------------------------
        # EXISTING APPLICATION FIELDS
        # ----------------------------------------------------

        "raw_text": (
            combined_raw_text + "\n" + targeted_text
        ).strip(),

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

        # This is field completeness, NOT calibrated OCR accuracy.
        "confidence": fields_found / 2.0,

        # ----------------------------------------------------
        # OCR DIAGNOSTICS
        # ----------------------------------------------------

        "ocr_score": best_full_result["ocr_score"],
        "ocr_confidence": best_full_result["ocr_confidence"],
        "ocr_variant": "original",
        "ocr_orientation": best_full_result["ocr_orientation"],

        "targeted_ocr_text": targeted_text,
        "ocr_regions": all_diagnostics,

        "expiry_candidates": expiry_candidates,
        "mfg_candidates": mfg_candidates,

        "field_consensus": accepted,
        "field_votes": votes,

        # Medicine information always requires human verification.
        "manual_verification_required": True
    }


# ============================================================
# 11. COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "Single image:\n"
            "python src/stage3_ocr.py image1.jpeg\n\n"
            "Multiple images:\n"
            "python src/stage3_ocr.py image1.jpeg image2.jpeg"
        )
        raise SystemExit(1)

    start = time.perf_counter()

    # One image or multiple images are both supported.
    output = run_ocr_check(sys.argv[1:])

    print("\n" + "=" * 60)
    print("MEDGUARD CONSENSUS-BASED OCR RESULTS")
    print("=" * 60)

    for key, value in output.items():
        print(f"{key}: {value}")

    print(
        "\nTotal processing time:",
        round(time.perf_counter() - start, 2),
        "seconds"
    )
