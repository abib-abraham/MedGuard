
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from stage3_ocr import (
    parse_month_year,
    extract_expiry_date,
    extract_mfg_date,
)

tests = [
    "OCT2024",
    "SEP2026",
    "SE22426",
    "OCH2024",
    "0012024",
    "DEC2027",
]

print("\nDATE PARSER TEST")
print("-" * 45)

for value in tests:
    print(f"{value:15} -> {parse_month_year(value)}")

print("\nLABELED DATE TEST")
print("-" * 45)

samples = [
    "EXP SEP2026",
    "EXP SE22426",
    "MFD OCH2024",
    "MFD 0012024",
]

for sample in samples:
    print("\nText:", sample)
    print("Expiry:", extract_expiry_date(sample))
    print("MFG:", extract_mfg_date(sample))
