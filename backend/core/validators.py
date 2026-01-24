# backend/core/validators.py

import re


def to_float(x):
    """
    Convert a value to float, handling currency symbols and junk text.

    Examples:
    - "£123.45"      -> 123.45
    - "123,45"       -> 123.45
    - "£99 inc VAT"  -> 99.0
    - "" or None     -> None
    """
    if x is None:
        return None

    s = str(x)
    if s.strip() == "":
        return None

    # Strip currency symbols and commas
    s = s.replace("£", "").replace(",", "").strip()

    # Try a direct float conversion first
    try:
        return float(s)
    except Exception:
        # Fallback: extract the first numeric part from the string
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
        return float(m.group(1)) if m else None

def normalise_barcode(x):
    """
    Convert barcodes/EANs into a safe string:
    - handles numeric/scientific notation
    - removes trailing .0
    - strips spaces
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return ""

    # If it looks like scientific notation (e.g., 4.059625e+12)
    if "e" in s.lower():
        try:
            # Convert via Decimal to avoid float rounding issues
            from decimal import Decimal
            s = format(Decimal(s), "f")
        except Exception:
            pass

    # Remove trailing .0 if it's a whole number
    if s.endswith(".0"):
        s = s[:-2]

    # Remove any stray spaces
    s = s.replace(" ", "")

    return s
