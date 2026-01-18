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
