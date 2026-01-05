# backend/core/cleaner.py

import re
import pandas as pd
from .validators import to_float

# Regexes to extract dimensions from product titles/descriptions
DIMENSION_REGEXES = [
    # e.g. L:425 X W:177 X H:154MM
    re.compile(
        r"\b[lhwd]:?\s*(\d{2,4})\s*[x×]\s*[lhwd]:?\s*(\d{2,4})\s*[x×]\s*[lhwd]:?\s*(\d{2,4})\s*mm\b",
        re.I,
    ),
    # e.g. 425x177x154 mm
    re.compile(
        r"\b(\d{2,4})\s*[x×]\s*(\d{2,4})\s*[x×]\s*(\d{2,4})\s*mm\b",
        re.I,
    ),
    # e.g. H154 x W177 x D425
    re.compile(
        r"\b[hwd](\d{2,4})\s*[x×]\s*w(\d{2,4})\s*[x×]\s*d(\d{2,4})\b",
        re.I,
    ),
]

# Canonical finishes map
FINISH_CANON = {
    "white matt": "Matt White",
    "matt white": "Matt White",
    "chrome": "Chrome",
    "polished chrome": "Chrome",
}

# Default row template (canonical schema)
DEFAULT_ROW = {
    "supplier": "",
    "supplier_code": "",
    "sku": "",
    "name": "",
    "finish": "",
    "dimensions_mm": "",
    "category": "",
    "cost_net": None,
    "rrp_net": None,
    "vat_rate": 0.2,
    "uom": "each",
    "pack_size": 1,
    "barcode": "",
    "notes": "",
}

CANONICAL_ORDER = list(DEFAULT_ROW.keys())


def clean_dataframe(df: pd.DataFrame, supplier: str, mapping: dict) -> pd.DataFrame:
    """
    Core cleaning function.
    Takes a raw supplier DataFrame + column mapping, returns a cleaned DataFrame
    in the canonical schema.
    """
    # Start with an empty frame using canonical columns
    out = pd.DataFrame(columns=CANONICAL_ORDER)
    out = out.assign(**DEFAULT_ROW)

    def get(col, default=""):
        """
        Helper to fetch a column based on mapping.
        Returns a Series if the column exists, otherwise a Series of defaults.
        """
        src = mapping.get(col)
        if src in df.columns:
            return df[src]
        # fall back to a series of default values (same length as df)
        return pd.Series([default] * len(df))

    # Basic fields
    out["supplier"] = supplier
    out["supplier_code"] = get("supplier_code")
    raw_name = get("name")
    raw_finish = get("finish")
    out["barcode"] = get("barcode")

    # VAT
    vat_series = get("vat_rate")
    out["vat_rate"] = _vat_series(vat_series, default=0.2)

    # Extract dimensions from name and clean title
    name_clean, dims = zip(*raw_name.fillna("").map(_extract_dimensions))
    out["name"] = name_clean
    out["dimensions_mm"] = dims

    # Finish normalisation
    out["finish"] = raw_finish.fillna("").map(_normalise_finish)

    # Costs
    out["cost_net"] = get("cost_net").map(to_float)
    out["rrp_net"] = get("rrp_net").map(to_float)

    # Deduplicate by supplier + supplier_code, keep first occurrence
    out = out.fillna("")
    out = out.groupby(["supplier", "supplier_code"], as_index=False).first()

    # Final column order
    out = out[CANONICAL_ORDER]
    return out


def _extract_dimensions(text):
    # Ensure text is always a string (handles ints, floats, NaN, None)
    if text is None:
        t = ""
    else:
        t = str(text)

    for rgx in DIMENSION_REGEXES:
        m = rgx.search(t)
        if m:
            dims = "x".join(m.groups())
            clean = rgx.sub("", t).strip()
            return _normalise_title(clean), dims

    return _normalise_title(t), ""


def _normalise_title(t: str) -> str:
    """Basic title cleanup and title-casing, preserving special tokens."""
    t = re.sub(r"\s+", " ", t).strip()

    def _tc(word: str) -> str:
        keep = {"WC", "XL", "LED", "RGB", "2-Tap", "3-Tap"}
        return word if word.upper() in keep else word.capitalize()

    return " ".join(_tc(w) for w in t.split(" "))


def _normalise_finish(f: str) -> str:
    """Normalise finish names using the FINISH_CANON map."""
    f2 = f.strip().lower()
    return FINISH_CANON.get(f2, f.title())


def _vat_series(series: pd.Series, default: float = 0.2) -> pd.Series:
    """Normalise VAT column to a numeric rate (e.g. 0.2)."""
    try:
        return series.fillna(default).map(
            lambda x: _parse_vat_value(x, default)
        )
    except Exception:
        # Fall back to flat default if anything goes wrong
        return pd.Series([default] * len(series))


def _parse_vat_value(x, default: float) -> float:
    if x is None:
        return default
    s = str(x).strip()
    if not s:
        return default
    s = s.replace("%", "")
    try:
        val = float(s)
    except ValueError:
        return default
    # If looks like "20", treat as 20% => 0.2
    return val / 100 if val > 1 else val
