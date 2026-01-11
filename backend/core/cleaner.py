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
    "catalogue_name": "",
    "cost_net": None,
    "rrp_net": None,
    "vat_rate": 0.2,
    "uom": "each",
    "pack_size": 1,
    "barcode": "",
    "notes": "",
}

CANONICAL_ORDER = list(DEFAULT_ROW.keys())


def clean_dataframe(df: pd.DataFrame, supplier: str, mapping: dict, options: dict | None = None):
    """
    Core cleaning function.
    - Maps supplier columns to canonical columns
    - Cleans codes & titles
    - Extracts dimensions
    - Normalises finishes
    - Rounds RRP to 2dp
    - Optionally computes cost_net using a discount percentage
    """
    options = options or {}
    discount_percent = float(options.get("discount_percent", 0.0))

    # Build output frame using canonical order
    out = pd.DataFrame(columns=CANONICAL_ORDER)
    out = out.assign(**DEFAULT_ROW)

    # Helper to safely extract mapped columns
    def get(col, default=""):
        src = mapping.get(col)
        if src and src in df.columns:
            return df[src]
        return pd.Series([default] * len(df))

    # -----------------------------------------------------------
    # Basic fields
    # -----------------------------------------------------------
    out["supplier"] = supplier

    # Supplier code cleaning (SKU normalisation)
    raw_code = get("supplier_code")
    out["supplier_code"] = (
        raw_code.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.upper()
    )

    raw_name = get("name")
    raw_finish = get("finish")
    raw_category = get("category")
    raw_rrp = get("rrp_net")
    raw_barcode = get("barcode")

    out["barcode"] = raw_barcode.fillna("").astype(str).str.strip()

    # VAT
    vat_series = get("vat_rate")
    out["vat_rate"] = _vat_series(vat_series, default=0.2)

    # -----------------------------------------------------------
    # Name + dimensions extraction
    # -----------------------------------------------------------
    name_clean, dims = zip(*raw_name.fillna("").map(_extract_dimensions))
    out["name"] = name_clean
    out["dimensions_mm"] = dims

    # -----------------------------------------------------------
    # Finish
    # -----------------------------------------------------------
    out["finish"] = raw_finish.fillna("").map(_normalise_finish)

    # -----------------------------------------------------------
    # Category
    # -----------------------------------------------------------
    out["category"] = raw_category.fillna("").astype(str).str.strip()

    # -----------------------------------------------------------
    # RRP / Prices
    # -----------------------------------------------------------
    out["rrp_net"] = raw_rrp.map(to_float)
    out["rrp_net"] = pd.to_numeric(out["rrp_net"], errors="coerce").fillna(0).round(2)

    # -----------------------------------------------------------
    # Optional discount → cost_net
    # discount_percent may be 40 or 0.4
    # -----------------------------------------------------------
    disc = discount_percent
    if disc > 1:
        disc = disc / 100.0
    if disc < 0:
        disc = 0.0
    if disc > 1:
        disc = 1.0

    out["cost_net"] = (out["rrp_net"] * (1 - disc)).round(2)

    # -----------------------------------------------------------
    # Create catalogue_name (Option B)
    # -----------------------------------------------------------
    def _safe_upper(val):
        s = "" if val is None else str(val)
        s = s.strip()
        return s.upper() if s else ""

    out["catalogue_name"] = (
        out["category"].map(_safe_upper)
        + ", "
        + out["name"].map(_safe_upper)
        + " - ("
        + out["finish"].map(_safe_upper)
        + ")"
    )

    # If no category → remove leading comma
    mask_blank_cat = out["category"].astype(str).str.strip() == ""
    out.loc[mask_blank_cat, "catalogue_name"] = (
        out["name"].map(_safe_upper)
        + " - ("
        + out["finish"].map(_safe_upper)
        + ")"
    )

    # -----------------------------------------------------------
    # Deduplicate by supplier + supplier_code
    # -----------------------------------------------------------
    out = out.fillna("")
    out = out.groupby(["supplier", "supplier_code"], as_index=False).first()

    # -----------------------------------------------------------
    # Return final frame in canonical order
    # -----------------------------------------------------------
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
    return val / 100 if val > 1 else val
