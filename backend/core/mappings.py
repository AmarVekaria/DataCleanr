# backend/core/mappings.py

import re
from difflib import get_close_matches
import pandas as pd

CANONICAL = [
    "supplier",
    "supplier_code",
    "sku",
    "name",
    "finish",
    "dimensions_mm",
    "category",
    "cost_net",
    "rrp_net",
    "rrp_gross",   # optional inc VAT/gross
    "vat_rate",
    "uom",
    "pack_size",
    "barcode",
    "notes",
]

ALIASES = {
    "supplier": ["brand", "manufacturer"],
    "supplier_code": [
        "code", "item_code", "product code", "product no", "product number",
        "mpn", "manufacturer product code", "manufacturerproductcode"
    ],
    "name": ["description", "short description", "product title", "full description", "title", "desc"],
    "finish": ["colour", "color", "finish/colour", "surface"],
    "cost_net": ["cost", "net_cost", "net", "buy price", "trade", "trade price", "dealer price"],

    # Net / Ex-VAT RRP synonyms (including “MRP”)
    "rrp_net": [
        "rrp ex vat", "rrp ex-vat", "rrp net",
        "list price ex vat", "list price ex-vat",
        "net rrp", "net list", "net price",
        "mrp", "m.r.p"
    ],

    # Gross / Inc-VAT RRP synonyms
    "rrp_gross": [
        "rrp inc vat", "rrp inc-vat", "rrp gross",
        "list price inc vat", "list price inc-vat",
        "gross rrp", "gross list", "gross price"
    ],

    "vat_rate": ["vat", "tax", "vat%", "vat rate"],
    "barcode": ["ean", "upc", "barcode", "gtin"],
    "uom": ["unit", "uom", "selling unit", "purchase unit"],
}

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")


def normalise_header(h: str) -> str:
    h = str(h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h


def _strip_month_year_currency(h: str) -> str:
    """
    Turns:
      'Feb 26 MRP (Ex Vat)' -> 'mrp ex vat'
      'List Price GBP 2026' -> 'list price gbp'
    """
    s = normalise_header(h)

    # remove brackets
    s = re.sub(r"[\(\)\[\]]", " ", s)

    # remove obvious year tokens
    s = re.sub(r"\b20\d{2}\b", " ", s)

    # remove common short year tokens like "26" (but only when surrounded by spaces)
    s = re.sub(r"\b\d{2}\b", " ", s)

    # remove month tokens at the start
    for m in MONTHS:
        s = re.sub(rf"^{m}\b", " ", s)
        s = re.sub(rf"^{m}\s+", " ", s)

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _score_price_header(h: str) -> int:
    """
    Score a header for rrp_net selection.
    Higher = more likely it is EX-VAT / NET list price.
    """
    raw = normalise_header(h)
    s = _strip_month_year_currency(h)

    score = 0

    # Strong net/ex VAT cues
    if "ex vat" in raw or "ex-vat" in raw or "ex vat" in s or "ex-vat" in s:
        score += 60
    if "net" in raw or "net" in s:
        score += 30

    # MRP cues (your convention: MRP = ex VAT)
    if "mrp" in raw or "m.r.p" in raw or "mrp" in s:
        score += 55

    # RRP cues (could be gross or net depending on supplier)
    if "rrp" in raw or "rrp" in s:
        score += 20

    # list price cues
    if "list price" in raw or "list price" in s:
        score += 15

    # Currency hints
    if "gbp" in raw or "gbp" in s or "£" in raw:
        score += 5

    # Penalise gross cues
    if "inc vat" in raw or "inc-vat" in raw or "gross" in raw:
        score -= 40

    # Generic "price" is acceptable but weak fallback
    if raw.strip() == "price" or s.strip() == "price":
        score += 8

    return score


def _pick_best_general(target_key: str, cols_norm: list[str], norm_to_real: dict) -> str | None:
    candidates = [target_key] + ALIASES.get(target_key, [])

    # exact match
    for cand in candidates:
        cand_norm = normalise_header(cand)
        for c_norm in cols_norm:
            if cand_norm == c_norm:
                return norm_to_real[c_norm]

    # contains match
    for cand in candidates:
        cand_norm = normalise_header(cand)
        for c_norm in cols_norm:
            if cand_norm in c_norm:
                return norm_to_real[c_norm]

    # fuzzy last resort
    base = candidates[0]
    matches = get_close_matches(normalise_header(base), cols_norm, n=1, cutoff=0.83)
    if matches:
        return norm_to_real[matches[0]]

    return None


def infer_column_mapping(columns):
    """
    Returns mapping from canonical key -> original column name.

    Key behaviour:
    - rrp_net is selected via scoring so it works for:
        'Feb 26 MRP (Ex Vat)', 'Price', 'List Price GBP 2026'
    - rrp_gross is selected best-effort (inc VAT).
    """
    mapping = {}
    cols = list(columns)
    cols_norm = [normalise_header(c) for c in cols]
    norm_to_real = {cols_norm[i]: cols[i] for i in range(len(cols))}

    # 1) pick everything except rrp_net / rrp_gross via general logic
    for key in CANONICAL:
        if key in ("rrp_net", "rrp_gross"):
            continue
        col = _pick_best_general(key, cols_norm, norm_to_real)
        if col:
            mapping[key] = col

    # 2) robust rrp_net selection via scoring
    price_candidates = []
    for c in cols:
        sc = _score_price_header(c)
        if sc > 0:
            price_candidates.append((sc, c))

    price_candidates.sort(key=lambda x: x[0], reverse=True)
    if price_candidates:
        mapping["rrp_net"] = price_candidates[0][1]

    # 3) rrp_gross selection (best effort)
    gross_best = _pick_best_general("rrp_gross", cols_norm, norm_to_real)
    if gross_best:
        mapping["rrp_gross"] = gross_best

    # 4) fallback if nothing matched but literal "price" exists
    if "rrp_net" not in mapping:
        for c in cols:
            if normalise_header(c) == "price":
                mapping["rrp_net"] = c
                break

    return mapping


def get_supplier_override(supplier: str) -> dict:
    """
    Optional hard overrides for specific suppliers.
    Leave empty for now.
    """
    return {}


def mapping_audit(mapping: dict, sheet_name: str) -> pd.DataFrame:
    """
    Human readable mapping report for Excel export.
    Used by app.py to write the MappingAudit sheet.
    """
    audit = {
        "Detected Supplier Code Column": mapping.get("supplier_code", ""),
        "Detected Name Column": mapping.get("name", ""),
        "Detected Finish Column": mapping.get("finish", ""),
        "Detected RRP (Ex VAT) Column": mapping.get("rrp_net", ""),
        "Detected RRP (Inc VAT) Column": mapping.get("rrp_gross", ""),
        "Detected VAT Column": mapping.get("vat_rate", ""),
        "Workbook Sheet Used": sheet_name,
    }
    return pd.DataFrame(audit.items(), columns=["Field", "Detected Value"])