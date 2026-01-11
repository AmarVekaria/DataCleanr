# backend/core/mappings.py

from difflib import get_close_matches

# Canonical column names for our cleaned output
CANONICAL = [
    "supplier",
    "supplier_code",
    "sku",
    "name",
    "finish",
    "dimensions_mm",
    "category",
    "catalogue_name",
    "cost_net",
    "rrp_net",
    "vat_rate",
    "uom",
    "pack_size",
    "barcode",
    "notes",
]

# Common alternative header names from suppliers
ALIASES = {
    "supplier": ["brand", "manufacturer"],
    "supplier_code": [
        "code",
        "item code",
        "item_code",
        "mpn",
        "manufacturer code",
        "model",
        "product no.",
        "product no",
        "product number",
    ],
    "name": ["description", "product", "product description", "title", "desc"],
    "finish": ["colour", "color", "finish/colour", "surface", "finish colour", "finish"],
    "category": ["range", "category", "product group", "master product group"],
    "cost_net": ["cost", "net cost", "net_cost", "net", "buy price", "trade", "trade price"],
    "rrp_net": ["rrp", "list price", "retail", "list price gbp 20", "list price gbp"],
    "vat_rate": ["vat", "tax", "vat%", "vat rate"],
    "barcode": ["ean", "upc", "barcode", "gtin", "ean code"],
}


def infer_column_mapping(columns):
    """
    Given the raw DataFrame columns, build a mapping from our canonical names
    to the supplier's actual column names.
    """
    mapping = {}
    cols_normalised = [str(c).strip().lower() for c in columns]

    for target in CANONICAL:
        candidates = [target] + ALIASES.get(target, [])
        best = _first_match(cols_normalised, candidates)
        if best is not None:
            original_name = columns[cols_normalised.index(best)]
            mapping[target] = original_name

    return mapping


def _first_match(cols_normalised, candidates):
    """
    Find the best matching column name for any of the candidate phrases.

    1) Try exact matches (case-insensitive).
    2) Then try fuzzy matches for each candidate phrase.
    """
    # 1) Exact / contains match
    for cand in candidates:
        cand_norm = cand.strip().lower()
        if cand_norm in cols_normalised:
            return cand_norm

    # 2) Fuzzy match: try each candidate phrase against all columns
    for cand in candidates:
        cand_norm = cand.strip().lower()
        matches = get_close_matches(cand_norm, cols_normalised, n=1, cutoff=0.8)
        if matches:
            return matches[0]

    return None

# ---- Supplier-specific overrides -------------------------------------------

SUPPLIER_OVERRIDES = {
    # Samuel Heath price list
    "samuel heath": {
        "supplier_code": "Product No.",
        "name": "Description",
        "finish": "Finish",
        "category": "Range",
        "rrp_net": "List Price GBP 20",
    }
}


def get_supplier_override(supplier: str) -> dict:
    """Return a mapping override for a given supplier, if we have one.

    Uses case-insensitive match and also accepts names that start with
    the known supplier key (e.g. "samuel heath jan26 list").
    """
    if not supplier:
        return {}
    key = supplier.strip().lower()

    for s_key, override in SUPPLIER_OVERRIDES.items():
        if key == s_key or key.startswith(s_key):
            return override

    return {}

