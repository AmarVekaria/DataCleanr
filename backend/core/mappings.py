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
        "code", "item code", "item_code", "mpn",
        "manufacturer code", "model", "product no.", "product no", "product number"
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
    # Normalise to lowercase/stripped for comparison
    cols_normalised = [str(c).strip().lower() for c in columns]

    for target in CANONICAL:
        candidates = [target] + ALIASES.get(target, [])
        best = _first_match(cols_normalised, candidates)
        if best is not None:
            # best is the normalised name; map back to original case
            original_name = columns[cols_normalised.index(best)]
            mapping[target] = original_name

    return mapping


def _first_match(cols_normalised, candidates):
    """
    Find the first candidate that matches exactly (case-insensitive),
    otherwise fall back to fuzzy matching.
    """
    # Exact / contains match
    for cand in candidates:
        cand_norm = cand.strip().lower()
        if cand_norm in cols_normalised:
            return cand_norm

    # Fuzzy match on the main canonical name only as a last resort
    main = candidates[0].strip().lower()
    matches = get_close_matches(main, cols_normalised, n=1, cutoff=0.8)
    return matches[0] if matches else None

# ---- Supplier-specific overrides -------------------------------------------

SUPPLIER_OVERRIDES = {
    "samuel heath": {
        "supplier_code": "Product No.",
        "name": "Description",
        "finish": "Finish",
        "category": "Range",
        "rrp_net": "List Price GBP 20",
    }
}


def get_supplier_override(supplier: str) -> dict:
    if not supplier:
        return {}
    key = supplier.strip().lower()
    return SUPPLIER_OVERRIDES.get(key, {})
