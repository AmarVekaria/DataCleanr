# backend/core/mappings.py

import re
from difflib import get_close_matches

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
    "rrp_gross",
    "vat_rate",
    "uom",
    "pack_size",
    "barcode",
    "notes",
]

ALIASES = {
    "supplier": ["brand", "manufacturer"],
    "supplier_code": [
        "code", "item_code", "item code", "product code", "product no", "product number",
        "product no.", "mpn", "manufacturer product code", "manufacturerproductcode",
        "manufacturerproduct", "manufacturerproductcode", "manufacturerproductcode",
        "manufacturerproductcode", "manufacturer productcode", "supplier product code"
    ],
    "name": ["description", "short description", "product title", "full description", "title", "desc"],
    "finish": ["colour", "color", "finish/colour", "surface"],
    "cost_net": ["cost", "net_cost", "net", "buy price", "trade", "trade price", "dealer price"],

    # Net / Ex-VAT RRP synonyms (including “MRP”)
    "rrp_net": [
        "rrp ex vat", "rrp ex-vat", "rrp net", "list price ex vat", "list price ex-vat",
        "mrp", "m.r.p", "net rrp", "net list", "net price", "ex vat", "ex-vat"
    ],

    # Gross / Inc-VAT RRP synonyms (IMPORTANT: include plain "rrp")
    "rrp_gross": [
        "rrp", "rrp inc vat", "rrp inc-vat", "rrp gross", "list price inc vat", "list price inc-vat",
        "gross rrp", "gross list", "gross price", "inc vat", "inc-vat"
    ],

    "vat_rate": ["vat", "tax", "vat%", "vat rate", "tax rate"],
    "barcode": ["ean", "upc", "barcode", "gtin"],
    "uom": ["unit", "uom", "selling unit", "purchase unit"],
}


# ---------- normalisation helpers ----------
def normalise_header(h: str) -> str:
    h = str(h or "").strip().lower()
    h = re.sub(r"\s+", " ", h)
    return h


def _has_any(h_norm: str, tokens: list[str]) -> bool:
    return any(t in h_norm for t in tokens)


def looks_like_month_price(h: str) -> bool:
    """
    Catches:
      "Feb MRP", "February RRP", "Mar-24 RRP", "FEB-2026 MRP", "Feb_26 RRP"
    """
    hh = normalise_header(h)

    months = ["jan", "january", "feb", "february", "mar", "march", "apr", "april",
              "may", "jun", "june", "jul", "july", "aug", "august", "sep", "sept",
              "september", "oct", "october", "nov", "november", "dec", "december"]

    # month at start: "feb ..." or "feb-26 ..." or "feb_26 ..."
    if any(hh.startswith(m + " ") for m in months):
        return True
    if any(hh.startswith(m + "-") for m in months):
        return True
    if any(hh.startswith(m + "_") for m in months):
        return True

    # month appears with a year marker: "feb 2026", "mar-24"
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[\s\-_]?\d{2,4}\b", hh):
        return True

    return False


# ---------- main mapping ----------
def infer_column_mapping(columns):
    """
    Returns mapping from canonical key -> original column name.
    Explicitly supports:
      - rrp_net (ex VAT): MRP preferred
      - rrp_gross (inc VAT): RRP preferred
    """

    mapping = {}
    cols_norm = [normalise_header(c) for c in columns]
    norm_to_real = {cols_norm[i]: columns[i] for i in range(len(columns))}

    def pick_best(target_key: str):
        candidates = [target_key] + ALIASES.get(target_key, [])
        best = None

        # Special logic for rrp_net / rrp_gross (handles "Feb MRP" / "Feb RRP")
        if target_key in ("rrp_net", "rrp_gross"):
            month_cols = [c for c in columns if looks_like_month_price(c)]
            if month_cols:
                # Rank month cols with strong rules first
                for c in month_cols:
                    cn = normalise_header(c)

                    if target_key == "rrp_net":
                        # Prefer MRP strongly (your convention: net)
                        if re.search(r"\b(mrp|m\.r\.p)\b", cn):
                            return c
                        # Explicit net/ex-vat also indicates net
                        if _has_any(cn, ["ex vat", "ex-vat", "net"]):
                            return c

                    if target_key == "rrp_gross":
                        # Explicit inc-vat/gross indicates gross
                        if _has_any(cn, ["inc vat", "inc-vat", "gross"]):
                            return c
                        # Month + RRP (common gross)
                        if re.search(r"\brrp\b", cn):
                            return c

            # If no month columns, still prioritise MRP vs RRP anywhere
            all_cols = list(columns)
            if target_key == "rrp_net":
                for c in all_cols:
                    cn = normalise_header(c)
                    if re.search(r"\b(mrp|m\.r\.p)\b", cn):
                        return c
                    if _has_any(cn, ["rrp ex vat", "rrp ex-vat", "net rrp", "ex vat", "ex-vat", "net"]):
                        return c

            if target_key == "rrp_gross":
                for c in all_cols:
                    cn = normalise_header(c)
                    if _has_any(cn, ["rrp inc vat", "rrp inc-vat", "inc vat", "inc-vat", "gross"]):
                        return c
                    # plain RRP (but do NOT steal from a net column if it contains MRP)
                    if re.search(r"\brrp\b", cn) and not re.search(r"\b(mrp|m\.r\.p)\b", cn):
                        return c

        # Exact match
        for cand in candidates:
            cand_norm = normalise_header(cand)
            for c_norm in cols_norm:
                if cand_norm == c_norm:
                    return norm_to_real[c_norm]

        # contains-style matching
        for cand in candidates:
            cand_norm = normalise_header(cand)
            for c_norm in cols_norm:
                if cand_norm and cand_norm in c_norm:
                    best = norm_to_real[c_norm]
                    break
            if best:
                break

        # fuzzy last resort
        if not best:
            base = candidates[0]
            matches = get_close_matches(normalise_header(base), cols_norm, n=1, cutoff=0.83)
            if matches:
                best = norm_to_real[matches[0]]

        return best

    for key in CANONICAL:
        col = pick_best(key)
        if col:
            mapping[key] = col

    # Safety: avoid mapping the same source column to both rrp_net and rrp_gross when possible
    if mapping.get("rrp_net") and mapping.get("rrp_gross") and mapping["rrp_net"] == mapping["rrp_gross"]:
        # If the chosen column contains MRP -> keep as net, drop gross
        cn = normalise_header(mapping["rrp_net"])
        if re.search(r"\b(mrp|m\.r\.p)\b", cn):
            mapping.pop("rrp_gross", None)
        else:
            # otherwise keep gross, drop net
            mapping.pop("rrp_net", None)

    return mapping


def get_supplier_override(supplier: str) -> dict:
    """
    Optional hard overrides for specific suppliers.
    Keep empty for now or add later.
    """
    return {}
