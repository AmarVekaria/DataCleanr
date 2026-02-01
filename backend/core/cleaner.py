# backend/core/cleaner.py

import re
import pandas as pd
from .validators import to_float


DEFAULT_ROW = {
    "supplier": "",
    "supplier_code": "",
    "sku": "",
    "name": "",
    "finish": "",
    "dimensions_mm": "",
    "category": "",
    "catalogue_name": "",
    "cost_net": 0.0,
    "rrp_net": 0.0,
    "vat_rate": 0.2,
    "uom": "each",
    "pack_size": 1,
    "barcode": "",
    "notes": "",
}

CANONICAL_ORDER = list(DEFAULT_ROW.keys())

FINISH_CANON = {
    "white matt": "Matt White",
    "matt white": "Matt White",
    "chrome": "Chrome",
    "polished chrome": "Chrome",
}

DIMENSION_REGEXES = [
    re.compile(r"\b(\d{2,4})\s*[x×]\s*(\d{2,4})\s*[x×]\s*(\d{2,4})\s*mm\b", re.I),
    re.compile(r"\b[lhwd]:?\s*(\d{2,4})\s*[x×]\s*[lhwd]:?\s*(\d{2,4})\s*[x×]\s*[lhwd]:?\s*(\d{2,4})\s*mm\b", re.I),
]


def clean_dataframe(df: pd.DataFrame, supplier: str, mapping: dict, options: dict | None = None):
    """
    If options['return_report'] is True -> returns (cleaned_df, report_df)
    Else -> returns cleaned_df only
    """
    options = options or {}

    discount_percent = float(options.get("discount_percent", 0.0) or 0.0)
    code_length = options.get("code_length", None)

    merge_fields_raw = (options.get("merge_fields") or "").strip()
    merge_dedupe = bool(options.get("merge_dedupe", True))

    # Dedup controls
    dedupe_by_supplier_column = (options.get("dedupe_by_supplier_column") or "").strip()  # e.g. Brand
    dedupe_mode = (options.get("dedupe_mode") or "keep_max_rrp").strip().lower()          # keep_max_rrp | keep_first

    return_report = bool(options.get("return_report", False))

    out = pd.DataFrame(index=df.index, columns=CANONICAL_ORDER)
    for k, v in DEFAULT_ROW.items():
        out[k] = v

    def _get_series(canon_key: str, default=""):
        src = mapping.get(canon_key)
        if src and src in df.columns:
            return df[src]
        return pd.Series([default] * len(df), index=df.index)

    # --- Basics ---
    out["supplier"] = supplier

    supplier_code = _get_series("supplier_code", "")
    out["supplier_code"] = supplier_code.map(lambda x: _clean_code(x, code_length))

    raw_name = _get_series("name", "")
    raw_finish = _get_series("finish", "")

    name_clean, dims = zip(*raw_name.map(_extract_dimensions_safe))
    out["name"] = list(name_clean)
    out["dimensions_mm"] = list(dims)

    out["finish"] = raw_finish.map(_normalise_finish)

    # RRP + Cost
    rrp = _get_series("rrp_net", 0).map(to_float).fillna(0.0)
    cost = _get_series("cost_net", 0).map(to_float).fillna(0.0)

    out["rrp_net"] = rrp.round(2)

    if (cost == 0).all() and discount_percent > 0:
        out["cost_net"] = (out["rrp_net"] * (1 - (discount_percent / 100.0))).round(2)
    else:
        out["cost_net"] = cost.round(2)

    # VAT rate
    vat = _get_series("vat_rate", 0.2)
    out["vat_rate"] = vat.map(_parse_vat).fillna(0.2)

    out["barcode"] = _get_series("barcode", "").astype(str).fillna("").str.strip()
    out["uom"] = _get_series("uom", "each").astype(str).fillna("each").str.strip()
    out["category"] = _get_series("category", "").astype(str).fillna("").str.strip()

    # --- catalogue_name ---
    out["catalogue_name"] = _build_catalogue_name(
        df=df,
        out=out,
        merge_fields_raw=merge_fields_raw,
        merge_dedupe=merge_dedupe,
    )

    # --- Deduplicate manufacturer codes (with report) ---
    out_deduped, dedupe_report = _dedupe_supplier_codes(
        out=out,
        df_source=df,
        supplier_col=dedupe_by_supplier_column,
        mode=dedupe_mode,
    )

    cleaned_final = out_deduped[CANONICAL_ORDER].reset_index(drop=True)

    if return_report:
        return cleaned_final, dedupe_report
    return cleaned_final


# ---------------------------
# Helpers
# ---------------------------
def _clean_code(x, code_length=None):
    """
    Preserve leading zeros.
    Only pads when code_length is provided (e.g. Hansgrohe 8).
    Never truncates longer codes.
    """
    if x is None:
        s = ""
    else:
        try:
            if isinstance(x, float) and x.is_integer():
                x = int(x)
        except Exception:
            pass
        s = str(x).strip()

    if not s or s.lower() == "nan":
        return ""

    if s.endswith(".0"):
        s = s[:-2]

    if code_length:
        try:
            n = int(code_length)
            if s.isdigit() and len(s) < n:
                s = s.zfill(n)
        except Exception:
            pass

    return s


def _extract_dimensions_safe(x):
    t = "" if x is None else str(x)
    for rgx in DIMENSION_REGEXES:
        m = rgx.search(t)
        if m:
            dims = "x".join(m.groups())
            clean = rgx.sub("", t).strip()
            return _normalise_title(clean), dims
    return _normalise_title(t), ""


def _normalise_title(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "")).strip()
    return t


def _normalise_finish(f):
    s = "" if f is None else str(f).strip()
    if not s or s.lower() == "nan":
        return ""
    key = s.lower()
    return FINISH_CANON.get(key, s.title())


def _parse_vat(x):
    if x is None:
        return 0.2
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0.2
    s = s.replace("%", "").strip()
    try:
        v = float(s)
        if v > 1:
            return v / 100.0
        return v
    except Exception:
        return 0.2


def _build_catalogue_name(df: pd.DataFrame, out: pd.DataFrame, merge_fields_raw: str, merge_dedupe: bool) -> pd.Series:
    """
    If merge_fields_raw provided (comma separated supplier headings),
    build catalogue_name from those supplier columns.

    For exactly 2 fields (Hansgrohe use-case):
    -> "SHORT DESCRIPTION - 'COLOUR'" (colour optional; skip if blank)

    For 3+ fields:
    -> join with ", " (still blank-safe)

    Always uppercases final output.
    """
    if merge_fields_raw:
        fields = [f.strip() for f in merge_fields_raw.split(",") if f.strip()]

        parts = []
        for f in fields:
            if f in df.columns:
                parts.append(df[f].astype(str).fillna("").str.strip())
            else:
                parts.append(pd.Series([""] * len(df), index=df.index))

        merged = []
        for i in range(len(df)):
            vals = []
            for s in parts:
                v = (s.iloc[i] or "").strip()
                if not v or v.lower() == "nan":
                    continue
                vals.append(v)

            if merge_dedupe and len(vals) >= 2:
                vals = _dedupe_parts(vals)

            if len(vals) == 0:
                merged.append("")
            elif len(vals) == 1:
                merged.append(vals[0])
            elif len(vals) == 2:
                merged.append(f"{vals[0]} - '{vals[1]}'")
            else:
                merged.append(", ".join(vals))

        return pd.Series(merged, index=df.index).str.upper()

    # fallback
    base = out.get("name", "").astype(str).fillna("").str.strip()
    fin = out.get("finish", "").astype(str).fillna("").str.strip()

    merged = []
    for i in range(len(out)):
        b = base.iloc[i]
        f = fin.iloc[i]
        if b and f:
            merged.append(f"{b} - ({f})".upper())
        else:
            merged.append((b or "").upper())
    return pd.Series(merged, index=out.index)


def _dedupe_parts(vals: list[str]) -> list[str]:
    cleaned = []
    for v in vals:
        v_strip = v.strip()
        if not v_strip:
            continue
        if any(v_strip.lower() == c.lower() for c in cleaned):
            continue
        cleaned.append(v_strip)

    if len(cleaned) <= 1:
        return cleaned

    final = []
    for v in cleaned:
        vlow = v.lower()
        contained = False
        for other in cleaned:
            if other is v:
                continue
            olow = other.lower()
            if vlow != olow and vlow in olow:
                contained = True
                break
        if not contained:
            final.append(v)

    return final


def _dedupe_supplier_codes(out: pd.DataFrame, df_source: pd.DataFrame, supplier_col: str, mode: str):
    """
    Returns (kept_df, report_df)

    Dedup key:
      - (supplier, supplier_code) OR (supplier, supplier_code, <supplier_col>)
    supplier_col example: "Brand"

    Report logs:
      - key fields
      - count before
      - distinct rrps/costs
      - kept values
      - kept original row index
      - reason flags
    """
    df = out.copy()

    # Build dedupe key
    key_cols = ["supplier", "supplier_code"]
    temp_key_name = None

    if supplier_col and supplier_col in df_source.columns:
        temp_key_name = "__dedupe_supplier_col__"
        df[temp_key_name] = df_source[supplier_col].astype(str).fillna("").str.strip()
        key_cols.append(temp_key_name)

    # If no duplicates, return empty report
    if not df.duplicated(subset=key_cols, keep=False).any():
        report = pd.DataFrame(columns=[
            "supplier", "supplier_code", supplier_col or "group",
            "duplicate_count", "rrp_values", "cost_values",
            "kept_rrp", "kept_cost", "kept_source_row",
            "flag"
        ])
        if temp_key_name and temp_key_name in df.columns:
            df = df.drop(columns=[temp_key_name])
        return df, report

    # Numeric helper columns
    df["__rrp_num__"] = pd.to_numeric(df["rrp_net"], errors="coerce").fillna(0.0).round(2)
    df["__cost_num__"] = pd.to_numeric(df["cost_net"], errors="coerce").fillna(0.0).round(2)

    # Build a per-group summary report FIRST (before dropping rows)
    grouped = df.groupby(key_cols, dropna=False)

    report_rows = []
    for key, g in grouped:
        if len(g) <= 1:
            continue

        rrps = sorted(g["__rrp_num__"].unique().tolist())
        costs = sorted(g["__cost_num__"].unique().tolist())
        price_conflict = (len(rrps) > 1) or (len(costs) > 1)

        # Decide which row to keep
        if mode == "keep_first":
            kept_row = g.sort_index().iloc[0]
        else:
            kept_row = g.sort_values("__rrp_num__", ascending=False).iloc[0]

        flag = "DUPLICATE_CODE"
        if price_conflict:
            flag = "DUPLICATE_CODE_DIFF_PRICE"

        supplier_val = kept_row.get("supplier", "")
        code_val = kept_row.get("supplier_code", "")

        if temp_key_name:
            supplier_group_val = kept_row.get(temp_key_name, "")
        else:
            supplier_group_val = ""

        report_rows.append({
            "supplier": supplier_val,
            "supplier_code": code_val,
            (supplier_col if supplier_col else "group"): supplier_group_val,
            "duplicate_count": int(len(g)),
            "rrp_values": ", ".join([f"{x:.2f}" for x in rrps]),
            "cost_values": ", ".join([f"{x:.2f}" for x in costs]),
            "kept_rrp": float(kept_row["__rrp_num__"]),
            "kept_cost": float(kept_row["__cost_num__"]),
            "kept_source_row": int(kept_row.name) if str(kept_row.name).isdigit() else str(kept_row.name),
            "flag": flag,
        })

    report_df = pd.DataFrame(report_rows)

    # Now keep rows according to mode
    if mode == "keep_first":
        kept = df.sort_index().groupby(key_cols, as_index=False).head(1)
    else:
        kept = df.sort_values("__rrp_num__", ascending=False).groupby(key_cols, as_index=False).head(1)

    # Append notes flag to kept rows where conflict exists
    conflict_keys = set(report_df.loc[report_df["flag"] == "DUPLICATE_CODE_DIFF_PRICE", ["supplier", "supplier_code"]].apply(tuple, axis=1).tolist())

    def _append_note(existing: str, note: str) -> str:
        existing = (existing or "").strip()
        if not existing:
            return note
        if note.lower() in existing.lower():
            return existing
        return existing + " | " + note

    kept["notes"] = kept.apply(
        lambda r: _append_note(r.get("notes", ""), "DUPLICATE_CODE_DIFF_PRICE")
        if (r.get("supplier", ""), r.get("supplier_code", "")) in conflict_keys
        else (r.get("notes", "") or ""),
        axis=1
    )

    # Cleanup temp/internal cols
    drop_cols = ["__rrp_num__", "__cost_num__"]
    if temp_key_name and temp_key_name in kept.columns:
        drop_cols.append(temp_key_name)
    kept = kept.drop(columns=[c for c in drop_cols if c in kept.columns])

    return kept, report_df
