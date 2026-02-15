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
    "rrp_gross": 0.0,
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

PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")  # {Supplier Heading}


def clean_dataframe(df: pd.DataFrame, supplier: str, mapping: dict, options: dict | None = None):
    """
    Canonical clean.

    options:
      discount_percent: float
      code_length: int|None (pads numeric codes only, never truncates)

      merge_fields: "ColA,ColB"
      merge_template: "{ColA} - '{ColB}'"
      merge_dedupe: bool

      dedupe_by_supplier_column: e.g. "Brand"
      dedupe_mode: "keep_max_rrp" | "keep_first"
      return_report: bool -> returns (cleaned_df, report_df)

      drop_blank_rows: bool (default True)
      drop_section_headers: bool (default True)
      drop_repeated_headers: bool (default True)
    """
    options = options or {}

    discount_percent = float(options.get("discount_percent", 0.0) or 0.0)
    code_length = options.get("code_length", None)

    merge_fields_raw = (options.get("merge_fields") or "").strip()
    merge_template = (options.get("merge_template") or "").strip()
    merge_dedupe = bool(options.get("merge_dedupe", True))

    dedupe_by_supplier_column = (options.get("dedupe_by_supplier_column") or "").strip()
    dedupe_mode = (options.get("dedupe_mode") or "keep_max_rrp").strip().lower()

    return_report = bool(options.get("return_report", False))

    drop_blank_rows = bool(options.get("drop_blank_rows", True))
    drop_section_headers = bool(options.get("drop_section_headers", True))
    drop_repeated_headers = bool(options.get("drop_repeated_headers", True))

    # -------------------------------------------------
    # Pre-filter messy supplier sheets (headers/blank blocks)
    # -------------------------------------------------
    df = _pre_filter_rows(
        df=df,
        mapping=mapping,
        drop_blank_rows=drop_blank_rows,
        drop_section_headers=drop_section_headers,
        drop_repeated_headers=drop_repeated_headers,
    )

    # If everything got filtered (rare but possible), return empty canonical
    if df.empty:
        out_empty = pd.DataFrame(columns=CANONICAL_ORDER)
        return (out_empty, pd.DataFrame()) if return_report else out_empty

    out = pd.DataFrame(index=df.index, columns=CANONICAL_ORDER)
    for k, v in DEFAULT_ROW.items():
        out[k] = v

    def _get_series(canon_key: str, default=""):
        src = mapping.get(canon_key)
        if src and src in df.columns:
            return df[src]
        return pd.Series([default] * len(df), index=df.index)

    # ---------------------------
    # Basics
    # ---------------------------
    out["supplier"] = supplier

    supplier_code = _get_series("supplier_code", "")
    out["supplier_code"] = supplier_code.map(lambda x: _clean_code(x, code_length))

    raw_name = _get_series("name", "")
    raw_finish = _get_series("finish", "")

    name_clean, dims = zip(*raw_name.map(_extract_dimensions_safe))
    out["name"] = list(name_clean)
    out["dimensions_mm"] = list(dims)
    out["finish"] = raw_finish.map(_normalise_finish)

    # ---------------------------
    # Prices
    # ---------------------------
    rrp_net = _get_series("rrp_net", 0).map(to_float).fillna(0.0).round(2)
    rrp_gross = _get_series("rrp_gross", 0).map(to_float).fillna(0.0).round(2)
    cost_net = _get_series("cost_net", 0).map(to_float).fillna(0.0)

    out["rrp_net"] = rrp_net
    out["rrp_gross"] = rrp_gross

    # If no explicit cost but discount provided, compute from NET RRP
    if (cost_net == 0).all() and discount_percent > 0:
        out["cost_net"] = (out["rrp_net"] * (1 - (discount_percent / 100.0))).round(2)
    else:
        out["cost_net"] = cost_net.round(2)

    # VAT
    vat = _get_series("vat_rate", 0.2)
    out["vat_rate"] = vat.map(_parse_vat).fillna(0.2)

    # Other
    out["barcode"] = _get_series("barcode", "").astype(str).fillna("").str.strip()
    out["uom"] = _get_series("uom", "each").astype(str).fillna("each").str.strip()
    out["category"] = _get_series("category", "").astype(str).fillna("").str.strip()
    out["notes"] = _get_series("notes", "").astype(str).fillna("").str.strip()

    # ---------------------------
    # catalogue_name (template > fields > fallback)
    # ---------------------------
    out["catalogue_name"] = _build_catalogue_name(
        df=df,
        out=out,
        merge_template=merge_template,
        merge_fields_raw=merge_fields_raw,
        merge_dedupe=merge_dedupe,
    )

    # ---------------------------
    # Deduplicate supplier codes (optional)
    # ---------------------------
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
# Pre-filter stage
# ---------------------------
def _pre_filter_rows(
    df: pd.DataFrame,
    mapping: dict,
    drop_blank_rows: bool,
    drop_section_headers: bool,
    drop_repeated_headers: bool,
) -> pd.DataFrame:
    """
    Generic cleanup for supplier 'catalogue layout' sheets:
      - drop fully blank rows
      - drop repeated header rows inside the body
      - drop section header rows (usually in col A) that break filtering
    """
    if df is None or df.empty:
        return df

    work = df.copy()

    # 1) Drop fully blank rows
    if drop_blank_rows:
        work = work.dropna(how="all")

        # Also treat rows where every cell is ""/whitespace as blank
        as_text = work.astype(str).applymap(lambda x: x.strip() if isinstance(x, str) else str(x).strip())
        work = work[~as_text.apply(lambda r: all(v == "" or v.lower() == "nan" for v in r.values), axis=1)]

    if work.empty:
        return work

    # Identify key columns using mapping (if present)
    code_col = mapping.get("supplier_code") if mapping else None
    name_col = mapping.get("name") if mapping else None
    rrp_net_col = mapping.get("rrp_net") if mapping else None
    rrp_gross_col = mapping.get("rrp_gross") if mapping else None

    # 2) Drop repeated header rows inside the sheet
    # A "repeated header row" often looks like the actual column names repeated as values.
    if drop_repeated_headers and name_col and name_col in work.columns:
        colnames_lower = set([str(c).strip().lower() for c in work.columns])

        def _row_looks_like_header(row) -> bool:
            vals = [str(v).strip().lower() for v in row.values]
            hits = sum(1 for v in vals if v in colnames_lower)
            # if lots of cells match column names, it's likely a header row
            return hits >= max(3, int(0.35 * len(vals)))

        work = work[~work.apply(_row_looks_like_header, axis=1)]

    if work.empty:
        return work

    # 3) Drop section header rows (text in first column, but no code + no price)
    if drop_section_headers:
        first_col = work.columns[0]  # you said column A usually contains section headers

        def _has_value(series, idx) -> bool:
            try:
                v = series.loc[idx]
            except Exception:
                return False
            s = "" if v is None else str(v).strip()
            return s != "" and s.lower() != "nan"

        code_series = work[code_col] if code_col and code_col in work.columns else None
        rrp_net_series = work[rrp_net_col] if rrp_net_col and rrp_net_col in work.columns else None
        rrp_gross_series = work[rrp_gross_col] if rrp_gross_col and rrp_gross_col in work.columns else None

        keep_mask = []
        for idx in work.index:
            a_text = str(work.at[idx, first_col]) if first_col in work.columns else ""
            a_text = "" if a_text is None else str(a_text).strip()

            code_present = _has_value(code_series, idx) if code_series is not None else False

            # Price present if it parses to >0 (string like "£12.34" should count)
            price_present = False
            if rrp_net_series is not None:
                price_present = (to_float(rrp_net_series.loc[idx]) or 0) > 0
            if (not price_present) and rrp_gross_series is not None:
                price_present = (to_float(rrp_gross_series.loc[idx]) or 0) > 0

            # Section header heuristic:
            # - text in column A
            # - no code
            # - no price
            # - and text is short-ish / looks like a category label
            looks_like_section = (
                (a_text != "" and a_text.lower() != "nan")
                and (not code_present)
                and (not price_present)
                and (len(a_text) <= 60)
            )

            # Keep rows unless they look like a pure section header
            keep_mask.append(not looks_like_section)

        work = work.loc[keep_mask]

    return work


# ---------------------------
# Helpers
# ---------------------------
def _clean_code(x, code_length=None):
    """
    Preserve leading zeros.
    Only pads when code_length is provided (e.g. Hansgrohe 8).
    Never truncates longer codes.
    Only pads numeric-only strings.
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
    return re.sub(r"\s+", " ", (t or "")).strip()


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
        return (v / 100.0) if v > 1 else v
    except Exception:
        return 0.2


def _build_catalogue_name(df: pd.DataFrame, out: pd.DataFrame, merge_template: str, merge_fields_raw: str, merge_dedupe: bool) -> pd.Series:
    # 1) Template mode
    if merge_template:
        placeholders = [p.strip() for p in PLACEHOLDER_RE.findall(merge_template) if p.strip()]

        series_map = {}
        for col in placeholders:
            if col in df.columns:
                series_map[col] = df[col].astype(str).fillna("").str.strip()
            else:
                series_map[col] = pd.Series([""] * len(df), index=df.index)

        merged = []
        for i in range(len(df)):
            raw_vals = []
            for col in placeholders:
                v = (series_map[col].iloc[i] or "").strip()
                if v and v.lower() != "nan":
                    raw_vals.append(v)

            if merge_dedupe and len(raw_vals) >= 2:
                raw_vals = _dedupe_parts(raw_vals)

            rendered = merge_template
            for col in placeholders:
                v = (series_map[col].iloc[i] or "").strip()
                if not v or v.lower() == "nan":
                    v = ""
                rendered = rendered.replace("{" + col + "}", v)

            rendered = _cleanup_optional_template(rendered)

            cleaned = rendered.strip().strip("- ,|")
            if not cleaned:
                cleaned = " - ".join(raw_vals) if raw_vals else ""

            merged.append(cleaned)

        return pd.Series(merged, index=df.index).str.upper()

    # 2) Fields mode
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

    # 3) Fallback
    base = out.get("name", "").astype(str).fillna("").str.strip()
    fin = out.get("finish", "").astype(str).fillna("").str.strip()

    merged = []
    for i in range(len(out)):
        b = base.iloc[i]
        f = fin.iloc[i]
        merged.append((f"{b} - ({f})" if (b and f) else (b or "")).upper())
    return pd.Series(merged, index=out.index)


def _cleanup_optional_template(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    s = re.sub(r"\s*-\s*''\s*$", "", s)
    s = re.sub(r"\s*-\s*\"\"\s*$", "", s)
    s = re.sub(r"\s*-\s*['\"]\s*$", "", s)
    return s.strip().rstrip(" -,")


def _dedupe_parts(vals: list[str]) -> list[str]:
    cleaned = []
    for v in vals:
        v = (v or "").strip()
        if not v:
            continue
        if any(v.lower() == c.lower() for c in cleaned):
            continue
        cleaned.append(v)

    if len(cleaned) <= 1:
        return cleaned

    final = []
    for v in cleaned:
        vlow = v.lower()
        if any((vlow != o.lower() and vlow in o.lower()) for o in cleaned):
            continue
        final.append(v)
    return final


def _dedupe_supplier_codes(out: pd.DataFrame, df_source: pd.DataFrame, supplier_col: str, mode: str):
    df = out.copy()

    key_cols = ["supplier", "supplier_code"]
    temp_key_name = None

    if supplier_col and supplier_col in df_source.columns:
        temp_key_name = "__dedupe_supplier_col__"
        df[temp_key_name] = df_source[supplier_col].astype(str).fillna("").str.strip()
        key_cols.append(temp_key_name)

    if not df.duplicated(subset=key_cols, keep=False).any():
        report = pd.DataFrame(
            columns=[
                "supplier",
                "supplier_code",
                (supplier_col if supplier_col else "group"),
                "duplicate_count",
                "rrp_values",
                "cost_values",
                "kept_rrp",
                "kept_cost",
                "kept_source_row",
                "flag",
            ]
        )
        if temp_key_name and temp_key_name in df.columns:
            df = df.drop(columns=[temp_key_name])
        return df, report

    df["__rrp_num__"] = pd.to_numeric(df["rrp_net"], errors="coerce").fillna(0.0).round(2)
    df["__cost_num__"] = pd.to_numeric(df["cost_net"], errors="coerce").fillna(0.0).round(2)

    report_rows = []
    for _, g in df.groupby(key_cols, dropna=False):
        if len(g) <= 1:
            continue

        rrps = sorted(g["__rrp_num__"].unique().tolist())
        costs = sorted(g["__cost_num__"].unique().tolist())
        price_conflict = (len(rrps) > 1) or (len(costs) > 1)

        if mode == "keep_first":
            kept_row = g.sort_index().iloc[0]
        else:
            kept_row = g.sort_values("__rrp_num__", ascending=False).iloc[0]

        flag = "DUPLICATE_CODE_DIFF_PRICE" if price_conflict else "DUPLICATE_CODE"
        group_val = kept_row.get(temp_key_name, "") if temp_key_name else ""

        report_rows.append(
            {
                "supplier": kept_row.get("supplier", ""),
                "supplier_code": kept_row.get("supplier_code", ""),
                (supplier_col if supplier_col else "group"): group_val,
                "duplicate_count": int(len(g)),
                "rrp_values": ", ".join([f"{x:.2f}" for x in rrps]),
                "cost_values": ", ".join([f"{x:.2f}" for x in costs]),
                "kept_rrp": float(kept_row["__rrp_num__"]),
                "kept_cost": float(kept_row["__cost_num__"]),
                "kept_source_row": int(kept_row.name) if str(kept_row.name).isdigit() else str(kept_row.name),
                "flag": flag,
            }
        )

    report_df = pd.DataFrame(report_rows)

    if mode == "keep_first":
        kept = df.sort_index().groupby(key_cols, as_index=False).head(1)
    else:
        kept = df.sort_values("__rrp_num__", ascending=False).groupby(key_cols, as_index=False).head(1)

    # Tag kept rows with conflict note
    conflict_keys = set(
        report_df.loc[report_df["flag"] == "DUPLICATE_CODE_DIFF_PRICE", ["supplier", "supplier_code"]]
        .apply(tuple, axis=1)
        .tolist()
    )

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
        axis=1,
    )

    drop_cols = ["__rrp_num__", "__cost_num__"]
    if temp_key_name and temp_key_name in kept.columns:
        drop_cols.append(temp_key_name)
    kept = kept.drop(columns=[c for c in drop_cols if c in kept.columns])

    return kept, report_df
