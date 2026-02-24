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

PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")  # {Column Heading}

SEPARATOR_RE = re.compile(r"^[-=_\s]{3,}$")


def clean_dataframe(df: pd.DataFrame, supplier: str, mapping: dict, options: dict | None = None):
    """
    Canonical clean.

    If options['return_report'] is True:
        returns (cleaned_df, report_df)
    else:
        returns cleaned_df
    """
    options = options or {}
    return_report = bool(options.get("return_report", False))

    discount_percent = float(options.get("discount_percent", 0.0) or 0.0)
    discount_rules_df = options.get("discount_rules_df", None)

    code_length = options.get("code_length", None)

    merge_fields_raw = (options.get("merge_fields") or "").strip()
    merge_template = (options.get("merge_template") or "").strip()
    merge_dedupe = bool(options.get("merge_dedupe", True))

    dedupe_by_supplier_column = (options.get("dedupe_by_supplier_column") or "").strip()
    dedupe_mode = (options.get("dedupe_mode") or "keep_max_rrp").strip().lower()

    # Build output dataframe
    out = pd.DataFrame(index=df.index, columns=CANONICAL_ORDER)
    for k, v in DEFAULT_ROW.items():
        out[k] = v

    def _get_series(canon_key: str, default=""):
        src = mapping.get(canon_key)
        if src and src in df.columns:
            return df[src]
        return pd.Series([default] * len(df), index=df.index)

    # ---------------------------
    # Report helper
    # ---------------------------
    report_rows = []
    supplier_code_raw = _get_series("supplier_code", "")

    def _add_report(idx, issue: str, severity: str = "ERROR"):
        row = {
            "source_row": int(idx) if str(idx).isdigit() else str(idx),
            "severity": severity,
            "issue": issue,
            "supplier_code_raw": str(supplier_code_raw.loc[idx]) if idx in supplier_code_raw.index else "",
            "supplier_code_clean": str(out.loc[idx, "supplier_code"]) if idx in out.index else "",
            "name": str(out.loc[idx, "name"]) if idx in out.index else "",
            "rrp_net": float(out.loc[idx, "rrp_net"]) if idx in out.index else 0.0,
            "cost_net": float(out.loc[idx, "cost_net"]) if idx in out.index else 0.0,
        }
        report_rows.append(row)

    # ---------------------------
    # Basics
    # ---------------------------
    out["supplier"] = supplier

    out["supplier_code"] = supplier_code_raw.map(lambda x: _clean_code(x, code_length))

    raw_name = _get_series("name", "")
    raw_finish = _get_series("finish", "")

    name_clean, dims = zip(*raw_name.map(_extract_dimensions_safe))
    out["name"] = list(name_clean)
    out["dimensions_mm"] = list(dims)

    out["finish"] = raw_finish.map(_normalise_finish)

    # ---------------------------
    # Prices (RRP + Cost)
    # ---------------------------
    rrp = _get_series("rrp_net", 0).map(to_float).fillna(0.0).round(2)
    cost = _get_series("cost_net", 0).map(to_float).fillna(0.0).round(2)

    out["rrp_net"] = rrp
    out["cost_net"] = cost

    # VAT
    vat = _get_series("vat_rate", 0.2)
    out["vat_rate"] = vat.map(_parse_vat).fillna(0.2)

    # Other fields
    out["barcode"] = _get_series("barcode", "").astype(str).fillna("").str.strip()
    out["uom"] = _get_series("uom", "each").astype(str).fillna("each").str.strip()
    out["category"] = _get_series("category", "").astype(str).fillna("").str.strip()

    # ---------------------------
    # catalogue_name build
    # ---------------------------
    out["catalogue_name"] = _build_catalogue_name(
        df=df,
        out=out,
        merge_template=merge_template,
        merge_fields_raw=merge_fields_raw,
        merge_dedupe=merge_dedupe,
    )

    # ---------------------------
    # EXCLUDE non-product rows (blank codes + section headers + separators)
    # ---------------------------
    code_series = out["supplier_code"].astype(str).fillna("").str.strip()
    rrp_series = pd.to_numeric(out["rrp_net"], errors="coerce").fillna(0.0)

    blank_code_mask = code_series.eq("") | code_series.str.lower().eq("nan")

    # Section header heuristic:
    # - looks like words (few digits) OR separator line
    # - and has no real price (rrp == 0) to avoid excluding legit codes
    section_like_mask = code_series.map(_looks_like_section_header) & (rrp_series == 0)

    exclude_mask = blank_code_mask | section_like_mask

    if exclude_mask.any():
        for idx in out.index[exclude_mask]:
            if blank_code_mask.loc[idx]:
                _add_report(idx, "Excluded row: missing supplier_code (likely break row)", "INFO")
            else:
                _add_report(idx, "Excluded row: section header/separator (non-product row)", "INFO")

    out = out[~exclude_mask].copy()

    # Keep df subset aligned for discount rules matching if needed
    df_for_rules = df.loc[out.index] if len(out.index) else df.iloc[0:0]

    # ---------------------------
    # Cost calculation logic
    # ---------------------------
    if len(out.index):
        cost_missing = (out["cost_net"].fillna(0) == 0)

        if discount_rules_df is not None:
            from .discounts import resolve_discount_with_debug

            computed_costs = []
            notes = []

            for idx, row in out.iterrows():
                rrp_val = float(row.get("rrp_net", 0) or 0)

                if not cost_missing.loc[idx]:
                    computed_costs.append(float(row.get("cost_net", 0) or 0))
                    notes.append(str(row.get("notes", "") or ""))
                    continue

                disc, rule_text = resolve_discount_with_debug(
                    _row_for_rule_matching(df_for_rules, idx, row),
                    discount_rules_df,
                    discount_percent,
                )

                cost_val = round(rrp_val * (1 - disc / 100.0), 2)
                computed_costs.append(cost_val)

                existing_note = str(row.get("notes", "") or "").strip()
                note = rule_text
                notes.append(note if not existing_note else (existing_note + " | " + note))

            out["cost_net"] = pd.Series(computed_costs, index=out.index).round(2)
            out["notes"] = pd.Series(notes, index=out.index)

        else:
            if discount_percent > 0 and cost_missing.any():
                out.loc[cost_missing, "cost_net"] = (
                    out.loc[cost_missing, "rrp_net"] * (1 - discount_percent / 100.0)
                ).round(2)

    # Missing RRP warnings
    if len(out.index):
        missing_rrp = (pd.to_numeric(out["rrp_net"], errors="coerce").fillna(0) == 0)
        if missing_rrp.any():
            for idx in out.index[missing_rrp]:
                _add_report(idx, "RRP not detected (rrp_net = 0). Check mapping/header detection.", "WARN")

    # ---------------------------
    # Deduplicate duplicate supplier codes (optional)
    # ---------------------------
    if len(out.index):
        out = _dedupe_supplier_codes(
            out=out,
            df_source=df.loc[out.index],
            supplier_col=dedupe_by_supplier_column,
            mode=dedupe_mode,
            report_cb=_add_report,
        )

    cleaned = out[CANONICAL_ORDER].reset_index(drop=True)
    report_df = pd.DataFrame(report_rows)

    if return_report:
        return cleaned, report_df
    return cleaned


# ---------------------------
# Helpers
# ---------------------------
def _row_for_rule_matching(df_source: pd.DataFrame, idx, out_row: pd.Series) -> pd.Series:
    data = {}
    try:
        src_row = df_source.loc[idx]
        if isinstance(src_row, pd.Series):
            for k, v in src_row.items():
                data[k] = v
    except Exception:
        pass

    for k, v in out_row.items():
        data[k] = v

    return pd.Series(data)


def _clean_code(x, code_length=None):
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


def _looks_like_section_header(code: str) -> bool:
    """
    Generic 'section header' detector for supplier_code column.
    Designed to be safe for real alphanumeric product codes.
    """
    s = (code or "").strip()
    if not s:
        return True

    if SEPARATOR_RE.match(s):
        return True

    # Too many spaces for a code (often phrases like "TOWEL RAILS")
    if s.count(" ") >= 2 and len(s) <= 40:
        # If it has very few digits, it's likely a header
        digits = sum(ch.isdigit() for ch in s)
        if digits <= 1:
            return True

    # Mostly letters (headers) and very few digits
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)

    if letters >= 4 and digits == 0 and len(s) <= 35:
        # common for ALL CAPS headers
        return True

    # single-word header like "SHOWERS" / "BRASSWARE"
    if " " not in s and digits == 0 and letters >= 5 and len(s) <= 25:
        # Avoid excluding real codes like "AXOR" (short brand-like codes)
        # If it's all caps and longer, it's very likely a header.
        if s.isupper() and len(s) >= 6:
            return True

    return False


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
        if v > 1:
            return v / 100.0
        return v
    except Exception:
        return 0.2


def _build_catalogue_name(df, out, merge_template, merge_fields_raw, merge_dedupe) -> pd.Series:
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
            rendered = merge_template
            raw_vals = []
            for col in placeholders:
                v = (series_map[col].iloc[i] or "").strip()
                if v and v.lower() != "nan":
                    raw_vals.append(v)
                rendered = rendered.replace("{" + col + "}", v if v.lower() != "nan" else "")

            if merge_dedupe and len(raw_vals) >= 2:
                raw_vals = _dedupe_parts(raw_vals)

            rendered = _cleanup_optional_template(rendered)
            cleaned = rendered.strip().strip("- ,|")
            if not cleaned:
                cleaned = " - ".join(raw_vals) if raw_vals else ""
            merged.append(cleaned)

        return pd.Series(merged, index=df.index).str.upper()

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


def _cleanup_optional_template(s: str) -> str:
    s = re.sub(r"\s*-\s*''\s*$", "", s)
    s = re.sub(r"\s*-\s*\"\"\s*$", "", s)
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = s.rstrip(" -,")
    s = re.sub(r"\s*-\s*['\"]\s*$", "", s).strip()
    return s


def _dedupe_parts(vals: list[str]) -> list[str]:
    cleaned = []
    for v in vals:
        v = (v or "").strip()
        if not v:
            continue
        if any(v.lower() == c.lower() for c in cleaned):
            continue
        cleaned.append(v)
    return cleaned


def _dedupe_supplier_codes(out: pd.DataFrame, df_source: pd.DataFrame, supplier_col: str, mode: str, report_cb=None) -> pd.DataFrame:
    df = out.copy()

    key_cols = ["supplier", "supplier_code"]
    temp_key_name = None

    if supplier_col and supplier_col in df_source.columns:
        temp_key_name = "__dedupe_supplier_col__"
        df[temp_key_name] = df_source[supplier_col].astype(str).fillna("").str.strip()
        key_cols.append(temp_key_name)

    if not df.duplicated(subset=key_cols, keep=False).any():
        if temp_key_name and temp_key_name in df.columns:
            df = df.drop(columns=[temp_key_name])
        return df

    df["__rrp_num__"] = pd.to_numeric(df["rrp_net"], errors="coerce").fillna(0.0).round(2)
    df["__cost_num__"] = pd.to_numeric(df["cost_net"], errors="coerce").fillna(0.0).round(2)

    grouped = df.groupby(key_cols, dropna=False)
    for _, g in grouped:
        if len(g) <= 1:
            continue
        rrps = g["__rrp_num__"].unique().tolist()
        costs = g["__cost_num__"].unique().tolist()
        if len(rrps) > 1 or len(costs) > 1:
            if report_cb is not None:
                report_cb(g.index[0], "Duplicate supplier_code with different prices. Kept highest RRP row.", "WARN")

    if mode == "keep_first":
        kept = df.sort_index().groupby(key_cols, as_index=False).head(1)
    else:
        kept = df.sort_values("__rrp_num__", ascending=False).groupby(key_cols, as_index=False).head(1)

    kept["notes"] = kept["notes"].astype(str).fillna("").apply(
        lambda n: n if "DUPLICATE_CODE" in n.upper() else (n + " | DUPLICATE_CODE") if n else "DUPLICATE_CODE"
    )

    drop_cols = ["__rrp_num__", "__cost_num__"]
    if temp_key_name and temp_key_name in kept.columns:
        drop_cols.append(temp_key_name)
    kept = kept.drop(columns=[c for c in drop_cols if c in kept.columns])

    return kept