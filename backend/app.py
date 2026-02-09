# backend/app.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
from io import BytesIO
import traceback
import re

from core.cleaner import clean_dataframe
from core.mappings import infer_column_mapping, get_supplier_override
from core.exporters import to_showroom_schema
from core.detectors import suggest_merge_candidates

app = FastAPI(title="DataCleanr MVP")

ALLOWED_NAME_PREFS = {"auto", "short_description", "product_title", "full_description"}
ALLOWED_DEDUPE_MODES = {"keep_max_rrp", "keep_first"}

PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")  # {Column Heading}


def _validate_name_preference(name_preference: str) -> str:
    pref = (name_preference or "auto").strip().lower()
    if pref not in ALLOWED_NAME_PREFS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid name_preference='{name_preference}'. "
                f"Allowed values: {sorted(ALLOWED_NAME_PREFS)}"
            ),
        )
    return pref


def _validate_dedupe_mode(dedupe_mode: str) -> str:
    mode = (dedupe_mode or "keep_max_rrp").strip().lower()
    if mode not in ALLOWED_DEDUPE_MODES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid dedupe_mode='{dedupe_mode}'. "
                f"Allowed values: {sorted(ALLOWED_DEDUPE_MODES)}"
            ),
        )
    return mode


async def _read_any(upload: UploadFile, sheet_name: str = "") -> pd.DataFrame:
    filename = (upload.filename or "").lower()

    if filename.endswith(".csv"):
        return pd.read_csv(upload.file)

    if sheet_name and sheet_name.strip():
        try:
            return pd.read_excel(upload.file, sheet_name=sheet_name.strip())
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Sheet '{sheet_name}' not found in '{upload.filename}'. "
                    f"Open the Excel file and confirm the tab name exactly.\n\nOriginal error: {e}"
                ),
            )

    return pd.read_excel(upload.file)


def _build_mapping(df: pd.DataFrame, supplier: str) -> dict:
    mapping = infer_column_mapping(df.columns)
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)
    return mapping


def _extract_template_columns(merge_template: str) -> list[str]:
    cols = [c.strip() for c in PLACEHOLDER_RE.findall(merge_template or "") if c.strip()]
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _to_float_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float64")
    x = s.astype(str).fillna("").str.strip()
    x = x.str.replace("£", "", regex=False).str.replace(",", "", regex=False)
    extracted = x.str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _build_duplicate_lookup_from_report(report_df: pd.DataFrame) -> dict:
    lookup = {}
    if report_df is None or report_df.empty:
        return lookup

    for _, r in report_df.iterrows():
        supplier = str(r.get("supplier", "")).strip()
        code = str(r.get("supplier_code", "")).strip()
        kept = r.get("kept_source_row", None)
        try:
            kept = int(kept)
        except Exception:
            kept = None
        if supplier and code:
            lookup[(supplier, code)] = kept
    return lookup


def _build_diff_sheet(
    df_raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    report_df: pd.DataFrame,
    mapping: dict,
    merge_template: str,
    supplier: str,
) -> pd.DataFrame:
    raw_code_col = mapping.get("supplier_code")
    raw_codes = df_raw[raw_code_col] if (raw_code_col and raw_code_col in df_raw.columns) else pd.Series([""] * len(df_raw))

    raw_codes_str = raw_codes.astype(str).fillna("").str.strip()
    raw_codes_str = raw_codes_str.str.replace(r"\.0$", "", regex=True)

    template_cols = _extract_template_columns(merge_template)
    raw_template_data = {}
    for c in template_cols:
        raw_template_data[f"raw::{c}"] = (
            df_raw[c].astype(str).fillna("").str.strip() if c in df_raw.columns else ""
        )

    raw_rrp_col = mapping.get("rrp_net")
    raw_cost_col = mapping.get("cost_net")

    raw_rrp = _to_float_series(df_raw[raw_rrp_col]) if raw_rrp_col in df_raw.columns else pd.Series([None] * len(df_raw))
    raw_cost = _to_float_series(df_raw[raw_cost_col]) if raw_cost_col in df_raw.columns else pd.Series([None] * len(df_raw))

    raw_frame = pd.DataFrame({
        "raw_row_index": df_raw.index.astype(int),
        "source_file": df_raw.get("source_file", ""),
        "source_sheet": df_raw.get("source_sheet", ""),
        "raw_supplier_code": raw_codes_str,
        **raw_template_data,
        "raw_rrp": raw_rrp.round(2),
        "raw_cost": raw_cost.round(2),
    })

    clean_frame = cleaned.copy()
    clean_frame["clean_supplier_code"] = clean_frame.get("supplier_code", "").astype(str).fillna("").str.strip()
    clean_frame["clean_catalogue_name"] = clean_frame.get("catalogue_name", "").astype(str).fillna("").str.strip()
    clean_frame["clean_rrp"] = pd.to_numeric(clean_frame.get("rrp_net", 0), errors="coerce").fillna(0).round(2)
    clean_frame["clean_cost"] = pd.to_numeric(clean_frame.get("cost_net", 0), errors="coerce").fillna(0).round(2)
    clean_frame["clean_notes"] = clean_frame.get("notes", "").astype(str).fillna("").str.strip()

    clean_frame = clean_frame[[
        "clean_supplier_code", "clean_catalogue_name", "clean_rrp", "clean_cost", "clean_notes"
    ]]

    df_join = raw_frame.merge(
        clean_frame,
        left_on="raw_supplier_code",
        right_on="clean_supplier_code",
        how="left",
    )

    dup_lookup = _build_duplicate_lookup_from_report(report_df)

    def _dup_status(row):
        code = str(row.get("raw_supplier_code", "")).strip()
        if not code:
            return "NO_CODE", None

        key = (supplier, code)
        kept_row = dup_lookup.get(key)

        if key in dup_lookup:
            if kept_row is None:
                return "DUPLICATE_UNKNOWN", None
            if int(row.get("raw_row_index")) == kept_row:
                return "KEPT", kept_row
            return "REMOVED", kept_row

        return "NO_DUPLICATE", None

    statuses = df_join.apply(lambda r: _dup_status(r), axis=1)
    df_join["duplicate_status"] = [s[0] for s in statuses]
    df_join["kept_source_row"] = [s[1] for s in statuses]

    def _changed_fields_row(r):
        changes = []
        if (r.get("raw_supplier_code") or "") != (r.get("clean_supplier_code") or "") and (r.get("clean_supplier_code") not in [None, ""]):
            changes.append("CODE_PADDED/CHANGED")

        if template_cols:
            raw_text = " | ".join([(r.get(f"raw::{c}") or "").strip() for c in template_cols]).strip()
            clean_text = (r.get("clean_catalogue_name") or "").strip()
            if raw_text and clean_text and raw_text.upper() != clean_text.upper():
                changes.append("DESC_BUILT/CHANGED")

        if pd.notna(r.get("raw_rrp")) and pd.notna(r.get("clean_rrp")):
            if float(r.get("raw_rrp") or 0) != float(r.get("clean_rrp") or 0):
                changes.append("RRP_CHANGED")

        if pd.notna(r.get("raw_cost")) and pd.notna(r.get("clean_cost")):
            if float(r.get("raw_cost") or 0) != float(r.get("clean_cost") or 0):
                changes.append("COST_CHANGED")

        return ", ".join(changes)

    df_join["changed_fields"] = df_join.apply(_changed_fields_row, axis=1)

    ordered_cols = [
        "raw_row_index", "source_file", "source_sheet", "raw_supplier_code",
        *[f"raw::{c}" for c in template_cols],
        "raw_rrp", "raw_cost",
        "clean_supplier_code", "clean_catalogue_name", "clean_rrp", "clean_cost",
        "duplicate_status", "kept_source_row",
        "changed_fields", "clean_notes"
    ]
    for c in ordered_cols:
        if c not in df_join.columns:
            df_join[c] = ""

    return df_join[ordered_cols]


# ---------------------------
# Preview (unchanged)
# ---------------------------
@app.post("/preview")
async def preview(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
    sheet_name: str = Form(""),
    name_preference: str = Form("auto"),
    code_length: int | None = Form(None),
    merge_fields: str = Form(""),
    merge_template: str = Form(""),
    merge_dedupe: bool = Form(True),
    dedupe_by_supplier_column: str = Form("Brand"),
    dedupe_mode: str = Form("keep_max_rrp"),
):
    try:
        pref = _validate_name_preference(name_preference)
        mode = _validate_dedupe_mode(dedupe_mode)

        df = await _read_any(file, sheet_name=sheet_name)
        mapping = _build_mapping(df, supplier)

        suggestions = suggest_merge_candidates(df)

        options = {
            "name_preference": pref,
            "code_length": code_length,
            "merge_fields": merge_fields,
            "merge_template": merge_template,
            "merge_dedupe": merge_dedupe,
            "dedupe_by_supplier_column": dedupe_by_supplier_column,
            "dedupe_mode": mode,
            "return_report": False,
        }

        cleaned = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

        return JSONResponse({
            "suggestions": suggestions,
            "preview_rows": cleaned.head(50).to_dict(orient="records"),
        })

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{e}\n\n{tb}")


# ---------------------------
# Single-file export (kept)
# ---------------------------
@app.post("/export")
async def export(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),
    sheet_name: str = Form(""),
    name_preference: str = Form("auto"),
    code_length: int | None = Form(None),
    merge_fields: str = Form(""),
    merge_template: str = Form(""),
    merge_dedupe: bool = Form(True),
    dedupe_by_supplier_column: str = Form("Brand"),
    dedupe_mode: str = Form("keep_max_rrp"),
    include_raw_sheet: bool = Form(False),
    include_diff_sheet: bool = Form(False),
):
    try:
        pref = _validate_name_preference(name_preference)
        mode = _validate_dedupe_mode(dedupe_mode)

        df = await _read_any(file, sheet_name=sheet_name)
        df = df.copy()
        df["source_file"] = file.filename or ""
        df["source_sheet"] = sheet_name or ""
        df.index = range(len(df))  # stabilise row ids

        mapping = _build_mapping(df, supplier)

        options = {
            "discount_percent": discount_percent,
            "name_preference": pref,
            "code_length": code_length,
            "merge_fields": merge_fields,
            "merge_template": merge_template,
            "merge_dedupe": merge_dedupe,
            "dedupe_by_supplier_column": dedupe_by_supplier_column,
            "dedupe_mode": mode,
            "return_report": True,
        }

        cleaned, report = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

        diff_df = None
        if include_diff_sheet:
            diff_df = _build_diff_sheet(df, cleaned, report, mapping, merge_template, supplier)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if include_raw_sheet:
                df.to_excel(writer, index=False, sheet_name="Raw")

            cleaned.to_excel(writer, index=False, sheet_name="Cleaned")
            report.to_excel(writer, index=False, sheet_name="Report")
            if diff_df is not None:
                diff_df.to_excel(writer, index=False, sheet_name="Diff")

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=cleaned_with_report.xlsx"},
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{e}\n\n{tb}")


# ---------------------------
# ✅ NEW: Combine 2 files -> Canonical Clean
# ---------------------------
@app.post("/export-combine")
async def export_combine(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),

    sheet_name1: str = Form(""),
    sheet_name2: str = Form(""),

    name_preference: str = Form("auto"),
    code_length: int | None = Form(None),

    merge_fields: str = Form(""),
    merge_template: str = Form(""),

    merge_dedupe: bool = Form(True),
    dedupe_by_supplier_column: str = Form("Brand"),
    dedupe_mode: str = Form("keep_max_rrp"),

    include_raw_sheet: bool = Form(True),   # for combine, default TRUE is helpful
    include_diff_sheet: bool = Form(True),  # for combine, default TRUE is helpful
):
    try:
        pref = _validate_name_preference(name_preference)
        mode = _validate_dedupe_mode(dedupe_mode)

        df1 = await _read_any(file1, sheet_name=sheet_name1)
        df2 = await _read_any(file2, sheet_name=sheet_name2)

        df1 = df1.copy()
        df2 = df2.copy()

        df1["source_file"] = file1.filename or "file1"
        df2["source_file"] = file2.filename or "file2"
        df1["source_sheet"] = sheet_name1 or ""
        df2["source_sheet"] = sheet_name2 or ""

        df = pd.concat([df1, df2], ignore_index=True)
        df.index = range(len(df))  # critical for duplicate kept_source_row mapping

        mapping = _build_mapping(df, supplier)

        options = {
            "discount_percent": discount_percent,
            "name_preference": pref,
            "code_length": code_length,
            "merge_fields": merge_fields,
            "merge_template": merge_template,
            "merge_dedupe": merge_dedupe,
            "dedupe_by_supplier_column": dedupe_by_supplier_column,
            "dedupe_mode": mode,
            "return_report": True,
        }

        cleaned, report = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

        diff_df = None
        if include_diff_sheet:
            diff_df = _build_diff_sheet(df, cleaned, report, mapping, merge_template, supplier)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if include_raw_sheet:
                df.to_excel(writer, index=False, sheet_name="RawCombined")

            cleaned.to_excel(writer, index=False, sheet_name="Cleaned")
            report.to_excel(writer, index=False, sheet_name="Report")

            if diff_df is not None:
                diff_df.to_excel(writer, index=False, sheet_name="Diff")

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=combined_cleaned_with_report.xlsx"},
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{e}\n\n{tb}")


# ---------------------------
# ✅ NEW: Combine 2 files -> Showroom/Intact Upload
# ---------------------------
@app.post("/export-showroom-combine")
async def export_showroom_combine(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),

    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),
    valid_from: str = Form(""),

    sheet_name1: str = Form(""),
    sheet_name2: str = Form(""),

    name_preference: str = Form("auto"),
    code_length: int | None = Form(None),

    merge_fields: str = Form(""),
    merge_template: str = Form(""),

    merge_dedupe: bool = Form(True),
    dedupe_by_supplier_column: str = Form("Brand"),
    dedupe_mode: str = Form("keep_max_rrp"),

    include_raw_sheet: bool = Form(True),
    include_diff_sheet: bool = Form(True),
):
    try:
        pref = _validate_name_preference(name_preference)
        mode = _validate_dedupe_mode(dedupe_mode)

        df1 = await _read_any(file1, sheet_name=sheet_name1)
        df2 = await _read_any(file2, sheet_name=sheet_name2)

        df1 = df1.copy()
        df2 = df2.copy()

        df1["source_file"] = file1.filename or "file1"
        df2["source_file"] = file2.filename or "file2"
        df1["source_sheet"] = sheet_name1 or ""
        df2["source_sheet"] = sheet_name2 or ""

        df = pd.concat([df1, df2], ignore_index=True)
        df.index = range(len(df))

        mapping = _build_mapping(df, supplier)

        options = {
            "discount_percent": discount_percent,
            "name_preference": pref,
            "code_length": code_length,
            "merge_fields": merge_fields,
            "merge_template": merge_template,
            "merge_dedupe": merge_dedupe,
            "dedupe_by_supplier_column": dedupe_by_supplier_column,
            "dedupe_mode": mode,
            "return_report": True,
        }

        canon, report = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

        upload_df = to_showroom_schema(
            canon,
            supplier=supplier,
            discount_percent=discount_percent,
            valid_from=valid_from,
        )

        diff_df = None
        if include_diff_sheet:
            diff_df = _build_diff_sheet(df, canon, report, mapping, merge_template, supplier)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if include_raw_sheet:
                df.to_excel(writer, index=False, sheet_name="RawCombined")

            upload_df.to_excel(writer, index=False, sheet_name="Upload")
            report.to_excel(writer, index=False, sheet_name="Report")

            if diff_df is not None:
                diff_df.to_excel(writer, index=False, sheet_name="Diff")

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=combined_showroom_upload_with_report.xlsx"},
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{e}\n\n{tb}")
