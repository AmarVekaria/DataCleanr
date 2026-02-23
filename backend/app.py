# backend/app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
from io import BytesIO

from core.cleaner import clean_dataframe
from core.mappings import infer_column_mapping, get_supplier_override, mapping_audit
from core.exporters import to_showroom_schema
from core.discounts import load_discount_rules

app = FastAPI(title="DataCleanr MVP")


def _apply_mapping_overrides(mapping: dict, supplier: str) -> dict:
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)
    return mapping


def _sheet_score(df_sheet: pd.DataFrame, supplier: str) -> tuple[int, dict]:
    """
    Scores a sheet for being a likely 'real product list' sheet.
    Returns (score, mapping_used).
    """
    if df_sheet is None or df_sheet.empty:
        return 0, {}

    mapping = infer_column_mapping(df_sheet.columns)
    mapping = _apply_mapping_overrides(mapping, supplier)

    score = 0

    # 1) Supplier code density
    sc_col = mapping.get("supplier_code")
    if sc_col and sc_col in df_sheet.columns:
        s = df_sheet[sc_col].astype(str).fillna("").str.strip()
        non_blank = (s != "") & (s.str.lower() != "nan")
        score += int(non_blank.sum()) * 3  # strongest signal

    # 2) RRP numeric density
    rrp_col = mapping.get("rrp_net")
    if rrp_col and rrp_col in df_sheet.columns:
        rrp_num = pd.to_numeric(df_sheet[rrp_col], errors="coerce")
        score += int(rrp_num.notna().sum()) * 2

    # 3) Name column density (helpful signal)
    name_col = mapping.get("name")
    if name_col and name_col in df_sheet.columns:
        n = df_sheet[name_col].astype(str).fillna("").str.strip()
        non_blank = (n != "") & (n.str.lower() != "nan")
        score += int(non_blank.sum()) * 1

    return score, mapping


async def _read_any(
    upload: UploadFile,
    sheet_mode: str = "first",   # first | all | named | auto
    sheet_name: str = "",
    supplier: str = "unknown",
) -> pd.DataFrame:
    filename = (upload.filename or "").lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(upload.file)
        df["__sheet_name"] = "CSV"
        return df

    sheet_mode = (sheet_mode or "first").strip().lower()

    # named
    if sheet_mode == "named" and sheet_name.strip():
        df = pd.read_excel(upload.file, sheet_name=sheet_name)
        df["__sheet_name"] = sheet_name
        return df

    # all
    if sheet_mode == "all":
        all_sheets = pd.read_excel(upload.file, sheet_name=None)
        frames = []
        for sname, sdf in all_sheets.items():
            if sdf is None or sdf.empty:
                continue
            sdf = sdf.copy()
            sdf["__sheet_name"] = sname
            frames.append(sdf)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # auto (pick best sheet, not concat)
    if sheet_mode == "auto":
        all_sheets = pd.read_excel(upload.file, sheet_name=None)
        best_name = None
        best_df = None
        best_score = -1

        for sname, sdf in all_sheets.items():
            if sdf is None or sdf.empty:
                continue
            score, _ = _sheet_score(sdf, supplier=supplier)
            if score > best_score:
                best_score = score
                best_name = sname
                best_df = sdf

        if best_df is None:
            return pd.DataFrame()

        best_df = best_df.copy()
        best_df["__sheet_name"] = str(best_name)
        return best_df

    # first (default)
    df = pd.read_excel(upload.file, sheet_name=0)
    df["__sheet_name"] = "Sheet1"
    return df


@app.post("/preview")
async def preview(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
    sheet_mode: str = Form("first"),
    sheet_name: str = Form(""),
):
    df = await _read_any(file, sheet_mode=sheet_mode, sheet_name=sheet_name, supplier=supplier)

    mapping = infer_column_mapping(df.columns)
    mapping = _apply_mapping_overrides(mapping, supplier)

    cleaned = clean_dataframe(df, supplier=supplier, mapping=mapping)
    return JSONResponse(cleaned.head(50).to_dict(orient="records"))


@app.post("/export")
async def export(
    file: UploadFile = File(...),
    discount_rules: UploadFile = File(None),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),

    sheet_mode: str = Form("first"),   # first | all | named | auto
    sheet_name: str = Form(""),

    code_length: int = Form(0),
    merge_fields: str = Form(""),
    merge_template: str = Form(""),
    merge_dedupe: bool = Form(True),
    dedupe_by_supplier_column: str = Form(""),
    dedupe_mode: str = Form("keep_max_rrp"),
):
    df = await _read_any(file, sheet_mode=sheet_mode, sheet_name=sheet_name, supplier=supplier)

    mapping = infer_column_mapping(df.columns)
    mapping = _apply_mapping_overrides(mapping, supplier)

    options = {
        "discount_percent": float(discount_percent or 0.0),
        "code_length": (int(code_length) if int(code_length) > 0 else None),
        "merge_fields": merge_fields,
        "merge_template": merge_template,
        "merge_dedupe": bool(merge_dedupe),
        "dedupe_by_supplier_column": dedupe_by_supplier_column,
        "dedupe_mode": dedupe_mode,
        "return_report": True,
    }

    used_rules = False
    if discount_rules is not None:
        rules_df = load_discount_rules(discount_rules.file)
        options["discount_rules_df"] = rules_df
        used_rules = True

    cleaned, report_df = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="Cleaned")

        # Mapping audit
        sheet_used = "Unknown"
        try:
            if "__sheet_name" in df.columns and len(df) > 0:
                sheet_used = str(df["__sheet_name"].iloc[0])
        except Exception:
            pass

        audit_df = mapping_audit(mapping, sheet_used)
        audit_df.to_excel(writer, index=False, sheet_name="MappingAudit")

        # Errors
        if isinstance(report_df, pd.DataFrame) and not report_df.empty:
            report_df.to_excel(writer, index=False, sheet_name="Errors")

        # Discount debug
        if used_rules:
            debug_cols = ["supplier", "supplier_code", "rrp_net", "cost_net", "category", "catalogue_name", "notes"]
            debug_cols = [c for c in debug_cols if c in cleaned.columns]
            cleaned[debug_cols].to_excel(writer, index=False, sheet_name="DiscountDebug")

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=cleaned.xlsx"},
    )


@app.post("/export-showroom")
async def export_showroom(
    file: UploadFile = File(...),
    discount_rules: UploadFile = File(None),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),
    valid_from: str = Form(""),

    sheet_mode: str = Form("first"),   # first | all | named | auto
    sheet_name: str = Form(""),

    code_length: int = Form(0),
    merge_fields: str = Form(""),
    merge_template: str = Form(""),
    merge_dedupe: bool = Form(True),
    dedupe_by_supplier_column: str = Form(""),
    dedupe_mode: str = Form("keep_max_rrp"),
):
    df = await _read_any(file, sheet_mode=sheet_mode, sheet_name=sheet_name, supplier=supplier)

    mapping = infer_column_mapping(df.columns)
    mapping = _apply_mapping_overrides(mapping, supplier)

    options = {
        "discount_percent": float(discount_percent or 0.0),
        "code_length": (int(code_length) if int(code_length) > 0 else None),
        "merge_fields": merge_fields,
        "merge_template": merge_template,
        "merge_dedupe": bool(merge_dedupe),
        "dedupe_by_supplier_column": dedupe_by_supplier_column,
        "dedupe_mode": dedupe_mode,
        "return_report": True,
    }

    if discount_rules is not None:
        rules_df = load_discount_rules(discount_rules.file)
        options["discount_rules_df"] = rules_df

    canon, report_df = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

    showroom_df = to_showroom_schema(
        canon,
        supplier=supplier,
        discount_percent=float(discount_percent or 0.0),
        valid_from=valid_from,
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        showroom_df.to_excel(writer, index=False, sheet_name="ShowroomUpload")

        sheet_used = "Unknown"
        try:
            if "__sheet_name" in df.columns and len(df) > 0:
                sheet_used = str(df["__sheet_name"].iloc[0])
        except Exception:
            pass

        audit_df = mapping_audit(mapping, sheet_used)
        audit_df.to_excel(writer, index=False, sheet_name="MappingAudit")

        if isinstance(report_df, pd.DataFrame) and not report_df.empty:
            report_df.to_excel(writer, index=False, sheet_name="Errors")

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=showroom_upload.xlsx"},
    )