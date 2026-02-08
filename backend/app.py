# backend/app.py

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
from io import BytesIO
import traceback

from core.cleaner import clean_dataframe
from core.mappings import infer_column_mapping, get_supplier_override
from core.exporters import to_showroom_schema
from core.detectors import suggest_merge_candidates

app = FastAPI(title="DataCleanr MVP")

ALLOWED_NAME_PREFS = {"auto", "short_description", "product_title", "full_description"}
ALLOWED_DEDUPE_MODES = {"keep_max_rrp", "keep_first"}


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
    """
    JSON preview only (no download). Returns:
      - suggestions (recommended merge template)
      - preview_rows (first 50 cleaned rows)
    """
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

    # NEW: include original supplier data as Raw sheet
    include_raw_sheet: bool = Form(False),
):
    """
    Downloads cleaned Excel:
      - Raw (optional)
      - Cleaned
      - Report (duplicates audit)
    """
    try:
        pref = _validate_name_preference(name_preference)
        mode = _validate_dedupe_mode(dedupe_mode)

        df = await _read_any(file, sheet_name=sheet_name)
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

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if include_raw_sheet:
                # keep Raw identical to supplier upload (no transformation)
                df.to_excel(writer, index=False, sheet_name="Raw")

            cleaned.to_excel(writer, index=False, sheet_name="Cleaned")
            report.to_excel(writer, index=False, sheet_name="Report")

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


@app.post("/export-showroom")
async def export_showroom(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),
    valid_from: str = Form(""),
    sheet_name: str = Form(""),
    name_preference: str = Form("auto"),
    code_length: int | None = Form(None),

    merge_fields: str = Form(""),
    merge_template: str = Form(""),
    merge_dedupe: bool = Form(True),

    dedupe_by_supplier_column: str = Form("Brand"),
    dedupe_mode: str = Form("keep_max_rrp"),

    # NEW: include original supplier data as Raw sheet
    include_raw_sheet: bool = Form(False),
):
    """
    Downloads Showroom/Intact upload Excel:
      - Raw (optional)
      - Upload (schema)
      - Report (duplicates audit)
    """
    try:
        pref = _validate_name_preference(name_preference)
        mode = _validate_dedupe_mode(dedupe_mode)

        df = await _read_any(file, sheet_name=sheet_name)
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

        showroom_df = to_showroom_schema(
            canon,
            supplier=supplier,
            discount_percent=discount_percent,
            valid_from=valid_from,
        )

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if include_raw_sheet:
                df.to_excel(writer, index=False, sheet_name="Raw")

            showroom_df.to_excel(writer, index=False, sheet_name="Upload")
            report.to_excel(writer, index=False, sheet_name="Report")

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=showroom_upload_with_report.xlsx"},
        )

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"{e}\n\n{tb}")
