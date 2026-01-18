# backend/app.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
from io import BytesIO

from core.cleaner import clean_dataframe
from core.mappings import infer_column_mapping, get_supplier_override
from core.exporters import to_showroom_schema

app = FastAPI(title="DataCleanr MVP")


# ---------------------------
# Helpers
# ---------------------------
async def _read_any(upload: UploadFile) -> pd.DataFrame:
    if upload.filename.lower().endswith(".csv"):
        return pd.read_csv(upload.file)
    return pd.read_excel(upload.file)


# ---------------------------
# Preview endpoint
# ---------------------------
@app.post("/preview")
async def preview(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
):
    df = await _read_any(file)
    mapping = infer_column_mapping(df.columns)
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)

    cleaned = clean_dataframe(df, supplier=supplier, mapping=mapping)
    return JSONResponse(cleaned.head(50).to_dict(orient="records"))


# ---------------------------
# Canonical export
# ---------------------------
@app.post("/export")
async def export(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),
):
    df = await _read_any(file)
    mapping = infer_column_mapping(df.columns)
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)

    options = {"discount_percent": discount_percent}
    cleaned = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

    output = BytesIO()
    cleaned.to_excel(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=cleaned.xlsx"},
    )


# ---------------------------
# Showroom / Intact export
# ---------------------------
@app.post("/export-showroom")
async def export_showroom(
    file: UploadFile = File(...),
    supplier: str = Form("unknown"),
    discount_percent: float = Form(0.0),
    valid_from: str = Form(""),
):
    df = await _read_any(file)

    # Stage 1 – canonical clean
    mapping = infer_column_mapping(df.columns)
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)

    options = {"discount_percent": discount_percent}
    canon = clean_dataframe(df, supplier=supplier, mapping=mapping, options=options)

    # Stage 2 – Showroom schema
    showroom_df = to_showroom_schema(
        canon,
        supplier=supplier,
        discount_percent=discount_percent,
        valid_from=valid_from,
    )

    output = BytesIO()
    showroom_df.to_excel(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=showroom_upload.xlsx"},
    )

async def _read_any(upload: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into a pandas DataFrame."""
    filename = upload.filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(upload.file)
    return pd.read_excel(upload.file)
