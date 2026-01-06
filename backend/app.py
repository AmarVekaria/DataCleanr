# backend/app.py

from typing import override
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
from io import BytesIO

from core.cleaner import clean_dataframe
from core.mappings import infer_column_mapping, get_supplier_override

app = FastAPI(title="DataCleanr MVP")


@app.post("/preview")
async def preview(
    file: UploadFile = File(...),
    supplier: str = Form("unknown")
):
    df = await _read_any(file)
    mapping = infer_column_mapping(df.columns)
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)
    cleaned = clean_dataframe(df, supplier=supplier, mapping=mapping)
    return JSONResponse(cleaned.head(50).to_dict(orient="records"))


@app.post("/export")
async def export(
    file: UploadFile = File(...),
    supplier: str = Form("unknown")
):
    df = await _read_any(file)
    mapping = infer_column_mapping(df.columns)
    override = get_supplier_override(supplier)
    if override:
        mapping.update(override)
    cleaned = clean_dataframe(df, supplier=supplier, mapping=mapping)

    output = BytesIO()
    cleaned.to_excel(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=cleaned.xlsx"},
    )


async def _read_any(upload: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into a pandas DataFrame."""
    filename = upload.filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(upload.file)
    return pd.read_excel(upload.file)
