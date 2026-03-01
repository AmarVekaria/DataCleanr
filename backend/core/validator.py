# backend/core/validator.py

import pandas as pd


def validate_cleaned_dataframe(df: pd.DataFrame):
    """
    Validates the canonical cleaned dataframe BEFORE showroom export.
    Returns:
        review_df (row-by-row issues)
        summary_df (counts)
    """
    issues = []

    # Track duplicates (only non-blank)
    codes = df.get("supplier_code", pd.Series([], dtype=str)).astype(str).str.strip()
    codes = codes[codes != ""]
    duplicate_codes = set(codes[codes.duplicated()])

    for idx, row in df.iterrows():
        code = str(row.get("supplier_code", "")).strip()
        desc = str(row.get("catalogue_name", "") or row.get("name", "")).strip()

        rrp = row.get("rrp_net", 0)
        cost = row.get("cost_net", 0)
        vat = row.get("vat_rate", 0.2)

        row_issues = []

        if code == "":
            row_issues.append("Missing Manufacturer Code")

        if desc == "":
            row_issues.append("Missing Description")

        # RRP validation
        try:
            rrp_val = float(rrp) if rrp is not None else 0.0
        except Exception:
            rrp_val = 0.0
        if rrp_val <= 0:
            row_issues.append("Missing or zero RRP")

        # Duplicates
        if code and code in duplicate_codes:
            row_issues.append("Duplicate Manufacturer Code")

        # VAT validation (canonical should be 0.2 etc.)
        try:
            vat_val = float(vat)
            if vat_val <= 0 or vat_val > 1:
                row_issues.append("Invalid VAT rate")
        except Exception:
            row_issues.append("Invalid VAT rate")

        # Cost > RRP
        try:
            cost_val = float(cost) if cost is not None else 0.0
            if rrp_val > 0 and cost_val > rrp_val:
                row_issues.append("Cost greater than RRP")
        except Exception:
            pass

        if row_issues:
            issues.append({
                "source_row": int(idx) if str(idx).isdigit() else str(idx),
                "supplier_code": code,
                "description": desc,
                "rrp_net": rrp_val,
                "cost_net": cost,
                "issues": " | ".join(row_issues),
            })

    review_df = pd.DataFrame(issues)

    summary = {
        "Total Products": int(len(df)),
        "Rows With Issues": int(len(review_df)),
        "Clean Rows": int(len(df) - len(review_df)),
        "Duplicate Codes": int(len(duplicate_codes)),
        "Ready For Review": "YES" if len(review_df) == 0 else "CHECK REQUIRED",
    }
    summary_df = pd.DataFrame([summary])

    return review_df, summary_df