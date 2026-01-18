# backend/core/exporters.py

import pandas as pd

SHOWROOM_COLUMNS = [
    "Code",
    "D_ValidFromDate",
    "ManufacturerProductCode",
    "Description",
    "StockingStatus",
    "StockingUnits",
    "Manufacturer",
    "Purchasing.DefaultSupplier",
    "Purchasing.SupplierProductCode",
    "Purchasing.PurchaseUnits",
    "Purchasing.ListPriceActual",
    "Purchasing.DiscountLevel1",
    "Purchasing.DiscountLevel2",
    "Purchasing.DiscountLevel3",
    "Purchasing.DiscountLevel4",
    "Purchasing.TradePriceActual",
    "Purchasing.ListPriceCurrency",
    "Purchasing.DefaultTaxRate",
    "Costings.StandardCost",
    "SellingPriceFactor1",
    "SellingPrice1",
    "Selling.SellingUnits",
    "SellingPriceDescription1",
    "SellingPriceCurrency1",
    "Selling.DefaultTaxRate",
    "SellingOptions.ProductSalesCostEditingControl",
    "Category",
    "Selling.MinimumMargin",
    "SellingPriceStartingPrice1",
    "SellingPriceCalculationMethod1",
    "ValuationMethod",
]


def to_showroom_schema(
    canon: pd.DataFrame,
    supplier: str,
    discount_percent: float,
    valid_from: str = "",
) -> pd.DataFrame:
    """
    Convert canonical DataCleanr frame into The Showroom Ltd / Intact IQ schema.
    - Uses canon['supplier_code'], ['catalogue_name'], ['rrp_net'], ['cost_net'], ['vat_rate'], ['category'], ['uom']
    - discount_percent is a single discount level (e.g. 40 for 40%)
    """

    df = canon.copy()

    # Ensure numeric/rounded RRP and cost
    rrp = pd.to_numeric(df.get("rrp_net"), errors="coerce").fillna(0).round(2)
    cost = pd.to_numeric(df.get("cost_net"), errors="coerce").fillna(0).round(2)

    # We already applied discount in Stage 1 to get cost_net.
    # Here we just carry that forward.
    trade = cost

    # VAT as percent (Intact usually expects 20 for 20%)
    vat_rate = pd.to_numeric(df.get("vat_rate", 0.2), errors="coerce").fillna(0.2)
    vat_percent = (vat_rate * 100).round(2)

    # Build output with required columns
    out = pd.DataFrame(index=df.index, columns=SHOWROOM_COLUMNS)

    # --- Core mappings ---
    out["Code"] = ""  # internal Showroom product code – you will fill or generate later
    out["D_ValidFromDate"] = valid_from  # e.g. "2026-01-01" or left blank

    out["ManufacturerProductCode"] = df.get("supplier_code", "")

    # Use catalogue_name if present, else fall back to name
    if "catalogue_name" in df.columns:
        desc_series = df["catalogue_name"]
    else:
        desc_series = df.get("name", "")
    out["Description"] = desc_series

    out["StockingStatus"] = "Non-Stock"
    out["StockingUnits"] = df.get("uom", "each")

    out["Manufacturer"] = supplier
    out["Purchasing.DefaultSupplier"] = supplier

    out["Purchasing.SupplierProductCode"] = df.get("supplier_code", "")
    out["Purchasing.PurchaseUnits"] = df.get("uom", "each")

    out["Purchasing.ListPriceActual"] = rrp
    out["Purchasing.DiscountLevel1"] = discount_percent  # keep as 40, 50 etc.
    out["Purchasing.DiscountLevel2"] = 0
    out["Purchasing.DiscountLevel3"] = 0
    out["Purchasing.DiscountLevel4"] = 0
    out["Purchasing.TradePriceActual"] = trade
    out["Purchasing.ListPriceCurrency"] = "GBP"
    out["Purchasing.DefaultTaxRate"] = vat_percent

    out["Costings.StandardCost"] = trade

    # Selling prices – simple starter logic: SellingPrice1 = RRP
    out["SellingPriceFactor1"] = 1.0
    out["SellingPrice1"] = rrp
    out["Selling.SellingUnits"] = df.get("uom", "each")
    out["SellingPriceDescription1"] = "RRP"
    out["SellingPriceCurrency1"] = "GBP"
    out["Selling.DefaultTaxRate"] = vat_percent

    # Defaults / placeholders – can be refined later
    out["SellingOptions.ProductSalesCostEditingControl"] = ""
    out["Category"] = df.get("category", "")
    out["Selling.MinimumMargin"] = ""
    out["SellingPriceStartingPrice1"] = trade
    out["SellingPriceCalculationMethod1"] = ""
    out["ValuationMethod"] = ""

    return out
