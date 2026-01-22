# backend/core/price_detector.py
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd


@dataclass
class PriceDetection:
    rrp_col: Optional[str] = None
    rrp_is_gross: bool = False
    trade_col: Optional[str] = None  # optional true trade column if present
    trade_is_gross: bool = False
    currency: str = "GBP"
    vat_rate: float = 0.2
    notes: List[str] = None


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\(\)\[\]\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _is_gross(h: str) -> bool:
    return any(t in h for t in ["inc vat", "incl vat", "including vat", "gross"])


def _score_rrp(h: str) -> int:
    score = 0
    if any(t in h for t in ["rrp", "msrp", "list price", "retail price", "retail"]):
        score += 8
    if h.strip() == "price":
        score += 6
    if any(t in h for t in ["ex vat", "exvat", "net", "excluding vat"]):
        score += 4
    if _is_gross(h):
        score += 1  # still RRP, but gross
    if any(t in h for t in ["discount", "vat", "tax", "rate"]):
        score -= 4
    return score


def _score_trade(h: str) -> int:
    score = 0
    if any(t in h for t in ["trade", "net cost", "cost", "buy", "purchase price", "wholesale"]):
        score += 8
    if "price" in h:
        score += 1
    if any(t in h for t in ["ex vat", "exvat", "net", "excluding vat"]):
        score += 3
    if _is_gross(h):
        score += 1
    if any(t in h for t in ["rrp", "msrp", "list", "retail"]):
        score -= 4
    return score


def detect_prices(df: pd.DataFrame) -> PriceDetection:
    cols = list(df.columns)
    norm_cols = [(c, _norm(c)) for c in cols]
    notes: List[str] = []

    # Currency hint from headers
    header_blob = " | ".join(h for _, h in norm_cols)
    currency = "GBP"
    if "eur" in header_blob or "€" in header_blob:
        currency = "EUR"
    elif "usd" in header_blob or "$" in header_blob:
        currency = "USD"
    elif "gbp" in header_blob or "£" in header_blob:
        currency = "GBP"

    # VAT default (we’ll improve later by reading VAT columns)
    vat_rate = 0.2

    rrp_candidates: List[Tuple[int, str, str]] = []
    trade_candidates: List[Tuple[int, str, str]] = []

    for original, h in norm_cols:
        rrp_candidates.append((_score_rrp(h), original, h))
        trade_candidates.append((_score_trade(h), original, h))

    rrp_candidates.sort(reverse=True, key=lambda x: x[0])
    trade_candidates.sort(reverse=True, key=lambda x: x[0])

    rrp_col = rrp_candidates[0][1] if rrp_candidates and rrp_candidates[0][0] >= 8 else None
    trade_col = trade_candidates[0][1] if trade_candidates and trade_candidates[0][0] >= 8 else None

    if not rrp_col:
        notes.append("No confident RRP/List price column detected.")
    if trade_col:
        notes.append(f"Trade column detected: {trade_col}")

    rrp_is_gross = _is_gross(_norm(rrp_col)) if rrp_col else False
    trade_is_gross = _is_gross(_norm(trade_col)) if trade_col else False

    return PriceDetection(
        rrp_col=rrp_col,
        rrp_is_gross=rrp_is_gross,
        trade_col=trade_col,
        trade_is_gross=trade_is_gross,
        currency=currency,
        vat_rate=vat_rate,
        notes=notes,
    )
