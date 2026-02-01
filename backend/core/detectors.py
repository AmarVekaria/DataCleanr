# backend/core/detectors.py
from __future__ import annotations

import re
import pandas as pd

STOPLIKE = {"picture", "drawing", "image", "url", "pdf"}
CODELIKE = {"code", "sku", "ean", "barcode", "gtin", "mpn", "productcode", "suppliercode"}
PRICELIKE = {"price", "rrp", "cost", "trade", "net", "gross", "vat"}
WEIGHTLIKE = {"weight", "kg", "g"}
DIMLIKE = {"length", "width", "height", "depth", "mm", "cm", "m"}
UNITLIKE = {"unit", "uom", "pack", "qty", "quantity", "volume"}

def suggest_merge_candidates(df: pd.DataFrame) -> dict:
    """
    Returns:
      {
        "recommended_title_col": str|None,
        "recommended_finish_col": str|None,
        "top_title_candidates": [str,...],
        "top_finish_candidates": [str,...],
        "templates": {
            "title_only": "{Col}",
            "title_plus_finish": "{Title} - '{Finish}'"
        }
      }
    """
    cols = list(df.columns)

    # Score columns for "title/description"
    title_scores = []
    finish_scores = []

    for c in cols:
        c_norm = _norm(c)

        # skip obvious non-description columns
        if _contains_any(c_norm, STOPLIKE | CODELIKE | PRICELIKE | WEIGHTLIKE | DIMLIKE | UNITLIKE):
            continue

        s = _safe_series(df[c])

        # must be text-ish and not mostly empty
        text_ratio = _text_ratio(s)
        if text_ratio < 0.35:
            continue

        non_empty = _non_empty_ratio(s)
        avg_len = _avg_len(s)
        uniq_ratio = _unique_ratio(s)

        # title columns are usually:
        # - high non-empty
        # - moderate avg length (not too short like "Chrome", not too long like a full paragraph)
        # - decent uniqueness
        score = 0.0
        score += non_empty * 2.0
        score += min(avg_len / 40.0, 2.0)              # reward length up to ~80 chars
        score += min(uniq_ratio * 2.0, 2.0)
        score += text_ratio

        # boost for common words
        if _contains_any(c_norm, {"description", "desc", "producttitle", "product title", "title", "short description"}):
            score += 2.5
        if "short" in c_norm and "description" in c_norm:
            score += 1.5

        title_scores.append((c, score))

        # finish/colour candidates (separate scoring)
        fscore = 0.0
        if _contains_any(c_norm, {"colour", "color", "finish", "surface"}):
            fscore += 5.0
        # finish columns tend to be short values and often lower uniqueness than titles
        if avg_len <= 25:
            fscore += 1.0
        if non_empty >= 0.3:
            fscore += 1.0
        finish_scores.append((c, fscore))

    title_scores.sort(key=lambda x: x[1], reverse=True)
    finish_scores.sort(key=lambda x: x[1], reverse=True)

    top_titles = [c for c, _ in title_scores[:6]]
    top_finishes = [c for c, s in finish_scores if s > 0][:6]

    recommended_title = top_titles[0] if top_titles else None
    recommended_finish = top_finishes[0] if top_finishes else None

    templates = {}
    if recommended_title:
        templates["title_only"] = f"{{{recommended_title}}}"
    if recommended_title and recommended_finish and recommended_title != recommended_finish:
        templates["title_plus_finish"] = f"{{{recommended_title}}} - '{{{recommended_finish}}}'"

    return {
        "recommended_title_col": recommended_title,
        "recommended_finish_col": recommended_finish,
        "top_title_candidates": top_titles,
        "top_finish_candidates": top_finishes,
        "templates": templates,
    }


# ---------------- helpers ----------------
def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip().lower())

def _contains_any(text: str, keys: set[str]) -> bool:
    return any(k in text for k in keys)

def _safe_series(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").map(lambda v: "" if str(v).lower() == "nan" else str(v).strip())

def _non_empty_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return float((s != "").mean())

def _avg_len(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    non_empty = s[s != ""]
    if len(non_empty) == 0:
        return 0.0
    return float(non_empty.map(len).mean())

def _unique_ratio(s: pd.Series) -> float:
    non_empty = s[s != ""]
    if len(non_empty) == 0:
        return 0.0
    return float(non_empty.nunique() / max(len(non_empty), 1))

def _text_ratio(s: pd.Series) -> float:
    """
    A rough check that values look text-ish (contain letters, spaces, etc),
    not mostly numbers.
    """
    if len(s) == 0:
        return 0.0
    non_empty = s[s != ""]
    if len(non_empty) == 0:
        return 0.0
    def is_texty(v: str) -> bool:
        return bool(re.search(r"[A-Za-z]", v))
    return float(non_empty.map(is_texty).mean())
