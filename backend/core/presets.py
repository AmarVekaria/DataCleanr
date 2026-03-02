# backend/core/presets.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import yaml


def _project_root_from_here() -> Path:
    # core/presets.py -> backend/core -> backend -> project root
    return Path(__file__).resolve().parents[2]


def _presets_dir() -> Path:
    return _project_root_from_here() / "backend" / "presets" / "suppliers"


def _norm_supplier_name(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_").replace("-", "_")


def list_supplier_presets() -> List[str]:
    """
    Returns supplier preset keys (filename without .yaml), e.g. ["hansgrohe", "samuel_heath"]
    """
    d = _presets_dir()
    if not d.exists():
        return []
    items = []
    for p in d.glob("*.yaml"):
        items.append(p.stem)
    return sorted(items)


def load_supplier_preset(supplier: str) -> Dict[str, Any]:
    """
    Loads backend/presets/suppliers/<supplier>.yaml if it exists.
    Returns a dict:
      - supplier, supplier_key
      - mapping_overrides: dict (canonical_key -> supplier column header)
      - defaults: dict (code_length, merge_template, merge_fields, dedupe_mode, etc.)
    """
    supplier_key = _norm_supplier_name(supplier)
    if not supplier_key:
        return {}

    presets_path = _presets_dir() / f"{supplier_key}.yaml"
    if not presets_path.exists():
        return {}

    with open(presets_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    preset: Dict[str, Any] = {}
    preset["supplier"] = supplier
    preset["supplier_key"] = supplier_key
    preset["mapping_overrides"] = data.get("mapping_overrides", {}) or {}
    preset["defaults"] = data.get("defaults", {}) or {}
    return preset


def apply_preset_to_mapping(mapping: Dict[str, str], preset: Dict[str, Any]) -> Dict[str, str]:
    if not preset:
        return mapping
    overrides = preset.get("mapping_overrides") or {}
    if overrides:
        mapping.update(overrides)
    return mapping


def apply_preset_to_options(options: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    if not preset:
        return options

    defaults = preset.get("defaults") or {}

    def _set_if_missing(key: str, value: Any):
        if value is None:
            return
        if key not in options or options[key] in ("", None) or options[key] == 0:
            options[key] = value

    _set_if_missing("code_length", defaults.get("code_length", None))
    _set_if_missing("merge_template", defaults.get("merge_template", ""))
    _set_if_missing("merge_fields", defaults.get("merge_fields", ""))
    _set_if_missing("merge_dedupe", defaults.get("merge_dedupe", True))
    _set_if_missing("dedupe_mode", defaults.get("dedupe_mode", "keep_max_rrp"))
    _set_if_missing("dedupe_by_supplier_column", defaults.get("dedupe_by_supplier_column", ""))

    _set_if_missing("sheet_mode", defaults.get("sheet_mode", ""))
    _set_if_missing("sheet_name", defaults.get("sheet_name", ""))

    return options