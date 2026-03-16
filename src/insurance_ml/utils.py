from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pick_col(columns: list[str], candidates: list[str], required: bool = False) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    if required:
        raise ValueError(f"Missing required column among: {candidates}")
    return None


def safe_join_unique(values, unknown: str = "Unknown") -> str:
    vals = [str(x).strip() for x in values.dropna().astype(str) if str(x).strip()]
    vals = [v for v in vals if v.lower() != "nan"]
    uniq = sorted(set(vals))
    return "|".join(uniq) if uniq else unknown


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
