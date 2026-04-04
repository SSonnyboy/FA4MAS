"""通用工具函数。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_agent(value: Any) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def normalize_step(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def numeric_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(character for character in path.stem if character.isdigit())
    return (int(digits) if digits else 0, path.name)


def list_json_files(directory: Path) -> List[Path]:
    return sorted(directory.glob("*.json"), key=numeric_sort_key)


def join_lines(lines: Iterable[str]) -> str:
    return "\n".join(line for line in lines if line)

