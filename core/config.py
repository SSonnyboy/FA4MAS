"""实验配置定义与加载。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .utils import load_json


@dataclass
class ExperimentConfig:
    method: str
    model: str
    data_dir: Path
    results_dir: Path
    badcase_dir: Path
    api_key: str | None = None
    base_url: str | None = None
    debug_mode: bool = False
    debug_limit: int | None = None
    max_samples: int | None = None
    temperature: float = 0.0
    method_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def dataset_tag(self) -> str:
        name = self.data_dir.name.lower()
        if "algorithm" in name or "generated" in name:
            return "AG"
        if "hand" in name or "crafted" in name:
            return "HC"
        return self.data_dir.name


def _resolve_path(base_dir: Path, raw_value: str | Path, default_value: str) -> Path:
    value = raw_value or default_value
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = path.expanduser().resolve()
    raw = load_json(config_path)
    base_dir = config_path.parent

    return ExperimentConfig(
        method=str(raw["method"]),
        model=str(raw["model"]),
        data_dir=_resolve_path(base_dir, raw["data_dir"], "data/Algorithm-Generated"),
        results_dir=_resolve_path(base_dir, raw.get("results_dir"), "../results"),
        badcase_dir=_resolve_path(base_dir, raw.get("badcase_dir"), "../results/badcases"),
        api_key=raw.get("api_key"),
        base_url=raw.get("base_url"),
        debug_mode=bool(raw.get("debug_mode", False)),
        debug_limit=raw.get("debug_limit"),
        max_samples=raw.get("max_samples"),
        temperature=float(raw.get("temperature", 0.0)),
        method_params=dict(raw.get("method_params") or {}),
    )

