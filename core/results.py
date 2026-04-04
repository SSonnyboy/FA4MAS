"""结果写出逻辑。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

from .config import ExperimentConfig
from .utils import append_jsonl, ensure_dir, write_json


@dataclass
class ResultPaths:
    per_sample: Path
    summary: Path
    badcase_folder: Path
    timestamp: str


class ExperimentResultWriter:
    def __init__(self, config: ExperimentConfig) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = config.results_dir / "experiments" / config.method / config.dataset_tag / timestamp
        badcase_dir = config.badcase_dir / config.method / config.dataset_tag / timestamp
        ensure_dir(base_dir)
        ensure_dir(badcase_dir)

        self.paths = ResultPaths(
            per_sample=base_dir / "samples.jsonl",
            summary=base_dir / "summary.json",
            badcase_folder=badcase_dir,
            timestamp=timestamp,
        )

    def write_sample(self, record: Dict[str, object]) -> None:
        append_jsonl(self.paths.per_sample, record)

    def write_summary(self, summary: Dict[str, object]) -> None:
        write_json(
            self.paths.summary,
            {
                **summary,
                "per_sample_path": str(self.paths.per_sample),
                "badcase_folder": str(self.paths.badcase_folder),
                "timestamp": self.paths.timestamp,
            },
        )

    def write_badcase(self, file_name: str, payload: Dict[str, object]) -> None:
        write_json(self.paths.badcase_folder / f"{file_name}.json", payload)

