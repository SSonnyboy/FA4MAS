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
    @staticmethod
    def _find_latest_run_id(experiment_root: Path) -> str | None:
        if not experiment_root.is_dir():
            return None
        candidates = []
        for item in experiment_root.iterdir():
            if not item.is_dir():
                continue
            if (item / "samples.jsonl").is_file():
                candidates.append(item.name)
        if not candidates:
            return None
        return max(candidates)

    def __init__(self, config: ExperimentConfig, *, resume: bool = False, run_id: str | None = None) -> None:
        experiment_root = config.results_dir / "experiments" / config.method / config.dataset_tag
        badcase_root = config.badcase_dir / config.method / config.dataset_tag
        ensure_dir(experiment_root)
        ensure_dir(badcase_root)

        if run_id:
            timestamp = run_id
        elif resume:
            latest = self._find_latest_run_id(experiment_root)
            if not latest:
                raise RuntimeError(
                    f"Resume requested but no existing run found under: {experiment_root}"
                )
            timestamp = latest
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        base_dir = experiment_root / timestamp
        badcase_dir = badcase_root / timestamp
        ensure_dir(base_dir)
        ensure_dir(badcase_dir)
        if resume and not (base_dir / "samples.jsonl").is_file():
            raise RuntimeError(
                f"Resume requested but samples.jsonl not found: {base_dir / 'samples.jsonl'}"
            )

        self.paths = ResultPaths(
            per_sample=base_dir / "samples.jsonl",
            summary=base_dir / "summary.json",
            badcase_folder=badcase_dir,
            timestamp=timestamp,
        )
        self.resume = resume

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
