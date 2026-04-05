"""实验调度器。"""
from __future__ import annotations

from typing import Dict, List

from tqdm import tqdm

from .config import ExperimentConfig
from .llm import build_openai_client
from .results import ExperimentResultWriter
from .utils import list_json_files, load_json
from methods import create_method


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.client = build_openai_client(config)
        self.method = create_method(self.client, config)
        self.writer = ExperimentResultWriter(config)

    def _list_samples(self) -> List:
        if not self.config.data_dir.is_dir():
            raise RuntimeError(f"Data directory not found: {self.config.data_dir}")

        files = list_json_files(self.config.data_dir)
        if self.config.max_samples is not None:
            files = files[: max(0, int(self.config.max_samples))]
        if self.config.debug_mode:
            debug_limit = self.config.debug_limit or 3
            files = files[: max(0, int(debug_limit))]
        return files

    @staticmethod
    def _is_badcase(record: Dict[str, object]) -> bool:
        return not (record.get("acc_agent") and record.get("acc_step"))

    @staticmethod
    def _normalize_step(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(float(str(value).strip()))
        except Exception:
            return None

    def _resolve_step_tolerances(self) -> List[int]:
        params = self.config.method_params or {}
        explicit = params.get("step_tolerances")
        if isinstance(explicit, list):
            values = sorted({int(item) for item in explicit if isinstance(item, (int, float, str)) and str(item).strip().lstrip("-").isdigit()})
            return [value for value in values if value >= 0]

        max_tol_raw = params.get("step_tolerance_max", 5)
        try:
            max_tol = int(max_tol_raw)
        except Exception:
            max_tol = 5
        if max_tol < 0:
            return []
        return list(range(0, max_tol + 1))

    def run(self) -> Dict[str, object]:
        sample_paths = self._list_samples()
        total = len(sample_paths)
        if total == 0:
            raise RuntimeError(f"No samples found in {self.config.data_dir}")

        agent_correct = 0
        step_correct = 0
        badcase_files: List[str] = []
        tolerances = self._resolve_step_tolerances()
        tolerance_hits = {tol: 0 for tol in tolerances}

        for index, path in enumerate(tqdm(sample_paths, desc="Processing samples")):
            record = self.method.process_sample(path, index=index)
            self.writer.write_sample(record)

            agent_correct += int(bool(record.get("acc_agent")))
            step_correct += int(bool(record.get("acc_step")))
            gt_step = self._normalize_step((record.get("gt") or {}).get("step") if isinstance(record.get("gt"), dict) else None)
            pred_step = self._normalize_step((record.get("pred") or {}).get("step") if isinstance(record.get("pred"), dict) else None)
            if gt_step is not None and pred_step is not None:
                diff = abs(pred_step - gt_step)
                for tol in tolerances:
                    if diff <= tol:
                        tolerance_hits[tol] += 1

            if self._is_badcase(record):
                badcase_files.append(path.stem)
                self.writer.write_badcase(
                    path.stem,
                    {
                        "input": load_json(path),
                        "prediction": record,
                        "dataset": self.config.dataset_tag,
                    },
                )

        summary = {
            "method": self.config.method,
            "model": self.config.model,
            "dataset": self.config.dataset_tag,
            "total_samples": total,
            "agent_accuracy": agent_correct / total,
            "step_accuracy": step_correct / total,
            "step_accuracy_with_tolerance": {
                f"±{tol}": (tolerance_hits[tol] / total) for tol in tolerances
            },
            "prompt_tokens": self.method.prompt_tokens,
            "completion_tokens": self.method.completion_tokens,
            "total_tokens": self.method.prompt_tokens + self.method.completion_tokens,
            "badcase_count": len(badcase_files),
            "badcase_samples": badcase_files,
            "method_params": self.config.method_params,
        }
        self.writer.write_summary(summary)
        summary["per_sample_path"] = str(self.writer.paths.per_sample)
        summary["summary_path"] = str(self.writer.paths.summary)
        summary["badcase_dir"] = str(self.writer.paths.badcase_folder)
        return summary
