"""实验调度器。"""
from __future__ import annotations

from typing import Dict, List

from .config import ExperimentConfig
from .llm import build_openai_client
from .results import ExperimentResultWriter
from .utils import list_json_files, load_json
from methods import create_method


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.client = build_openai_client()
        self.method = create_method(self.client, config)
        self.writer = ExperimentResultWriter(config)

    def _list_samples(self) -> List:
        if not self.config.data_dir.is_dir():
            raise RuntimeError(f"Data directory not found: {self.config.data_dir}")

        files = list_json_files(self.config.data_dir)
        if self.config.max_samples is not None:
            files = files[: max(0, int(self.config.max_samples))]
        if self.config.debug_mode:
            debug_limit = self.config.debug_limit or 1
            files = files[: max(0, int(debug_limit))]
        return files

    @staticmethod
    def _is_badcase(record: Dict[str, object]) -> bool:
        return not (record.get("acc_agent") and record.get("acc_step"))

    def run(self) -> Dict[str, object]:
        sample_paths = self._list_samples()
        total = len(sample_paths)
        if total == 0:
            raise RuntimeError(f"No samples found in {self.config.data_dir}")

        agent_correct = 0
        step_correct = 0
        badcase_files: List[str] = []

        for index, path in enumerate(sample_paths):
            record = self.method.process_sample(path, index=index)
            self.writer.write_sample(record)

            agent_correct += int(bool(record.get("acc_agent")))
            step_correct += int(bool(record.get("acc_step")))

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
            "badcase_count": len(badcase_files),
            "badcase_samples": badcase_files,
            "method_params": self.config.method_params,
        }
        self.writer.write_summary(summary)
        summary["per_sample_path"] = str(self.writer.paths.per_sample)
        summary["summary_path"] = str(self.writer.paths.summary)
        summary["badcase_dir"] = str(self.writer.paths.badcase_folder)
        return summary

