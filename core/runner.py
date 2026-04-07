"""实验调度器。"""
from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from .config import ExperimentConfig
from .llm import build_openai_client
from .results import ExperimentResultWriter
from .utils import list_json_files, load_json, normalize_agent
from methods import create_method


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, *, resume: bool = False, run_id: str | None = None) -> None:
        self.config = config
        self.client = build_openai_client(config)
        self.method = create_method(self.client, config)
        self.writer = ExperimentResultWriter(config, resume=resume, run_id=run_id)
        self.resume_existing_prompt_tokens = 0
        self.resume_existing_completion_tokens = 0
        if self.writer.resume and self.writer.paths.summary.is_file():
            previous_summary = load_json(self.writer.paths.summary)
            self.resume_existing_prompt_tokens = int(previous_summary.get("prompt_tokens") or 0)
            self.resume_existing_completion_tokens = int(previous_summary.get("completion_tokens") or 0)

    def _list_samples(self) -> List:
        if not self.config.data_dir.is_dir():
            raise RuntimeError(f"Data directory not found: {self.config.data_dir}")

        files = list_json_files(self.config.data_dir)
        if self.config.max_samples is not None:
            files = files[: max(0, int(self.config.max_samples))]
        if self.config.debug_mode:
            debug_limit = self.config.debug_limit or 10
            limit = max(0, int(debug_limit))
            rng = random.Random()
            if limit >= len(files):
                files = list(files)
                rng.shuffle(files)
            else:
                files = rng.sample(files, limit)
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

    @staticmethod
    def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
        if not path.is_file():
            return []

        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except Exception:
                    tqdm.write(f"[warn] invalid json at {path}:{line_no}, ignored.")
                    continue
                if isinstance(row, dict):
                    records.append(row)
        return records

    @staticmethod
    def _collect_record_by_file(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for record in records:
            file_path = str(record.get("file", "")).strip()
            if not file_path:
                continue
            mapping[file_path] = record
        return mapping

    @staticmethod
    def _build_error_record(path, *, index: int, exc: Exception) -> Dict[str, Any]:
        sample = load_json(path)
        gt_agent_raw = sample.get("mistake_agent")
        gt_step_raw = sample.get("mistake_step")
        if (gt_agent_raw is None or gt_step_raw is None) and isinstance(sample.get("gt"), dict):
            gt_agent_raw = gt_agent_raw if gt_agent_raw is not None else sample["gt"].get("agent")
            gt_step_raw = gt_step_raw if gt_step_raw is not None else sample["gt"].get("step")

        return {
            "file": str(path),
            "question": str(sample.get("question", "")),
            "gt": {
                "agent": normalize_agent(gt_agent_raw),
                "step": ExperimentRunner._normalize_step(gt_step_raw),
            },
            "pred": {"agent": None, "step": None},
            "acc_agent": 0,
            "acc_step": 0,
            "final_pred": {"mistake_agent": None, "mistake_step": None, "reason": ""},
            "status": "error",
            "error": {
                "sample_index": index,
                "sample_id": path.stem,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().strip(),
            },
        }

    def _build_summary_from_records(
        self,
        *,
        sample_paths: List[Path],
        records: List[Dict[str, Any]],
    ) -> Dict[str, object]:
        total = len(sample_paths)
        tolerances = self._resolve_step_tolerances()
        tolerance_hits = {tol: 0 for tol in tolerances}

        record_by_file = self._collect_record_by_file(records)
        ordered_records = [record_by_file[str(path)] for path in sample_paths if str(path) in record_by_file]
        missing_paths = [str(path) for path in sample_paths if str(path) not in record_by_file]

        agent_correct = 0
        step_correct = 0
        badcase_files: List[str] = []
        failed_samples: List[Dict[str, str]] = []

        for record in ordered_records:
            agent_correct += int(bool(record.get("acc_agent")))
            step_correct += int(bool(record.get("acc_step")))
            gt_step = self._normalize_step((record.get("gt") or {}).get("step") if isinstance(record.get("gt"), dict) else None)
            pred_step = self._normalize_step((record.get("pred") or {}).get("step") if isinstance(record.get("pred"), dict) else None)
            if gt_step is not None and pred_step is not None:
                diff = abs(pred_step - gt_step)
                for tol in tolerances:
                    if diff <= tol:
                        tolerance_hits[tol] += 1

            record_file = str(record.get("file", "")).strip()
            if self._is_badcase(record):
                badcase_files.append(Path(record_file).stem if record_file else "")

            if str(record.get("status", "")).lower() == "error":
                error_info = record.get("error") if isinstance(record.get("error"), dict) else {}
                failed_samples.append(
                    {
                        "sample_id": str(error_info.get("sample_id") or (Path(record_file).stem if record_file else "")),
                        "file": record_file,
                        "error_type": str(error_info.get("type") or ""),
                        "error_message": str(error_info.get("message") or ""),
                    }
                )

        prompt_tokens = self.resume_existing_prompt_tokens + self.method.prompt_tokens
        completion_tokens = self.resume_existing_completion_tokens + self.method.completion_tokens

        summary = {
            "method": self.config.method,
            "model": self.config.model,
            "dataset": self.config.dataset_tag,
            "total_samples": total,
            "completed_samples": len(ordered_records),
            "missing_sample_count": len(missing_paths),
            "missing_samples": [Path(path).stem for path in missing_paths],
            "successful_samples": len(ordered_records) - len(failed_samples),
            "failed_sample_count": len(failed_samples),
            "failed_samples": failed_samples,
            "agent_accuracy": (agent_correct / total) if total else 0.0,
            "step_accuracy": (step_correct / total) if total else 0.0,
            "step_accuracy_with_tolerance": {
                f"±{tol}": ((tolerance_hits[tol] / total) if total else 0.0) for tol in tolerances
            },
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "badcase_count": len(badcase_files),
            "badcase_samples": badcase_files,
            "method_params": self.config.method_params,
        }
        return summary

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

        existing_records = self._load_jsonl_records(self.writer.paths.per_sample)
        existing_record_by_file = self._collect_record_by_file(existing_records)
        pending_items = [
            (index, path)
            for index, path in enumerate(sample_paths)
            if str(path) not in existing_record_by_file
        ]
        if self.writer.resume:
            tqdm.write(
                f"[info] resume run {self.writer.paths.timestamp}: "
                f"existing={len(existing_record_by_file)}, pending={len(pending_items)}, total={total}"
            )

        newly_processed = 0
        for index, path in tqdm(pending_items, desc="Processing samples", total=len(pending_items)):
            try:
                record = self.method.process_sample(path, index=index)
            except Exception as exc:
                record = self._build_error_record(path, index=index, exc=exc)
                tqdm.write(f"[warn] failed sample {path.stem}: {type(exc).__name__}: {exc}")

            self.writer.write_sample(record)
            newly_processed += 1

            if self._is_badcase(record):
                self.writer.write_badcase(
                    path.stem,
                    {
                        "input": load_json(path),
                        "prediction": record,
                        "dataset": self.config.dataset_tag,
                    },
                )

        all_records = self._load_jsonl_records(self.writer.paths.per_sample)
        summary = self._build_summary_from_records(sample_paths=sample_paths, records=all_records)
        summary["resumed"] = bool(self.writer.resume)
        summary["run_timestamp"] = self.writer.paths.timestamp
        summary["existing_samples_before_run"] = len(existing_record_by_file)
        summary["newly_processed_samples"] = newly_processed

        self.writer.write_summary(summary)
        summary["per_sample_path"] = str(self.writer.paths.per_sample)
        summary["summary_path"] = str(self.writer.paths.summary)
        summary["badcase_dir"] = str(self.writer.paths.badcase_folder)
        return summary
