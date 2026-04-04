"""Baseline 共享逻辑。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.llm import chat_completion
from core.utils import load_json, normalize_agent, normalize_step
from methods.base import BaseMethod


class BaselineMethodBase(BaseMethod):
    max_tokens: int | None = None

    def load_sample(self, file_path: Path) -> Dict[str, Any]:
        return load_json(file_path)

    @staticmethod
    def extract_ground_truth(sample: Dict[str, Any]) -> Tuple[str | None, int | None]:
        gt_agent = sample.get("mistake_agent")
        gt_step = sample.get("mistake_step")
        if (gt_agent is None or gt_step is None) and isinstance(sample.get("gt"), dict):
            gt_agent = gt_agent if gt_agent is not None else sample["gt"].get("agent")
            gt_step = gt_step if gt_step is not None else sample["gt"].get("step")
        return normalize_agent(gt_agent), normalize_step(gt_step)

    @staticmethod
    def extract_answer(sample: Dict[str, Any]) -> str:
        return str(sample.get("ground_truth") or sample.get("answer") or "")

    @staticmethod
    def resolve_agent_field(sample: Dict[str, Any]) -> str:
        history = sample.get("history") or []
        if not history:
            return "name"
        first_entry = history[0]
        if "role" in first_entry:
            return "role"
        if "name" in first_entry:
            return "name"
        return "agent"

    def call_model(self, prompt: str, *, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        return chat_completion(
            self.client,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            system_prompt=system_prompt or "You are a helpful assistant skilled in analyzing multi-agent conversations.",
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

    @staticmethod
    def parse_final_prediction(text: str) -> Dict[str, Any]:
        agent_match = re.search(r"Agent Name:\s*([^\n*]+)", text)
        step_match = re.search(r"Step Number:\s*([0-9]+)", text)
        reason_match = re.search(r"Reason for Mistake:\s*(.*)", text, re.DOTALL)
        return {
            "mistake_agent": agent_match.group(1).strip() if agent_match else None,
            "mistake_step": int(step_match.group(1)) if step_match else None,
            "reason": reason_match.group(1).strip() if reason_match else "",
        }

    @staticmethod
    def build_final_record(
        file_path: Path,
        question: str,
        gt_agent: str | None,
        gt_step: int | None,
        final_prediction: Dict[str, Any],
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        pred_agent = normalize_agent(final_prediction.get("mistake_agent"))
        pred_step = normalize_step(final_prediction.get("mistake_step"))
        acc_agent = int(pred_agent == gt_agent and pred_agent is not None)
        acc_step = int(pred_step == gt_step and pred_step is not None)
        record = {
            "file": str(file_path),
            "question": question,
            "gt": {"agent": gt_agent, "step": gt_step},
            "pred": {"agent": pred_agent, "step": pred_step},
            "acc_agent": acc_agent,
            "acc_step": acc_step,
            "final_pred": final_prediction,
        }
        if extra:
            record.update(extra)
        return record

    @staticmethod
    def render_history_as_dialogue(history: List[Dict[str, Any]], agent_field: str) -> str:
        lines = []
        for index, entry in enumerate(history):
            agent_name = entry.get(agent_field, "Unknown Agent")
            content = entry.get("content", "")
            lines.append(f"Step {index} - {agent_name}: {content}")
        return "\n".join(lines)

