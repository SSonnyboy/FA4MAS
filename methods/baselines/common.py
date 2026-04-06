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
        # 两类数据集的 agent 字段不一致，这里统一做兼容。
        history = sample.get("history") or []
        if not history:
            return "name"
        first_entry = history[0]
        # 优先使用 name（通常是专家名），避免把 assistant/user 当成责任 agent。
        if "name" in first_entry and str(first_entry.get("name", "")).strip():
            return "name"
        if "role" in first_entry and str(first_entry.get("role", "")).strip():
            return "role"
        if "name" in first_entry:
            return "name"
        return "agent"

    def call_model(self, prompt: str, *, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        result = chat_completion(
            self.client,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            system_prompt=system_prompt or "You are a helpful assistant skilled in analyzing multi-agent conversations.",
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        return result.content

    @staticmethod
    def parse_final_prediction(text: str) -> Dict[str, Any]:
        # 所有 baseline 都复用同一套最终输出格式。
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
        # 在统一结果结构里补齐预测、标注和准确率，便于 runner 直接写出。
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
        # 某些 baseline 更适合读取线性对话文本，而不是原始 JSON。
        lines = []
        for index, entry in enumerate(history):
            agent_name = entry.get(agent_field, "Unknown Agent")
            content = entry.get("content", "")
            lines.append(f"Step {index} - {agent_name}: {content}")
        return "\n".join(lines)
