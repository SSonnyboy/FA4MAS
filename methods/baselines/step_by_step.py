"""逐步审查 baseline。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from .common import BaselineMethodBase


YES_RE = re.compile(r"^\s*(?:1\.)?\s*yes\b", re.IGNORECASE)
NO_RE = re.compile(r"^\s*(?:1\.)?\s*no\b", re.IGNORECASE)


class StepByStepBaselineMethod(BaselineMethodBase):
    max_tokens = 1024

    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        self.use_ground_truth = bool(self.params.get("use_ground_truth_in_prompt", True))

    @staticmethod
    def parse_verdict(text: str) -> str:
        if YES_RE.search(text or ""):
            return "yes"
        if NO_RE.search(text or ""):
            return "no"
        return "unknown"

    def build_prompt(
        self,
        *,
        question: str,
        ground_truth: str,
        conversation_so_far: str,
        step_index: int,
        agent_name: str,
    ) -> str:
        answer_block = f"The Answer for the problem is: {ground_truth}\n" if self.use_ground_truth else ""
        return (
            "You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent "
            "conversation aimed at solving a real-world problem.\n"
            f"The problem being addressed is: {question}\n"
            f"{answer_block}"
            f"Here is the conversation history up to the current step:\n{conversation_so_far}\n"
            f"The most recent step ({step_index}) was by '{agent_name}'.\n"
            "Determine whether this most recent step contains an error that could derail the process.\n"
            "Respond ONLY in the format:\n"
            "1. Yes/No.\n"
            "2. Reason: [Your explanation here]\n"
        )

    def inspect_steps(
        self,
        *,
        history: List[Dict[str, Any]],
        question: str,
        ground_truth: str,
        agent_field: str,
    ) -> tuple[str | None, int | None, list[Dict[str, Any]]]:
        if not history:
            return None, None, []

        logs = []
        conversation_so_far = ""
        for index, entry in enumerate(history):
            agent_name = entry.get(agent_field, "Unknown Agent")
            conversation_so_far += f"Step {index} - {agent_name}: {entry.get('content', '')}\n"
            raw = self.call_model(
                self.build_prompt(
                    question=question,
                    ground_truth=ground_truth,
                    conversation_so_far=conversation_so_far,
                    step_index=index,
                    agent_name=agent_name,
                ),
                system_prompt="You are a precise step-by-step conversation evaluator.",
                max_tokens=self.max_tokens,
            )
            verdict = self.parse_verdict(raw)
            logs.append({"step": index, "agent": agent_name, "verdict": verdict, "raw": raw})
            if verdict == "yes":
                return agent_name, index, logs

        last_index = len(history) - 1
        return history[last_index].get(agent_field, "Unknown Agent"), last_index, logs

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        sample = self.load_sample(file_path)
        question = str(sample.get("question", ""))
        history = sample.get("history", [])
        ground_truth = self.extract_answer(sample)
        gt_agent, gt_step = self.extract_ground_truth(sample)
        agent_field = self.resolve_agent_field(sample)

        pred_agent, pred_step, logs = self.inspect_steps(
            history=history,
            question=question,
            ground_truth=ground_truth,
            agent_field=agent_field,
        )
        return self.build_final_record(
            file_path,
            question,
            gt_agent,
            gt_step,
            {
                "mistake_agent": pred_agent,
                "mistake_step": pred_step,
                "reason": "Predicted by step-by-step inspection of the dialogue.",
            },
            {"step_logs": logs},
        )

