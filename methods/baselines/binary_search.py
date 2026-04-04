"""二分定位 baseline。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .common import BaselineMethodBase


class BinarySearchBaselineMethod(BaselineMethodBase):
    max_tokens = 1024

    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        self.use_ground_truth = bool(self.params.get("use_ground_truth_in_prompt", True))

    def build_prompt(
        self,
        *,
        question: str,
        answer: str,
        segment_text: str,
        range_description: str,
        upper_half_desc: str,
        lower_half_desc: str,
    ) -> str:
        answer_block = f"The Answer for the problem is: {answer}\n" if self.use_ground_truth else ""
        return (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are "
            "collaborating to address a user query.\n"
            f"The problem to address is as follows: {question}\n"
            f"{answer_block}"
            f"Review the following conversation segment {range_description}:\n\n{segment_text}\n\n"
            f"Predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) "
            f"or the lower half ({lower_half_desc}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'."
        )

    @staticmethod
    def parse_half(text: str) -> str:
        lowered = text.strip().lower()
        if "upper half" in lowered:
            return "upper"
        if "lower half" in lowered:
            return "lower"
        return "unknown"

    def run_binary_search(
        self,
        *,
        history: List[Dict[str, Any]],
        question: str,
        answer: str,
        agent_field: str,
    ) -> tuple[str | None, int | None, list[Dict[str, Any]]]:
        if not history:
            return None, None, []

        rounds: list[Dict[str, Any]] = []
        start = 0
        end = len(history) - 1

        while True:
            if start > end:
                chosen = max(0, min(end, len(history) - 1))
                agent = history[chosen].get(agent_field, "Unknown Agent")
                rounds.append({"range": [start, end], "mid": None, "decision": "invalid_range_fallback", "raw": ""})
                return agent, chosen, rounds

            if start == end:
                agent = history[start].get(agent_field, "Unknown Agent")
                rounds.append({"range": [start, end], "mid": start, "decision": "done", "raw": ""})
                return agent, start, rounds

            mid = start + (end - start) // 2
            prompt = self.build_prompt(
                question=question,
                answer=answer,
                segment_text=self.render_history_as_dialogue(history[start : end + 1], agent_field),
                range_description=f"from step {start} to step {end}",
                upper_half_desc=f"from step {start} to step {mid}",
                lower_half_desc=f"from step {mid + 1} to step {end}",
            )
            raw = self.call_model(
                prompt,
                system_prompt="You are an AI assistant specializing in localizing errors in conversation segments.",
                max_tokens=self.max_tokens,
            )
            decision = self.parse_half(raw)
            rounds.append(
                {
                    "range": [start, end],
                    "mid": mid,
                    "upper": [start, mid],
                    "lower": [mid + 1, end],
                    "decision": decision,
                    "raw": raw,
                }
            )

            if decision == "upper":
                end = mid
            elif decision == "lower":
                start = min(mid + 1, end)
            else:
                chosen = start
                agent = history[chosen].get(agent_field, "Unknown Agent")
                rounds.append({"range": [start, end], "mid": mid, "decision": "unknown_fallback", "raw": raw})
                return agent, chosen, rounds

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        sample = self.load_sample(file_path)
        question = str(sample.get("question", ""))
        history = sample.get("history", [])
        ground_truth = self.extract_answer(sample)
        gt_agent, gt_step = self.extract_ground_truth(sample)
        agent_field = self.resolve_agent_field(sample)

        pred_agent, pred_step, rounds = self.run_binary_search(
            history=history,
            question=question,
            answer=ground_truth,
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
                "reason": "Predicted by iterative binary search over the dialogue.",
            },
            {"rounds": rounds},
        )

