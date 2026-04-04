"""全轨迹 baseline。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import BaselineMethodBase


class FullTrajectoryBaselineMethod(BaselineMethodBase):
    def predict(self, history: Any, question: str, ground_truth: str) -> Dict[str, Any]:
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation that failed to correctly solve a real-world problem.\n"
            f"The problem is: {question}\n"
            f"The correct answer for the problem should be: {ground_truth}\n\n"
            "Here is the full conversation log (in JSON format):\n"
            + str(history)
            + f"\n\nThere are total {len(history)} steps, each representing one agent's message.\n"
            "Your goal:\n"
            "1. Identify which agent made the reasoning mistake directly responsible for the incorrect outcome.\n"
            "2. Determine the exact step number where this mistake first occurred.\n"
            "3. Explain why this step represents a reasoning error.\n\n"
            "Please strictly follow this plain text format:\n"
            "Agent Name: (your prediction)\n"
            "Step Number: (your prediction)\n"
            "Reason for Mistake: (your explanation)\n"
            "No extra commentary or markdown symbols.\n"
        )
        raw = self.call_model(prompt)
        return {"final": self.parse_final_prediction(raw), "raw": raw}

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        sample = self.load_sample(file_path)
        question = str(sample.get("question", ""))
        history = sample.get("history", [])
        ground_truth = self.extract_answer(sample)
        gt_agent, gt_step = self.extract_ground_truth(sample)

        prediction = self.predict(history, question, ground_truth)
        return self.build_final_record(
            file_path,
            question,
            gt_agent,
            gt_step,
            prediction["final"],
            {"raw": prediction["raw"]},
        )

