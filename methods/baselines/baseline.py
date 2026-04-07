"""全轨迹 baseline。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import BaselineMethodBase


class FullTrajectoryBaselineMethod(BaselineMethodBase):
    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        # 保留旧实现能力：可以决定提示词里是否包含标准答案。
        self.use_ground_truth = bool(self.config.method_params.get("use_ground_truth_in_prompt", True))
        
    def predict(self, history: Any, question: str, ground_truth: str) -> Dict[str, Any]:
        # 最直接的 baseline：把完整轨迹一次性给模型定位责任点。
        answer_block = f"The Answer for the problem is: {ground_truth}\n" if self.use_ground_truth else ""
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation that failed to correctly solve a real-world problem.\n"
            f"The problem is: {question}\n"
            # f"The correct answer for the problem should be: {ground_truth}\n\n"
            f"{answer_block}"
            "Here is the full conversation log (in JSON format):\n"
            + str(history)
            + f"\n\nThere are total {len(history)} steps, each representing one agent's message.\n"
            "Step indexing rule: all step numbers are 0-based (the first step is 0).\n"
            "Your goal:\n"
            "1. Identify which agent made the reasoning mistake directly responsible for the incorrect outcome.\n"
            "2. Determine the exact step number where this mistake first occurred.\n"
            "3. Explain why this step represents a reasoning error.\n\n"
            "Please strictly follow this plain text format:\n"
            "Agent Name: (your prediction)\n"
            "Step Number: (your prediction, 0-based)\n"
            "Reason for Mistake: (your explanation)\n"
            "No extra commentary or markdown symbols.\n"
        )
        raw = self.call_model(prompt)
        return {"final": self.parse_final_prediction(raw), "raw": raw}

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        # 单样本处理流程保持极简：读数据 -> 调模型 -> 组装统一结果。
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
