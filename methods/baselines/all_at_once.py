"""两阶段 all-at-once baseline。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from .common import BaselineMethodBase


class AllAtOnceBaselineMethod(BaselineMethodBase):
    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        # 保留旧实现能力：可以决定提示词里是否包含标准答案。
        self.use_ground_truth = bool(self.config.method_params.get("use_ground_truth_in_prompt", True))

    def generate_subtasks(self, history: List[Dict[str, Any]], question: str, ground_truth: str) -> tuple[list[Dict[str, str]], str]:
        # 先让模型把长轨迹压缩成子任务结构，减轻第二阶段定位难度。
        answer_block = f"The Answer for the problem is: {ground_truth}\n" if self.use_ground_truth else ""
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem.\n"
            f"The problem is: {question}\n"
            # f"The correct answer for the problem is: {ground_truth}\n\n"
            f"{answer_block}"
            "Here is the conversation in JSON format:\n"
            + str(history)
            + f"\n\nThere are total {len(history)} steps, each entry provides the input of the agent and its role.\n"
            "Based on this conversation, please:\n"
            "1. Decompose the reasoning into semantic subtasks.\n"
            "2. Predict the correct output (oracle) for each subtask.\n\n"
            "Please answer in the format below, strictly following the plain text structure:\n\n"
            "The Subtask Name: (your prediction)\n"
            "Step Range: (start-end)\n"
            "The Oracle: (your prediction)\n\n"
            "No overlaps between step ranges are allowed.\n"
            "Now generate your output below:\n"
        )
        raw = self.call_model(prompt)
        names = re.findall(r"The Subtask Name:\s*([^\n*]+)", raw)
        ranges = re.findall(r"Step Range:\s*([0-9]+-[0-9]+)", raw)
        oracles = re.findall(r"The Oracle:\s*([^\n*]+)", raw)
        subtasks = []
        # 这里只抽取结构化字段，保留原始文本供后续排查。
        for name, step_range, oracle in zip(names, ranges, oracles):
            subtasks.append(
                {
                    "Subtask Name": name.strip(),
                    "Step Range": step_range.strip(),
                    "The Oracle": oracle.strip(),
                }
            )
        return subtasks, raw

    def predict_error(
        self,
        history: List[Dict[str, Any]],
        question: str,
        ground_truth: str,
        subtasks: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        # 第二阶段在“子任务摘要 + 原始对话”上做错误归因。
        answer_block = f"The Answer for the problem is: {ground_truth}\n" if self.use_ground_truth else ""
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem.\n"
            f"The problem is: {question}\n"
            # f"The correct answer for the problem is: {ground_truth}\n\n"
            f"{answer_block}"
            "Here is the conversation (in JSON format):\n"
            + str(history)
            + f"\n\nThere are total {len(history)} steps, each entry provides an agent's input.\n"
            "Below are the subtasks, their step ranges, and the expected oracle outputs:\n"
            + str(subtasks)
            + "\n\nYour job:\n"
            "1. Identify which agent made a reasoning mistake directly responsible for the wrong final result.\n"
            "2. Determine the exact step number where this mistake first occurred.\n"
            "3. Explain why that step is the reasoning slip.\n\n"
            "Please answer in this exact plain text format:\n"
            "Agent Name: (your prediction)\n"
            "Step Number: (your prediction)\n"
            "Reason for Mistake: (your explanation)\n"
            "No special symbols, no extra commentary.\n"
        )
        raw = self.call_model(prompt)
        return {"final": self.parse_final_prediction(raw), "raw": raw}

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        # all-at-once 的核心是两跳推理：先分段，再定位。
        sample = self.load_sample(file_path)
        question = str(sample.get("question", ""))
        history = sample.get("history", [])
        ground_truth = self.extract_answer(sample)
        gt_agent, gt_step = self.extract_ground_truth(sample)

        subtasks, step1_raw = self.generate_subtasks(history, question, ground_truth)
        prediction = self.predict_error(history, question, ground_truth, subtasks)
        return self.build_final_record(
            file_path,
            question,
            gt_agent,
            gt_step,
            prediction["final"],
            {
                "step1_raw": step1_raw,
                "step2_raw": prediction["raw"],
                "subtasks": subtasks,
            },
        )
