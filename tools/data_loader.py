"""数据加载模块"""

import json
import os
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class EvalCase:
    """评估样本"""
    question_id: str
    question: str
    history: List
    ground_truth_answer: str
    ground_truth_agent: str
    ground_truth_step: int
    mistake_reason: str = ""


def load_eval_case(path: str) -> EvalCase:
    """从JSON文件加载单个评估样本"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    return EvalCase(
        question_id=d.get("question_ID", os.path.basename(path)),
        question=d.get("question", ""),
        history=d.get("history", []),
        ground_truth_answer=str(d.get("ground_truth", "")),
        ground_truth_agent=d.get("mistake_agent", "unknown"),
        ground_truth_step=int(d.get("mistake_step", 0)),
        mistake_reason=d.get("mistake_reason", ""),
    )


def load_dataset(data_dir: str, limit: Optional[int] = None) -> List[EvalCase]:
    """批量加载数据集"""
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if limit:
        files = files[:limit]

    cases = []
    for path in files:
        try:
            case = load_eval_case(path)
            cases.append(case)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    return cases
