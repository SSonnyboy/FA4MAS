# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/30 19:36 
'''

from typing import List, Dict
def evaluate(predictions: List[Dict], instances: List[Dict], tolerance: int = 0) -> Dict:
    agent_correct = 0
    step_correct  = 0
    total = len(predictions)

    for pred, inst in zip(predictions, instances):
        gt_agent = (inst.get("mistake_agent") or inst.get("responsible_agent")
                    or inst.get("failure_agent") or inst.get("gt_agent") or "")

        gt_step_raw = (inst.get("mistake_step") or inst.get("error_step")
                       or inst.get("decisive_step") or inst.get("gt_step"))
        try:
            gt_step = int(gt_step_raw) + 1  # 0-indexed → 1-indexed
        except (TypeError, ValueError):
            gt_step = -1

        pred_agent = pred.get("responsible_agent", "").strip().lower()
        if pred_agent and pred_agent == str(gt_agent).strip().lower():
            agent_correct += 1

        try:
            pred_step = int(pred.get("error_step", -1))
            if pred_step >= 0 and gt_step >= 0 and abs(pred_step - gt_step) <= tolerance:
                step_correct += 1
        except (ValueError, TypeError):
            pass

    return {
        "total":          total,
        "agent_accuracy": agent_correct / total if total else 0,
        "step_accuracy":  step_correct  / total if total else 0,
        "agent_correct":  agent_correct,
        "step_correct":   step_correct,
    }
