"""BLADE 提示词构建。"""
from __future__ import annotations

from typing import Any, Dict, List


def _ground_truth_block(ground_truth: str) -> str:
    text = str(ground_truth or "").strip()
    if not text:
        return ""
    return f"Reference Correct Task Answer: {text}\n"


def build_screening_system_prompt() -> str:
    return (
        "You are BLADE-Screener, a budget-aware analyst for multi-agent error attribution. "
        "Given compact event capsules, estimate likely root-cause steps from two views: "
        "forward-origin risk and backward-failure necessity. "
        "Return strict JSON in <json></json> only."
    )


def build_screening_prompt(
    *,
    question: str,
    final_answer: str,
    ground_truth: str,
    event_capsules: List[Dict[str, Any]],
    deterministic_forward: List[int],
    deterministic_backward: List[int],
    top_k: int,
) -> str:
    gt_block = _ground_truth_block(ground_truth)
    return (
        f"Problem: {question}\n"
        f"{gt_block}"
        f"Observed Final Answer from the trajectory: {final_answer}\n\n"
        "Event Capsules (ordered by step):\n"
        f"{event_capsules}\n\n"
        f"Deterministic forward hints: {deterministic_forward}\n"
        f"Deterministic backward hints: {deterministic_backward}\n\n"
        f"Select up to {top_k} steps for each direction.\n"
        "JSON schema:\n"
        "<json>\n"
        "{\n"
        '  "forward_candidates": [{"step": 0, "score": 0.0, "reason": "..." }],\n'
        '  "backward_candidates": [{"step": 0, "score": 0.0, "reason": "..." }],\n'
        '  "global_confidence": 0.0\n'
        "}\n"
        "</json>\n"
        "Rules: step must be integer, score/confidence in [0,1], no extra text."
    )


def build_tournament_system_prompt() -> str:
    return (
        "You are BLADE-Duel, a strict pairwise causal judge. "
        "Choose which candidate is the EARLIER root-cause step for the wrong final answer. "
        "Follow the requested output schema exactly. "
        "If asked to output one token, return only upper or lower."
    )


def build_tournament_prompt(
    *,
    question: str,
    final_answer: str,
    ground_truth: str,
    candidate_a_step: int,
    candidate_b_step: int,
    candidate_a_context: str,
    candidate_b_context: str,
) -> str:
    gt_block = _ground_truth_block(ground_truth)
    return (
        f"Problem: {question}\n"
        f"{gt_block}"
        f"Observed Final Answer: {final_answer}\n\n"
        f"Candidate A (step {candidate_a_step}) local context:\n{candidate_a_context}\n\n"
        f"Candidate B (step {candidate_b_step}) local context:\n{candidate_b_context}\n\n"
        "Output:\n"
        "<json>\n"
        "{\n"
        '  "winner": "A or B or tie",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "..."\n'
        "}\n"
        "</json>\n"
        "Tie means both have similar causal priority."
    )


def build_finalize_system_prompt() -> str:
    return (
        "You are BLADE-Finalizer. Infer the single most responsible agent and earliest mistake step from local evidence. "
        "Output plain text fields exactly."
    )


def build_finalize_prompt(
    *,
    question: str,
    final_answer: str,
    ground_truth: str,
    focused_step: int,
    local_context: str,
) -> str:
    gt_block = _ground_truth_block(ground_truth)
    return (
        f"Problem: {question}\n"
        f"{gt_block}"
        f"Observed Final Answer: {final_answer}\n"
        f"Focused Candidate Step: {focused_step}\n\n"
        f"Local Context:\n{local_context}\n\n"
        "Return exactly:\n"
        "Agent Name: ...\n"
        "Step Number: ...\n"
        "Reason for Mistake: ...\n"
        "Confidence: ...\n"
        "No markdown."
    )


def build_escalation_system_prompt(role: str) -> str:
    return (
        "You are BLADE-Escalation Analyst. "
        f"Role style: {role}. "
        "Re-check local attribution only. Return strict JSON in <json></json> only."
    )


def build_escalation_prompt(
    *,
    question: str,
    final_answer: str,
    ground_truth: str,
    local_context: str,
    seed_step: int,
) -> str:
    gt_block = _ground_truth_block(ground_truth)
    return (
        f"Problem: {question}\n"
        f"{gt_block}"
        f"Observed Final Answer: {final_answer}\n"
        f"Seed Mistake Step: {seed_step}\n\n"
        f"Local Context:\n{local_context}\n\n"
        "<json>\n"
        "{\n"
        '  "agent": "...",\n'
        '  "step": 0,\n'
        '  "confidence": 0.0,\n'
        '  "reason": "..."\n'
        "}\n"
        "</json>\n"
        "No extra text."
    )
