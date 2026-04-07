"""ECHO 提示词构建。"""
from __future__ import annotations

from typing import Dict


ANALYST_FOCUS_MAP: Dict[str, str] = {
    "conservative": (
        "You are conservative with high confidence thresholds. "
        "Only attribute errors with strong and explicit evidence."
    ),
    "liberal": (
        "You are liberal and can attribute subtle and multi-agent errors "
        "when evidence is moderate but coherent."
    ),
    "detail_focused": (
        "You focus on exact wording, concrete evidence, and fine-grained logical inconsistencies."
    ),
    "pattern_focused": (
        "You focus on systemic patterns, long-range propagation, and recurring reasoning flaws."
    ),
    "skeptical": (
        "You challenge assumptions and consider alternative explanations before attributing errors."
    ),
    "general": "You maintain a balanced and objective perspective across all evidence types.",
}


def build_objective_system_prompt(analyst_focus: str, *, attribution_target: str = "joint") -> str:
    focus = ANALYST_FOCUS_MAP.get(analyst_focus, ANALYST_FOCUS_MAP["general"])
    target = str(attribution_target or "joint").lower()
    if target == "agent":
        target_instructions = (
            "PRIMARY GOAL: identify the most responsible agent(s). "
            "Use mistake_step only as supporting context."
        )
    elif target == "step":
        target_instructions = (
            "PRIMARY GOAL: identify the earliest responsible mistake step. "
            "Attribution is secondary support."
        )
    else:
        target_instructions = "PRIMARY GOAL: jointly identify responsible agent(s) and mistake step."
    return (
        "You are an Objective Analysis Agent for multi-agent error attribution.\n"
        f"ANALYST SPECIALIZATION: {focus}\n"
        f"{target_instructions}\n"
        "You must analyze ALL agents and identify who/when caused the wrong final answer.\n"
        "Output JSON wrapped in <json></json> using this schema:\n"
        "<json>\n"
        "{\n"
        '  "analysis_summary": "string",\n'
        '  "agent_evaluations": [\n'
        "    {\n"
        '      "agent_name": "string",\n'
        '      "step_index": 0,\n'
        '      "error_likelihood": 0.0,\n'
        '      "reasoning": "string",\n'
        '      "evidence": "string"\n'
        "    }\n"
        "  ],\n"
        '  "primary_conclusion": {\n'
        '    "type": "single_agent or multi_agent",\n'
        '    "attribution": ["agent_name"],\n'
        '    "mistake_step": 0,\n'
        '    "confidence": 0.0,\n'
        '    "reasoning": "string"\n'
        "  },\n"
        '  "alternative_hypotheses": [\n'
        "    {\n"
        '      "type": "single_agent or multi_agent",\n'
        '      "attribution": ["agent_name"],\n'
        '      "mistake_step": 0,\n'
        '      "confidence": 0.0,\n'
        '      "reasoning": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "</json>\n"
        "Do not output any text outside <json></json>."
    )


def build_objective_prompt(
    *,
    query: str,
    ground_truth: str,
    final_answer: str,
    context_summary: str,
    include_ground_truth: bool,
    attribution_target: str = "joint",
) -> str:
    target = str(attribution_target or "joint").lower()
    if target == "agent":
        target_block = (
            "Attribution Task: AGENT-LEVEL.\n"
            "Prioritize who is responsible. Step is optional supporting evidence.\n"
        )
    elif target == "step":
        target_block = (
            "Attribution Task: STEP-LEVEL.\n"
            "Prioritize the earliest responsible mistake step. Agent is supporting evidence.\n"
        )
    else:
        target_block = "Attribution Task: JOINT agent+step attribution.\n"
    gt_block = f"Ground Truth: {ground_truth}\n" if include_ground_truth and ground_truth else ""
    return (
        f"Original Query: {query}\n"
        f"{gt_block}"
        f"Final Answer: {final_answer}\n\n"
        f"{target_block}\n"
        "Conversation Analysis Context:\n"
        f"{context_summary}\n\n"
        "Please conduct objective attribution based on hierarchical context and causal impact.\n"
        "Focus on the earliest responsible mistake that directly leads to the wrong final answer."
    )
