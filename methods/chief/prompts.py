"""CHIEF 提示词构建。"""
from __future__ import annotations

from typing import Any, Dict, List

from core.utils import join_lines


def _ground_truth_block(ground_truth: str) -> str:
    value = str(ground_truth or "").strip()
    if not value:
        return ""
    return f"The correct answer for the problem is: {value}\n\n"


def build_subtask_prompt(history: List[Dict[str, Any]], question: str, ground_truth: str, rag_text: str) -> str:
    answer_block = _ground_truth_block(ground_truth)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"{answer_block}"
        "Here is the conversation in JSON format:\n"
        + str(history)
        + f"\n\nThere are total {len(history)} steps, each entry provides the agent output and its role.\n\n"
        "Here is the retrieved reference example:\n"
        + rag_text
        + "\n\nBased on this conversation and retrieved example, please decompose the reasoning into semantic subtasks.\n"
        "You must perform a self-reflection process to optimize your decomposition:\n"
        "1. Draft Optimization: propose an initial set of subtasks.\n"
        "2. Evidence Alignment: ensure each subtask’s step range aligns with the dialogue.\n"
        "3. Final Optimization: ensure step ranges are continuous, cover all steps, and do NOT overlap.\n\n"
        "Output format (plain text, NO markdown, NO bullets, NO '**'):\n"
        "The Subtask Name: <your prediction>\n"
        "Step Range: <start-end>\n"
        "The Oracle: <your prediction>\n"
        "Evidence: <one-line summary>\n"
        "Loop Info:\n"
        "{\n"
        "  is_loop_related: true/false,\n"
        "  loop_role: entry/internal/exit/none,\n"
        "  loop_group_id: L1/L2/... or null,\n"
        "  reversibility: reversible/partial/irreversible,\n"
        "  loop_risk_score: float(0-1)\n"
        "}\n"
    )


def build_subtask_edge_prompt(history: List[Dict[str, Any]], question: str, ground_truth: str, subtasks: List[Dict[str, Any]]) -> str:
    answer_block = _ground_truth_block(ground_truth)
    return (
        "You are an expert in causal reasoning and multi-agent task analysis.\n"
        f"The problem is: {question}\n"
        f"{answer_block}"
        "Here is the conversation in JSON format:\n"
        + str(history)
        + f"\n\nThere are total {len(history)} steps.\n\n"
        "Here are the subtasks with their IDs (in execution order):\n"
        + str(subtasks)
        + "\n\nNow, construct causal edges ONLY for consecutive subtask pairs: (S1->S2), (S2->S3), ...\n"
        "For each consecutive pair output one block containing From, To, Type, Strength, Explanation, Data_Transfer and Failure Modes.\n"
        "Use the exact field names from the original schema. No extra commentary."
    )


def build_agent_prompt(history: List[Dict[str, Any]], question: str, ground_truth: str, subtasks: List[Dict[str, Any]]) -> str:
    answer_block = _ground_truth_block(ground_truth)
    subtask_lines = [
        f"- id: {subtask.get('id')}, name: {subtask.get('name')}, step_range: {subtask.get('step_range')}"
        for subtask in subtasks
    ]
    return (
        "You are an AI assistant tasked with analyzing multi-agent execution traces.\n"
        f"The problem is: {question}\n"
        f"{answer_block}"
        "Here is the conversation in JSON format:\n"
        + str(history)
        + f"\n\nThere are total {len(history)} steps.\n\n"
        "Below are the subtasks:\n"
        + join_lines(subtask_lines)
        + "\n\nFor each subtask, identify the agents, summarize Action / Observation / Thought / Result, "
        "and provide within-subtask Data_Flow blocks using the original schema."
    )


def build_agent_edge_prompt(history: List[Dict[str, Any]], question: str, ground_truth: str, subtasks_agents: List[Dict[str, Any]]) -> str:
    answer_block = _ground_truth_block(ground_truth)
    subtask_lines = []
    for subtask in subtasks_agents:
        agent_names = [agent.get("agent") for agent in subtask.get("agents", []) if agent.get("agent")]
        subtask_lines.append(
            f"- id: {subtask.get('id')}, name: {subtask.get('name')}, step_range: {subtask.get('step_range')}, agents: {agent_names}"
        )
    return (
        "You are an expert in causal reasoning and multi-agent task analysis.\n"
        f"The problem is: {question}\n"
        f"{answer_block}"
        "Here is the conversation in JSON format:\n"
        + str(history)
        + f"\n\nThere are total {len(history)} steps.\n\n"
        "Below are the subtasks with their agents:\n"
        + join_lines(subtask_lines)
        + "\n\nFor each subtask, construct causal edges BETWEEN agents inside this subtask only using the original schema."
    )


def build_candidate_prompt(history: List[Dict[str, Any]], question: str, ground_truth: str, dag_graph: Dict[str, Any]) -> str:
    answer_block = _ground_truth_block(ground_truth)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"{answer_block}"
        "Here is the conversation:\n"
        + str(history)
        + f"\n\nThere are total {len(history)} steps.\n\n"
        "Here is the graph describing the reasoning structure:\n"
        + str(dag_graph)
        + "\n\nIdentify candidate error subtasks, agents and at least 5 candidate error steps using the original schema."
    )


def build_final_prompt(history: List[Dict[str, Any]], question: str, ground_truth: str, candidate_set: Dict[str, Any], dag_graph: Dict[str, Any]) -> str:
    answer_block = _ground_truth_block(ground_truth)
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation solving a real-world problem.\n"
        f"The problem is: {question}\n"
        f"{answer_block}"
        "Here is the multi-agent conversation:\n"
        + str(history)
        + f"\n\nThere are total {len(history)} steps.\n\n"
        "Here is the structured candidate_set generated by a previous analysis stage.\n"
        + str(candidate_set)
        + "\n\nHere is the graph:\n"
        + str(dag_graph)
        + "\n\nIdentify the SINGLE most responsible reasoning mistake.\n"
        "Answer in this exact format:\n"
        "Agent Name: ...\n"
        "Step Number: ...\n"
        "Reason for Mistake: ...\n"
    )
