"""CHIEF 输出解析器。"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.S | re.M)
    return match.group(1).strip() if match else None


def _extract_float(pattern: str, text: str, default: float = 0.0) -> float:
    value = _extract(pattern, text)
    try:
        return float(value) if value is not None else default
    except Exception:
        return default


def _extract_int(pattern: str, text: str) -> int | None:
    value = _extract(pattern, text)
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _parse_inline_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    items = []
    for item in raw.split(","):
        cleaned = item.strip().strip('"').strip("'")
        if cleaned:
            items.append(cleaned)
    return items


def parse_subtasks(raw: str) -> List[Dict[str, Any]]:
    names = re.findall(r"The Subtask Name:\s*([^\n]+)", raw)
    range_matches = re.findall(r"Step Range:\s*(?:step)?(\d+)\s*-\s*(?:step)?(\d+)", raw)
    oracles = re.findall(r"The Oracle:\s*([^\n]+)", raw)
    evidences = re.findall(r"Evidence:\s*([^\n]+)", raw)
    loop_blocks = re.findall(r"Loop Info:\s*\{([\s\S]+?)\}", raw)

    subtasks = []
    size = min(len(names), len(range_matches), len(oracles), len(evidences))
    for index in range(size):
        loop_text = loop_blocks[index] if index < len(loop_blocks) else ""
        subtasks.append(
            {
                "id": f"S{index + 1}",
                "name": names[index].strip(),
                "step_range": f"{range_matches[index][0]}-{range_matches[index][1]}",
                "oracle": oracles[index].strip(),
                "evidence": evidences[index].strip(),
                "loop_info": {
                    "is_loop_related": (_extract(r"is_loop_related:\s*(true|false)", loop_text) == "true"),
                    "loop_role": _extract(r"loop_role:\s*([a-zA-Z_]+)", loop_text) or "none",
                    "loop_group_id": None if (_extract(r"loop_group_id:\s*([A-Za-z0-9_]+|null)", loop_text) in (None, "null")) else _extract(r"loop_group_id:\s*([A-Za-z0-9_]+|null)", loop_text),
                    "reversibility": _extract(r"reversibility:\s*([a-zA-Z_]+)", loop_text) or "reversible",
                    "loop_risk_score": _extract_float(r"loop_risk_score:\s*([0-9]*\.?[0-9]+)", loop_text),
                },
            }
        )
    return subtasks


def parse_subtask_edges(raw: str) -> List[Dict[str, Any]]:
    blocks = [block.strip() for block in re.split(r"(?=^From:\s*S[0-9]+)", raw, flags=re.M) if block.strip()]
    edges = []
    for block in blocks:
        edges.append(
            {
                "from": _extract(r"^From:\s*(S[0-9]+)", block) or "",
                "to": _extract(r"^To:\s*(S[0-9]+)", block) or "",
                "type": _extract(r"^Type:\s*([^\n]+)", block) or "",
                "strength": _extract_float(r"^Strength:\s*([0-9]*\.?[0-9]+)", block),
                "explanation": _extract(r"^Explanation:\s*(.*)", block) or "",
            }
        )
    return edges


def parse_subtask_agents(raw: str, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections = [section.strip() for section in re.split(r"^The Subtask Name:\s*", raw, flags=re.M) if section.strip()]
    parsed_by_name: Dict[str, Dict[str, Any]] = {}

    for section in sections:
        first_newline = section.find("\n")
        name = section if first_newline == -1 else section[:first_newline].strip()
        rest = "" if first_newline == -1 else section[first_newline + 1 :]

        agents: List[Dict[str, Any]] = []
        agent_blocks = re.findall(
            r"-\s*Agent:\s*(.*?)\s*--\s*Action:\s*(.*?)\s*--\s*Observation:\s*(.*?)\s*--\s*Thought:\s*(.*?)\s*--\s*Result:\s*(.*?)(?=\n-\s*Agent:|\nData_Flow:|\Z)",
            rest,
            flags=re.S,
        )
        for agent, action, observation, thought, result in agent_blocks:
            agents.append(
                {
                    "agent": agent.strip(),
                    "action": action.strip(),
                    "observation": observation.strip(),
                    "thought": thought.strip(),
                    "result": result.strip(),
                }
            )

        data_flow = []
        for block in re.split(r"(?=^\s*-\s*from_step:\s*)", rest, flags=re.M):
            block = block.strip()
            if not block.startswith("-"):
                continue
            from_step = _extract_int(r"from_step:\s*([0-9]+)", block)
            to_step = _extract_int(r"to_step:\s*([0-9]+)", block)
            if from_step is None or to_step is None:
                continue
            data_flow.append(
                {
                    "from_step": from_step,
                    "to_step": to_step,
                    "source_agent": _extract(r"source_agent:\s*([^\n]+)", block) or "",
                    "target_agent": _extract(r"target_agent:\s*([^\n]+)", block) or "",
                    "data_item": _extract(r"data_item:\s*\"([^\"]+)\"", block) or "",
                    "data_type": _extract(r"data_type:\s*\"([^\"]+)\"", block) or "",
                    "transformation": _extract(r"transformation:\s*\"([^\"]+)\"", block) or "",
                    "correctness": _extract(r"correctness:\s*\"([^\"]+)\"", block) or "",
                    "confidence": _extract_float(r"confidence:\s*([0-9]*\.?[0-9]+)", block),
                }
            )
        parsed_by_name[name] = {"agents": agents, "data_flow": data_flow}

    merged = []
    for subtask in subtasks:
        parsed = parsed_by_name.get(subtask["name"], {"agents": [], "data_flow": []})
        merged.append({**subtask, "agents": parsed["agents"], "data_flow": parsed["data_flow"]})
    return merged


def parse_agent_edges(raw: str) -> List[Dict[str, Any]]:
    sections = [section.strip() for section in re.split(r"^The Subtask Name:\s*", raw, flags=re.M) if section.strip()]
    all_edges = []
    for section in sections:
        first_newline = section.find("\n")
        name = section if first_newline == -1 else section[:first_newline].strip()
        rest = "" if first_newline == -1 else section[first_newline + 1 :]
        edges = []
        for block in re.split(r"(?=^\s*-\s*From_agent:\s*)", rest, flags=re.M):
            block = block.strip()
            if not block.startswith("-"):
                continue
            edges.append(
                {
                    "From_agent": _extract(r"From_agent:\s*([^\n]+)", block) or "",
                    "To_agent": _extract(r"To_agent:\s*([^\n]+)", block) or "",
                    "agent_dependency_type": _extract(r"agent_dependency_type:\s*([^\n]+)", block) or "",
                    "agent_strength": _extract_float(r"agent_strength:\s*([0-9]*\.?[0-9]+)", block),
                    "agent_explanation": _extract(r"agent_explanation:\s*(.*)", block) or "",
                }
            )
        all_edges.append({"The Subtask Name": name, "Agents_edges": edges})
    return all_edges


def parse_candidate_set(raw: str) -> Dict[str, Any]:
    candidate_steps: List[Dict[str, Any]] = []
    steps_block = _extract(r"Candidate Error Steps:\s*([\s\S]+)$", raw) or ""
    for block in re.findall(r"-\s*step_id:\s*.+?(?=\n\s*-\s*step_id:|\Z)", steps_block, re.S):
        step_id = _extract_int(r"step_id:\s*([0-9]+)", block)
        if step_id is None:
            continue
        source_step_raw = _extract(r"source_step:\s*([0-9]+|null)", block)
        source_step = None if source_step_raw in (None, "null") else _extract_int(r"source_step:\s*([0-9]+)", block)
        candidate_steps.append(
            {
                "step_id": step_id,
                "agent_in_step": _parse_inline_list(_extract(r"agent_in_step:\s*\[([^\]]*)\]", block)),
                "loop_issue": {
                    "is_in_loop": _extract(r"is_in_loop:\s*(true|false)", block) == "true",
                    "loop_role": _extract(r"loop_role:\s*([a-zA-Z_]+)", block) or "none",
                    "loop_group_id": None if (_extract(r"loop_group_id:\s*([A-Za-z0-9_]+|null)", block) in (None, "null")) else _extract(r"loop_group_id:\s*([A-Za-z0-9_]+|null)", block),
                },
                "data_issue": {
                    "has_issue": _extract(r"has_issue:\s*(true|false)", block) == "true",
                    "data_item": _extract(r"data_item:\s*\"?([^\n\"]+)\"?", block) or "",
                    "source_step": source_step,
                    "consistency_score": _extract_float(r"consistency_score:\s*([0-9]*\.?[0-9]+)", block),
                    "explanation": _extract(r"explanation:\s*([^\n]+)", block) or "",
                },
                "irrecoverability_issue": {
                    "is_irrecoverable": _extract(r"is_irrecoverable:\s*(true|false)", block) == "true",
                    "reason": _extract(r"reason:\s*([^\n]+)", block) or "",
                },
                "impact": {
                    "affected_steps": [int(item) for item in _parse_inline_list(_extract(r"affected_steps:\s*\[([^\]]*)\]", block)) if item.isdigit()],
                    "impact_score": _extract_float(r"impact_score:\s*([0-9]*\.?[0-9]+)", block),
                },
                "information": {
                    "input": _extract(r"input:\s*([^\n]+)", block) or "",
                    "output": _extract(r"output:\s*([^\n]+)", block) or "",
                },
                "confidence": _extract_float(r"confidence:\s*([0-9]*\.?[0-9]+)", block),
            }
        )

    return {
        "candidate_error_subtasks": _parse_inline_list(_extract(r"Candidate Error Subtasks:\s*\[([^\]]*)\]", raw)),
        "candidate_error_agents": _parse_inline_list(_extract(r"Candidate Error Agents:\s*\[([^\]]*)\]", raw)),
        "candidate_error_steps": candidate_steps,
    }


def parse_final_prediction(raw: str) -> Dict[str, Any]:
    return {
        "mistake_agent": _extract(r"Agent Name:\s*([^\n*]+)", raw),
        "mistake_step": _extract_int(r"Step Number:\s*([0-9]+)", raw),
        "reason": _extract(r"Reason for Mistake:\s*(.*)", raw) or "",
    }

