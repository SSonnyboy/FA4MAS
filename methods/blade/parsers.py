"""BLADE 输出解析器。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from core.utils import normalize_agent, normalize_step


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def extract_json_block(text: str) -> Dict[str, Any]:
    block = re.search(r"<json>\s*([\s\S]*?)\s*</json>", text, flags=re.IGNORECASE)
    candidate = block.group(1).strip() if block else str(text or "").strip()
    if not candidate:
        return {"parse_error": "empty_response", "raw_response": text}
    try:
        payload = json.loads(candidate)
        payload["raw_response"] = text
        return payload
    except json.JSONDecodeError as error:
        return {"parse_error": f"json_decode_error: {error}", "raw_response": text}


def _unique_ints(values: List[int]) -> List[int]:
    seen = set()
    ordered: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _extract_step_mentions(text: str) -> List[int]:
    step_hits = [int(hit) for hit in re.findall(r"[Ss]tep\s*(\d+)", text)]
    if step_hits:
        return _unique_ints(step_hits)
    raw_ints = [int(hit) for hit in re.findall(r"\b\d+\b", text)]
    return _unique_ints(raw_ints)


def _extract_section_candidates(text: str, heading: str, top_k: int) -> List[Dict[str, Any]]:
    pattern = re.compile(
        rf"{heading}[^:\n]*[:\n](.*?)(?:\n\s*(?:forward|backward)[^:\n]*[:\n]|\Z)",
        flags=re.IGNORECASE | re.S,
    )
    match = pattern.search(text)
    if not match:
        return []
    section = match.group(1)
    steps = [step for step in _extract_step_mentions(section) if step >= 0][: max(1, top_k)]
    return [{"step": step, "score": 0.45, "reason": "text_fallback"} for step in steps]


def _extract_global_confidence(text: str) -> float:
    match = re.search(r"global[_\s-]*confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if not match:
        return 0.0
    return max(0.0, min(1.0, _safe_float(match.group(1), 0.0)))


def _screening_text_fallback(raw: str, *, top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    text = str(raw or "")
    if not text.strip():
        return [], [], 0.0

    forward = _extract_section_candidates(text, "forward", top_k)
    backward = _extract_section_candidates(text, "backward", top_k)

    if not forward and not backward:
        # 最弱回退：从文本中抓 step 提及并分给两侧，保持去重与预算。
        steps = [step for step in _extract_step_mentions(text) if step >= 0][: max(2, top_k)]
        split = (len(steps) + 1) // 2
        forward = [{"step": step, "score": 0.40, "reason": "text_step_fallback"} for step in steps[:split]]
        backward = [{"step": step, "score": 0.40, "reason": "text_step_fallback"} for step in steps[split:]]

    conf = _extract_global_confidence(text)
    return forward[: max(1, top_k)], backward[: max(1, top_k)], conf


def parse_screening(payload: Dict[str, Any], *, top_k: int) -> Dict[str, Any]:
    forward: List[Dict[str, Any]] = []
    backward: List[Dict[str, Any]] = []
    for row in payload.get("forward_candidates") or []:
        if not isinstance(row, dict):
            continue
        step = normalize_step(row.get("step"))
        if step is None:
            continue
        forward.append(
            {
                "step": step,
                "score": max(0.0, min(1.0, _safe_float(row.get("score"), 0.0))),
                "reason": str(row.get("reason") or ""),
            }
        )
    for row in payload.get("backward_candidates") or []:
        if not isinstance(row, dict):
            continue
        step = normalize_step(row.get("step"))
        if step is None:
            continue
        backward.append(
            {
                "step": step,
                "score": max(0.0, min(1.0, _safe_float(row.get("score"), 0.0))),
                "reason": str(row.get("reason") or ""),
            }
        )

    global_conf = max(0.0, min(1.0, _safe_float(payload.get("global_confidence"), 0.0)))
    parse_error = payload.get("parse_error")
    raw_response = str(payload.get("raw_response") or "")

    if (not forward or not backward) and raw_response:
        fb_forward, fb_backward, fb_conf = _screening_text_fallback(raw_response, top_k=top_k)
        if not forward and fb_forward:
            forward = fb_forward
        if not backward and fb_backward:
            backward = fb_backward
        if global_conf <= 0.0 and fb_conf > 0.0:
            global_conf = fb_conf

    return {
        "forward_candidates": forward[: max(1, top_k)],
        "backward_candidates": backward[: max(1, top_k)],
        "global_confidence": global_conf,
        "parse_error": parse_error,
        "raw_response": raw_response,
    }


def parse_tournament(payload: Dict[str, Any]) -> Dict[str, Any]:
    winner = str(payload.get("winner") or "").strip().lower()
    if winner not in {"a", "b", "tie"}:
        winner = "tie"
    return {
        "winner": winner,
        "confidence": max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.0))),
        "reason": str(payload.get("reason") or ""),
        "parse_error": payload.get("parse_error"),
        "raw_response": str(payload.get("raw_response") or ""),
    }


def parse_final_text(text: str) -> Dict[str, Any]:
    content = str(text or "")
    agent_match = re.search(r"Agent Name:\s*([^\n*]+)", content, flags=re.IGNORECASE)
    step_match = re.search(r"Step Number:\s*([0-9]+)", content, flags=re.IGNORECASE)
    reason_match = re.search(r"Reason for Mistake:\s*(.*?)(?:\nConfidence:|\Z)", content, flags=re.IGNORECASE | re.S)
    confidence_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", content, flags=re.IGNORECASE)
    step = normalize_step(step_match.group(1)) if step_match else None

    if step is None:
        # Free-form fallback for reasoning-heavy responses.
        fallback_patterns = [
            r"earliest\s+(?:mistake|error|root[-\s]?cause)\s+step\s*(?:is|:)?\s*(?:likely\s*)?step\s*(\d+)",
            r"most\s+responsible\s+step\s*(?:is|:)?\s*(?:likely\s*)?step\s*(\d+)",
            r"mistake\s+appears\s+to\s+be\s+in\s+step\s*(\d+)",
            r"step\s*(\d+)\s*(?:is|was)\s*(?:the\s*)?(?:earliest|root[-\s]?cause|most responsible)",
            r"earliest\s+step\s*(?:is|:)?\s*(\d+)",
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if not match:
                continue
            step = normalize_step(match.group(1))
            if step is not None:
                break

    if not agent_match:
        fallback_agent_patterns = [
            r"mistake[_\s-]*agent\s*[:=]\s*([a-zA-Z0-9_\- ]+)",
            r"agent\s+name\s*(?:is|:)\s*([a-zA-Z0-9_\- ]+)",
            r"responsible\s+agent\s*(?:is|:)\s*([a-zA-Z0-9_\- ]+)",
        ]
        for pattern in fallback_agent_patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                agent_match = match
                break

    return {
        "mistake_agent": normalize_agent(agent_match.group(1).strip()) if agent_match else None,
        "mistake_step": step,
        "reason": reason_match.group(1).strip() if reason_match else "",
        "confidence": max(0.0, min(1.0, _safe_float(confidence_match.group(1), 0.0))) if confidence_match else 0.0,
    }


def parse_escalation(payload: Dict[str, Any]) -> Dict[str, Any]:
    agent = normalize_agent(payload.get("agent"))
    step = normalize_step(payload.get("step"))
    confidence = max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.0)))
    reason = str(payload.get("reason") or "")
    parse_error = payload.get("parse_error")
    raw_response = str(payload.get("raw_response") or "")

    if (agent is None or step is None) and raw_response:
        if agent is None:
            match = re.search(r"agent(?:\s+name)?\s*[:=]\s*([^\n,]+)", raw_response, flags=re.IGNORECASE)
            if match:
                agent = normalize_agent(match.group(1).strip())
        if step is None:
            match = re.search(r"(?:mistake\s+)?step(?:\s+number)?\s*[:=]\s*(\d+)", raw_response, flags=re.IGNORECASE)
            if not match:
                match = re.search(r"earliest.*?step\s*(\d+)", raw_response, flags=re.IGNORECASE | re.S)
            if match:
                step = normalize_step(match.group(1))
        if confidence <= 0.0:
            match = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)", raw_response, flags=re.IGNORECASE)
            if match:
                confidence = max(0.0, min(1.0, _safe_float(match.group(1), 0.0)))
        if not reason:
            match = re.search(r"reason\s*[:=]\s*(.*)", raw_response, flags=re.IGNORECASE)
            if match:
                reason = match.group(1).strip()

    return {
        "agent": agent,
        "step": step,
        "confidence": confidence,
        "reason": reason,
        "parse_error": parse_error,
        "raw_response": raw_response,
    }
