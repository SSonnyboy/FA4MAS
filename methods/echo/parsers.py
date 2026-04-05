"""ECHO 输出解析。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from core.utils import normalize_step


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def extract_json_block(text: str) -> Dict[str, Any]:
    block = re.search(r"<json>\s*([\s\S]*?)\s*</json>", text, flags=re.IGNORECASE)
    candidate = block.group(1).strip() if block else text.strip()
    if not candidate:
        return {"parse_error": "empty_response", "raw_response": text}
    try:
        payload = json.loads(candidate)
        payload["raw_response"] = text
        return payload
    except json.JSONDecodeError as error:
        return {"parse_error": f"json_decode_error: {error}", "raw_response": text}


def _normalize_conclusion(conclusion: Any) -> Dict[str, Any]:
    if not isinstance(conclusion, dict):
        return {"type": "single_agent", "attribution": [], "mistake_step": None, "confidence": 0.0, "reasoning": ""}
    attribution_raw = conclusion.get("attribution")
    if isinstance(attribution_raw, list):
        attribution = [str(item).strip() for item in attribution_raw if str(item).strip()]
    elif attribution_raw:
        attribution = [str(attribution_raw).strip()]
    else:
        attribution = []
    return {
        "type": str(conclusion.get("type") or "single_agent"),
        "attribution": attribution,
        "mistake_step": normalize_step(conclusion.get("mistake_step")),
        "confidence": _safe_float(conclusion.get("confidence"), 0.0),
        "reasoning": str(conclusion.get("reasoning") or ""),
    }


def normalize_objective_analysis(payload: Dict[str, Any], analyst_role: str, analyst_index: int) -> Dict[str, Any]:
    agent_evaluations: List[Dict[str, Any]] = []
    for row in payload.get("agent_evaluations") or []:
        if not isinstance(row, dict):
            continue
        agent_name = str(row.get("agent_name") or "").strip()
        if not agent_name:
            continue
        agent_evaluations.append(
            {
                "agent_name": agent_name,
                "step_index": normalize_step(row.get("step_index")),
                "error_likelihood": _safe_float(row.get("error_likelihood"), 0.0),
                "reasoning": str(row.get("reasoning") or ""),
                "evidence": str(row.get("evidence") or ""),
            }
        )

    alternatives: List[Dict[str, Any]] = []
    for row in payload.get("alternative_hypotheses") or []:
        alternatives.append(_normalize_conclusion(row))

    normalized = {
        "analyst_role": analyst_role,
        "analyst_id": analyst_index,
        "analysis_summary": str(payload.get("analysis_summary") or ""),
        "agent_evaluations": agent_evaluations,
        "primary_conclusion": _normalize_conclusion(payload.get("primary_conclusion")),
        "alternative_hypotheses": alternatives,
        "raw_response": str(payload.get("raw_response") or ""),
    }
    if payload.get("parse_error"):
        normalized["parse_error"] = payload["parse_error"]
    return normalized

