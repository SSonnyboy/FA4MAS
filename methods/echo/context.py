"""ECHO 分层上下文提取。"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _extract_with_patterns(text: str, patterns: List[str], *, max_words: int) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return "No content available."
    for pattern in patterns:
        matches = re.findall(pattern, cleaned, flags=re.IGNORECASE)
        if not matches:
            continue
        extracted = matches[0].strip()
        words = extracted.split()[:max_words]
        return " ".join(words) + ("..." if len(extracted.split()) > max_words else "")
    return ""


def extract_key_decision(agent_content: str, *, max_words: int = 50, context_type: str = "decision_quality") -> str:
    if context_type == "handoff":
        patterns = [
            r"(?:received|got|obtained|from)\s+([^.!?]*[.!?])",
            r"(?:passing|providing|sending|to)\s+([^.!?]*[.!?])",
            r"(?:based on|using)\s+([^.!?]*[.!?])",
            r"(?:will|need to|should)\s+([^.!?]*(?:next|continue)[^.!?]*[.!?])",
        ]
    elif context_type == "error_propagation":
        patterns = [
            r"(?:error|mistake|wrong|incorrect|failed)\s+([^.!?]*[.!?])",
            r"(?:cannot|unable|couldn't|can't)\s+([^.!?]*[.!?])",
            r"(?:However|But|Unfortunately)\s+([^.!?]*[.!?])",
        ]
    else:
        patterns = [
            r"(?:I (?:conclude|determine|decide|believe|think))\s+([^.!?]*[.!?])",
            r"(?:Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The (?:answer|solution|result))\s+(?:is|appears)\s+([^.!?]*[.!?])",
            r"(?:Based on|Given)\s+([^.!?]*[.!?])",
        ]
    extracted = _extract_with_patterns(agent_content, patterns, max_words=max_words)
    if extracted:
        return extracted
    fallback = _clean_text(agent_content)
    words = fallback.split()[:max_words]
    return " ".join(words) + ("..." if len(fallback.split()) > max_words else "")


def summarize_agent(agent_content: str, *, max_words: int = 20, context_type: str = "general") -> str:
    if context_type == "handoff":
        patterns = [r"(?:received|got|obtained)\s+([^.!?]*[.!?])", r"(?:providing|sending)\s+([^.!?]*[.!?])"]
    elif context_type == "error_propagation":
        patterns = [r"(?:error|mistake|failed)\s+([^.!?]*[.!?])", r"(?:cannot|unable)\s+([^.!?]*[.!?])"]
    else:
        patterns = [
            r"(?:In conclusion|To conclude|Therefore|Thus|So|Hence),?\s+([^.!?]*[.!?])",
            r"(?:The (?:answer|result|solution|output))\s+(?:is|appears to be|seems to be)\s+([^.!?]*[.!?])",
            r"(?:I (?:found|determined|concluded|calculated))\s+([^.!?]*[.!?])",
        ]
    extracted = _extract_with_patterns(agent_content, patterns, max_words=max_words)
    if extracted:
        return extracted
    fallback = _clean_text(agent_content)
    words = fallback.split()[:max_words]
    return " ".join(words) + ("..." if len(fallback.split()) > max_words else "")


def obtain_milestones(agent_content: str, *, max_words: int = 15, context_type: str = "general") -> str:
    if context_type == "handoff":
        patterns = [
            r"(?:received|obtained|got)\s+([^.!?]*(?:from|data|information)[^.!?]*[.!?])",
            r"(?:provided|sent|passed)\s+([^.!?]*(?:to|data|information)[^.!?]*[.!?])",
            r"(?:completed|finished)\s+([^.!?]*(?:handoff|transfer)[^.!?]*[.!?])",
        ]
    elif context_type == "error_propagation":
        patterns = [
            r"(?:error|mistake|failure)\s+(?:occurred|detected)\s+([^.!?]*[.!?])",
            r"(?:identified|found)\s+(?:error|issue|problem)\s+([^.!?]*[.!?])",
            r"(?:corrected|fixed|resolved)\s+([^.!?]*[.!?])",
        ]
    else:
        patterns = [
            r"(?:completed|finished|achieved|accomplished)\s+([^.!?]*[.!?])",
            r"(?:created|generated|produced|built)\s+([^.!?]*[.!?])",
            r"(?:step\s+\d+|phase\s+\d+|stage\s+\d+)\s*[:-]?\s*([^.!?]*[.!?])",
            r"(?:successfully|finally)\s+([^.!?]*[.!?])",
        ]
    extracted = _extract_with_patterns(agent_content, patterns, max_words=max_words)
    if extracted:
        return extracted
    fallback = _clean_text(agent_content)
    words = fallback.split()[:max_words]
    return " ".join(words) + ("..." if len(fallback.split()) > max_words else "")


def _resolve_agent_name(step: Dict[str, Any], index: int) -> str:
    return str(step.get("name") or step.get("role") or step.get("agent") or f"agent_{index}")


def _resolve_agent_role(step: Dict[str, Any]) -> str:
    return str(step.get("role") or step.get("name") or step.get("agent") or "unknown")


def build_hierarchical_contexts(
    conversation_history: List[Dict[str, Any]],
    *,
    context_type: str = "decision_quality",
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for current_idx in range(len(conversation_history)):
        current = conversation_history[current_idx]
        context = {
            "current_agent": {
                "index": current_idx,
                "name": _resolve_agent_name(current, current_idx),
                "role": _resolve_agent_role(current),
                "content": str(current.get("content", "")),
            },
            "context_levels": {"immediate": [], "nearby": [], "distant": [], "milestones": []},
        }

        for idx, item in enumerate(conversation_history):
            if idx == current_idx:
                continue
            distance = abs(idx - current_idx)
            info = {
                "index": idx,
                "name": _resolve_agent_name(item, idx),
                "role": _resolve_agent_role(item),
                "distance": distance,
            }
            content = str(item.get("content", ""))
            if distance == 1:
                info["detail_level"] = "full"
                info["content"] = content
                context["context_levels"]["immediate"].append(info)
            elif distance <= 3:
                info["detail_level"] = "key_decisions"
                info["content"] = extract_key_decision(content, context_type=context_type)
                context["context_levels"]["nearby"].append(info)
            elif distance <= 6:
                info["detail_level"] = "summary"
                info["content"] = summarize_agent(content, context_type=context_type)
                context["context_levels"]["distant"].append(info)
            else:
                info["detail_level"] = "milestones"
                info["content"] = obtain_milestones(content, context_type=context_type)
                context["context_levels"]["milestones"].append(info)

        for level in context["context_levels"].values():
            level.sort(key=lambda row: row["index"])
        contexts.append(context)
    return contexts


def build_conversation_summary(
    contexts: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    *,
    max_chars: int = 12000,
) -> str:
    lines = ["=== CONVERSATION AGENTS ==="]
    for idx, step in enumerate(history):
        name = _resolve_agent_name(step, idx)
        role = _resolve_agent_role(step)
        lines.append(f"Step {idx} - {name} ({role}):")
        lines.append(_clean_text(step.get("content", "")))
        lines.append("")

    if contexts:
        lines.append("=== HIERARCHICAL CONTEXT EXAMPLE ===")
        example = contexts[min(1, len(contexts) - 1)]
        current = example["current_agent"]
        lines.append(f"Current Agent: Step {current['index']} - {current['name']} ({current['role']})")
        for level_name in ("immediate", "nearby", "distant", "milestones"):
            lines.append(f"{level_name.upper()}:")
            for item in example["context_levels"][level_name]:
                lines.append(
                    f"  Step {item['index']} - {item['name']} ({item['role']}), detail={item['detail_level']}: {item['content']}"
                )
            lines.append("")

    summary = "\n".join(lines).strip()
    if len(summary) > max_chars:
        return summary[:max_chars] + "\n... [truncated]"
    return summary

