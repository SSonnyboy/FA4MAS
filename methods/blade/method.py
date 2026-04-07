"""BLADE: Budgeted Local Attribution with Dual Evidence."""
from __future__ import annotations

from pathlib import Path
import re
from statistics import mean
from typing import Any, Dict, List, Tuple

from core.llm import chat_completion
from core.utils import load_json, normalize_agent, normalize_step
from methods.base import BaseMethod

from .parsers import (
    extract_json_block,
    parse_escalation,
    parse_final_text,
    parse_screening,
)
from .prompts import (
    build_escalation_prompt,
    build_escalation_system_prompt,
    build_finalize_prompt,
    build_finalize_system_prompt,
    build_screening_prompt,
    build_screening_system_prompt,
    build_tournament_system_prompt,
)


class BLADEMethod(BaseMethod):
    """双证据候选筛选 + 局部因果锦标赛 + 预算可控归因。"""

    ESCALATION_ROLES = [
        "conservative-evidence-only",
        "counterfactual-skeptic",
        "propagation-focused",
    ]

    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        self.use_ground_truth = bool(self.params.get("use_ground_truth_in_prompt", False))
        self.max_event_chars = int(self.params.get("max_event_chars", 220))
        self.max_events_for_screening = int(self.params.get("max_events_for_screening", 42))
        self.candidate_top_k = int(self.params.get("candidate_top_k", 8))
        self.max_tournament_candidates = int(self.params.get("max_tournament_candidates", 8))
        self.local_window = int(self.params.get("local_window", 2))
        self.enable_screening_model = bool(self.params.get("enable_screening_model", True))
        self.use_finalize_model = bool(self.params.get("use_finalize_model", True))
        self.use_pointwise_reranker = bool(self.params.get("use_pointwise_reranker", True))
        self.enable_terminal_projection = bool(self.params.get("enable_terminal_projection", False))
        self.rerank_top_k = int(self.params.get("rerank_top_k", 4))
        self.screening_max_tokens = int(self.params.get("screening_max_tokens", 700))
        self.tournament_max_tokens = int(self.params.get("tournament_max_tokens", 120))
        self.rerank_max_tokens = int(self.params.get("rerank_max_tokens", 180))
        self.finalize_max_tokens = int(self.params.get("finalize_max_tokens", 420))
        self.escalation_max_tokens = int(self.params.get("escalation_max_tokens", 320))
        self.uncertainty_threshold = float(self.params.get("uncertainty_threshold", 0.60))
        self.enable_escalation = bool(self.params.get("enable_escalation", True))
        self.escalation_num_analysts = int(self.params.get("escalation_num_analysts", 2))

    def _call_model(
        self,
        *,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float | None = None,
    ) -> str:
        result = chat_completion(
            self.client,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature if temperature is None else float(temperature),
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        return result.content

    @staticmethod
    def _extract_ground_truth(sample: Dict[str, Any]) -> Tuple[str | None, int | None]:
        gt_agent = sample.get("mistake_agent")
        gt_step = sample.get("mistake_step")
        if (gt_agent is None or gt_step is None) and isinstance(sample.get("gt"), dict):
            gt_agent = gt_agent if gt_agent is not None else sample["gt"].get("agent")
            gt_step = gt_step if gt_step is not None else sample["gt"].get("step")
        return normalize_agent(gt_agent), normalize_step(gt_step)

    @staticmethod
    def _extract_answer(sample: Dict[str, Any]) -> str:
        return str(sample.get("ground_truth") or sample.get("answer") or "")

    @staticmethod
    def _resolve_final_answer(sample: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        final_answer = str(sample.get("final_answer") or sample.get("answer") or "").strip()
        if final_answer:
            return final_answer
        for item in reversed(history):
            content = str(item.get("content", "")).strip()
            if content and content.upper() != "TERMINATE":
                return content
        return ""

    @staticmethod
    def _resolve_agent_field(sample: Dict[str, Any]) -> str:
        history = sample.get("history") or []
        if not history:
            return "name"
        first_entry = history[0]
        if "name" in first_entry and str(first_entry.get("name", "")).strip():
            return "name"
        if "role" in first_entry and str(first_entry.get("role", "")).strip():
            return "role"
        if "name" in first_entry:
            return "name"
        return "agent"

    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _truncate(self, value: str, *, max_chars: int | None = None) -> str:
        text = " ".join(str(value or "").split())
        limit = self.max_event_chars if max_chars is None else max_chars
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    @staticmethod
    def _infer_action_type(content: str) -> str:
        lowered = content.lower()
        if "terminate" in lowered:
            return "termination"
        if "exitcode:" in lowered:
            return "execution_result"
        if "```python" in lowered or "```sh" in lowered:
            return "code_proposal"
        if "assume" in lowered or "hypothetical" in lowered:
            return "assumption"
        if "verify" in lowered or "confirmed" in lowered or "agreed" in lowered:
            return "verification"
        if "plan" in lowered or "step-by-step" in lowered:
            return "planning"
        return "reasoning"

    @staticmethod
    def _infer_tool_status(content: str) -> str:
        lowered = content.lower()
        if "exitcode:" not in lowered and "timeout" not in lowered:
            return "none"
        if "timeout" in lowered or "exitcode: 124" in lowered:
            return "timeout"
        if "exitcode: 0" in lowered:
            return "success"
        if "exitcode:" in lowered:
            return "failure"
        return "none"

    @staticmethod
    def _infer_claim_type(content: str) -> str:
        lowered = content.lower()
        if "assume" in lowered or "hypothetical" in lowered:
            return "assumption"
        if "verify" in lowered or "confirmed" in lowered or "agreed" in lowered:
            return "verified_claim"
        if "output" in lowered or "final answer" in lowered or "result" in lowered:
            return "result_claim"
        if "error" in lowered or "failed" in lowered or "wrong" in lowered:
            return "error_report"
        return "none"

    def _estimate_noise_score(self, *, action_type: str, content: str, agent: str) -> float:
        score = 0.10
        lowered = content.lower()
        if action_type == "termination":
            score += 0.80
        if len(content.strip()) < 20:
            score += 0.25
        if "computer_terminal" in agent and "exitcode: 0" in lowered:
            score += 0.08
        if lowered.strip() in {"agreed.", "ok", "yes", "no"}:
            score += 0.35
        if "you are given: (1) a task" in lowered:
            score += 0.15
        return self._clip(score)

    def _estimate_forward_score(
        self,
        *,
        step: int,
        total_steps: int,
        action_type: str,
        tool_status: str,
        claim_type: str,
        noise_score: float,
        content: str,
    ) -> float:
        lowered = content.lower()
        early_weight = 1.0 - (step / max(1.0, float(total_steps)))
        score = 0.20 * early_weight
        if claim_type == "assumption":
            score += 0.36
        if claim_type == "error_report":
            score += 0.22
        if tool_status in {"failure", "timeout"}:
            score += 0.28
        if action_type in {"planning", "code_proposal"}:
            score += 0.10
        if "wrong" in lowered or "incorrect" in lowered:
            score += 0.18
        score -= 0.30 * noise_score
        return self._clip(score)

    def _estimate_backward_score(
        self,
        *,
        step: int,
        total_steps: int,
        action_type: str,
        tool_status: str,
        claim_type: str,
        noise_score: float,
        content: str,
    ) -> float:
        lowered = content.lower()
        late_weight = step / max(1.0, float(total_steps))
        score = 0.16 * late_weight
        if action_type in {"verification", "execution_result"}:
            score += 0.20
        if claim_type in {"result_claim", "error_report"}:
            score += 0.22
        if "final" in lowered or "terminate" in lowered:
            score += 0.18
        if tool_status in {"failure", "timeout"}:
            score += 0.10
        score -= 0.20 * noise_score
        return self._clip(score)

    def _eventize_history(self, history: List[Dict[str, Any]], *, agent_field: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        total_steps = len(history)
        for step, entry in enumerate(history):
            raw_content = str(entry.get("content", ""))
            action_type = self._infer_action_type(raw_content)
            tool_status = self._infer_tool_status(raw_content)
            claim_type = self._infer_claim_type(raw_content)
            agent = normalize_agent(entry.get(agent_field) or "unknown_agent") or "unknown_agent"
            evidence = self._truncate(raw_content, max_chars=self.max_event_chars)
            noise_score = self._estimate_noise_score(action_type=action_type, content=raw_content, agent=agent)
            forward_score = self._estimate_forward_score(
                step=step,
                total_steps=max(1, total_steps - 1),
                action_type=action_type,
                tool_status=tool_status,
                claim_type=claim_type,
                noise_score=noise_score,
                content=raw_content,
            )
            backward_score = self._estimate_backward_score(
                step=step,
                total_steps=max(1, total_steps - 1),
                action_type=action_type,
                tool_status=tool_status,
                claim_type=claim_type,
                noise_score=noise_score,
                content=raw_content,
            )
            events.append(
                {
                    "step": step,
                    "agent": agent,
                    "action_type": action_type,
                    "tool_status": tool_status,
                    "claim_type": claim_type,
                    "noise_score": round(noise_score, 4),
                    "forward_score": round(forward_score, 4),
                    "backward_score": round(backward_score, 4),
                    "evidence": evidence,
                }
            )
        return events

    def _deterministic_hints(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not events:
            return {"forward_steps": [], "backward_steps": [], "score_map": {}}

        forward_sorted = sorted(events, key=lambda item: (item["forward_score"], -item["step"]), reverse=True)
        backward_sorted = sorted(events, key=lambda item: (item["backward_score"], item["step"]), reverse=True)
        top_k = max(self.candidate_top_k, 4)
        forward_steps = [int(row["step"]) for row in forward_sorted[:top_k]]
        backward_steps = [int(row["step"]) for row in backward_sorted[:top_k]]

        score_map: Dict[int, float] = {}
        for row in events:
            combined = 0.65 * float(row["forward_score"]) + 0.35 * float(row["backward_score"])
            score_map[int(row["step"])] = round(self._clip(combined), 4)

        return {
            "forward_steps": forward_steps,
            "backward_steps": backward_steps,
            "score_map": score_map,
        }

    def _select_events_for_screening(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(events) <= self.max_events_for_screening:
            return events

        keep_steps = set()
        boundary = min(8, len(events))
        for row in events[:boundary]:
            keep_steps.add(int(row["step"]))
        for row in events[-boundary:]:
            keep_steps.add(int(row["step"]))

        scored = sorted(
            events,
            key=lambda item: (max(float(item["forward_score"]), float(item["backward_score"])), -float(item["noise_score"])),
            reverse=True,
        )
        for row in scored:
            keep_steps.add(int(row["step"]))
            if len(keep_steps) >= self.max_events_for_screening:
                break

        selected = [row for row in events if int(row["step"]) in keep_steps]
        selected.sort(key=lambda item: int(item["step"]))
        return selected

    def _run_screening(
        self,
        *,
        question: str,
        final_answer: str,
        ground_truth: str,
        events: List[Dict[str, Any]],
        deterministic_hints: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected_events = self._select_events_for_screening(events)
        prompt = build_screening_prompt(
            question=question,
            final_answer=final_answer,
            ground_truth=ground_truth,
            event_capsules=selected_events,
            deterministic_forward=deterministic_hints.get("forward_steps") or [],
            deterministic_backward=deterministic_hints.get("backward_steps") or [],
            top_k=max(1, self.candidate_top_k),
        )
        raw = self._call_model(
            prompt=prompt,
            system_prompt=build_screening_system_prompt(),
            max_tokens=self.screening_max_tokens,
            temperature=0.0,
        )
        payload = extract_json_block(raw)
        parsed = parse_screening(payload, top_k=max(1, self.candidate_top_k))

        if not parsed["forward_candidates"]:
            parsed["forward_candidates"] = [
                {"step": step, "score": 0.5, "reason": "deterministic_fallback"}
                for step in (deterministic_hints.get("forward_steps") or [])[: self.candidate_top_k]
            ]
        if not parsed["backward_candidates"]:
            parsed["backward_candidates"] = [
                {"step": step, "score": 0.5, "reason": "deterministic_fallback"}
                for step in (deterministic_hints.get("backward_steps") or [])[: self.candidate_top_k]
            ]
        parsed["selected_event_steps"] = [int(row["step"]) for row in selected_events]
        return parsed

    def _merge_candidates(
        self,
        *,
        screening: Dict[str, Any],
        deterministic_hints: Dict[str, Any],
    ) -> Dict[str, Any]:
        forward_rows = screening.get("forward_candidates") or []
        backward_rows = screening.get("backward_candidates") or []
        forward_steps = [int(row["step"]) for row in forward_rows if row.get("step") is not None]
        backward_steps = [int(row["step"]) for row in backward_rows if row.get("step") is not None]
        intersection = sorted(set(forward_steps).intersection(backward_steps))

        screen_score: Dict[int, float] = {}
        for row in forward_rows:
            step = normalize_step(row.get("step"))
            if step is None:
                continue
            screen_score[step] = max(float(screen_score.get(step, 0.0)), 0.60 * float(row.get("score", 0.0)))
        for row in backward_rows:
            step = normalize_step(row.get("step"))
            if step is None:
                continue
            screen_score[step] = max(float(screen_score.get(step, 0.0)), 0.40 * float(row.get("score", 0.0)))

        det_score = deterministic_hints.get("score_map") or {}
        union = set(forward_steps) | set(backward_steps) | set(deterministic_hints.get("forward_steps") or []) | set(
            deterministic_hints.get("backward_steps") or []
        )
        score_map: Dict[int, float] = {}
        for step in union:
            step_int = int(step)
            score_map[step_int] = round(
                self._clip(0.70 * float(screen_score.get(step_int, 0.0)) + 0.30 * float(det_score.get(step_int, 0.0))),
                4,
            )

        ranked = sorted(score_map.keys(), key=lambda s: (score_map[s], -s), reverse=True)
        if intersection:
            intersection_ranked = sorted(intersection, key=lambda s: (score_map.get(s, 0.0), -s), reverse=True)
            candidates = intersection_ranked + [step for step in ranked if step not in intersection_ranked]
        else:
            candidates = ranked

        if not candidates:
            candidates = [0]
        candidates = candidates[: max(1, self.candidate_top_k)]
        return {
            "forward_steps": forward_steps,
            "backward_steps": backward_steps,
            "intersection_steps": intersection,
            "candidate_steps": candidates,
            "candidate_score_map": score_map,
        }

    @staticmethod
    def _render_local_context(
        history: List[Dict[str, Any]],
        *,
        agent_field: str,
        center_step: int,
        radius: int,
        max_chars_per_step: int = 260,
    ) -> str:
        if not history:
            return ""
        start = max(0, center_step - max(0, radius))
        end = min(len(history) - 1, center_step + max(0, radius))
        lines = []
        for step in range(start, end + 1):
            row = history[step]
            agent = str(row.get(agent_field) or "Unknown Agent")
            content = " ".join(str(row.get("content", "")).split())
            if len(content) > max_chars_per_step:
                content = content[:max_chars_per_step].rstrip() + "..."
            lines.append(f"Step {step} - {agent}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_candidate_capsules(events: List[Dict[str, Any]]) -> Dict[int, str]:
        capsule_map: Dict[int, str] = {}
        for row in events:
            step = int(row.get("step", -1))
            if step < 0:
                continue
            capsule_map[step] = (
                f"Step {step} | agent={row.get('agent')} | action={row.get('action_type')} | "
                f"tool={row.get('tool_status')} | claim={row.get('claim_type')} | evidence={row.get('evidence')}"
            )
        return capsule_map

    @staticmethod
    def _parse_half_decision(
        raw: str,
        *,
        upper_steps: List[int],
        lower_steps: List[int],
        upper_score: float,
        lower_score: float,
    ) -> Dict[str, Any]:
        text = str(raw or "")
        lowered = text.lower()
        compact = lowered.strip()

        tag_match = re.search(r"<decision>\s*(upper|lower)\s*</decision>", lowered, flags=re.IGNORECASE)
        if tag_match:
            return {"decision": tag_match.group(1).lower(), "confidence": 0.82, "reason": "decision_tag", "parse_error": None}

        line_matches = re.findall(r"(?:^|\n)\s*decision\s*[:=]\s*(upper|lower)\b", lowered, flags=re.IGNORECASE)
        if line_matches:
            return {"decision": line_matches[-1].lower(), "confidence": 0.80, "reason": "decision_line", "parse_error": None}

        if compact in {"upper", "lower"}:
            return {"decision": compact, "confidence": 0.78, "reason": "single_token", "parse_error": None}

        terminal = lowered[-200:]
        terminal_tokens = re.findall(r"\b(upper|lower)\b", terminal)
        if terminal_tokens:
            return {
                "decision": terminal_tokens[-1].lower(),
                "confidence": 0.60,
                "reason": "terminal_keyword",
                "parse_error": "weak_parse",
            }

        step_mentions = [int(item) for item in re.findall(r"[Ss]tep\s*(\d+)", text)]
        for mentioned in reversed(step_mentions):
            if mentioned in upper_steps:
                return {"decision": "upper", "confidence": 0.52, "reason": "step_mention_upper", "parse_error": "weak_parse"}
            if mentioned in lower_steps:
                return {"decision": "lower", "confidence": 0.52, "reason": "step_mention_lower", "parse_error": "weak_parse"}

        # 回退：使用先验分数和 earliest-root-cause 偏置。
        if upper_score > lower_score:
            return {"decision": "upper", "confidence": 0.40, "reason": "score_fallback_upper", "parse_error": "unparsed_response"}
        if lower_score > upper_score:
            return {"decision": "lower", "confidence": 0.40, "reason": "score_fallback_lower", "parse_error": "unparsed_response"}
        return {"decision": "upper", "confidence": 0.34, "reason": "earliest_tie_break", "parse_error": "unparsed_response"}

    def _run_tournament(
        self,
        *,
        question: str,
        final_answer: str,
        ground_truth: str,
        events: List[Dict[str, Any]],
        candidate_steps: List[int],
        candidate_score_map: Dict[int, float],
    ) -> Dict[str, Any]:
        participants = sorted({int(step) for step in candidate_steps[: max(1, self.max_tournament_candidates)]})
        rounds: List[Dict[str, Any]] = []
        if not participants:
            return {"winner_step": 0, "rounds": rounds, "avg_confidence": 0.0}
        if len(participants) == 1:
            return {"winner_step": participants[0], "rounds": rounds, "avg_confidence": 1.0}

        capsule_map = self._build_candidate_capsules(events)
        round_confidences: List[float] = []
        round_index = 0
        while len(participants) > 1:
            split = (len(participants) + 1) // 2
            upper_steps = participants[:split]
            lower_steps = participants[split:]
            if not lower_steps:
                break

            upper_capsules = [capsule_map.get(step, f"Step {step}") for step in upper_steps]
            lower_capsules = [capsule_map.get(step, f"Step {step}") for step in lower_steps]
            gt_block = f"Reference Correct Task Answer: {ground_truth}\n" if str(ground_truth or "").strip() else ""
            prompt = (
                "You are BLADE-Localizer.\n"
                f"Problem: {question}\n"
                f"{gt_block}"
                f"Observed Final Answer from trajectory: {final_answer}\n\n"
                "Find which half is MORE LIKELY to contain the earliest root-cause step.\n"
                f"Upper half candidate steps: {upper_steps}\n"
                + "\n".join(upper_capsules)
                + "\n\n"
                f"Lower half candidate steps: {lower_steps}\n"
                + "\n".join(lower_capsules)
                + "\n\n"
                "Output format (strict):\n"
                "Decision: upper or lower\n"
                "Confidence: 0.0-1.0\n"
                "Reason: one short line\n"
            )
            raw = self._call_model(
                prompt=prompt,
                system_prompt=build_tournament_system_prompt(),
                max_tokens=self.tournament_max_tokens,
                temperature=0.0,
            )
            upper_score = sum(float(candidate_score_map.get(step, 0.0)) for step in upper_steps)
            lower_score = sum(float(candidate_score_map.get(step, 0.0)) for step in lower_steps)
            decision = self._parse_half_decision(
                raw,
                upper_steps=upper_steps,
                lower_steps=lower_steps,
                upper_score=upper_score,
                lower_score=lower_score,
            )
            if decision.get("parse_error") is not None:
                retry_prompt = (
                    f"{prompt}\n"
                    "Your previous output was not parseable.\n"
                    "Reply with ONLY one token: upper or lower."
                )
                retry_raw = self._call_model(
                    prompt=retry_prompt,
                    system_prompt="Return one token only: upper or lower.",
                    max_tokens=12,
                    temperature=0.0,
                )
                retry_decision = self._parse_half_decision(
                    retry_raw,
                    upper_steps=upper_steps,
                    lower_steps=lower_steps,
                    upper_score=upper_score,
                    lower_score=lower_score,
                )
                if retry_decision.get("parse_error") is None or float(retry_decision.get("confidence", 0.0)) >= float(
                    decision.get("confidence", 0.0)
                ):
                    decision = retry_decision
                    raw = retry_raw
            chosen_steps = upper_steps if decision["decision"] == "upper" else lower_steps
            round_confidences.append(float(decision["confidence"]))
            rounds.append(
                {
                    "round": round_index,
                    "participants": list(participants),
                    "upper_steps": upper_steps,
                    "lower_steps": lower_steps,
                    "decision": decision["decision"],
                    "confidence": decision["confidence"],
                    "reason": decision["reason"],
                    "parse_error": decision.get("parse_error"),
                    "raw_response": raw,
                }
            )
            participants = sorted(set(chosen_steps))
            round_index += 1

        return {
            "winner_step": int(participants[0]),
            "rounds": rounds,
            "avg_confidence": mean(round_confidences) if round_confidences else 0.0,
        }

    @staticmethod
    def _parse_pointwise_score(raw: str, *, fallback_score: float) -> Dict[str, Any]:
        text = str(raw or "")
        lowered = text.lower()

        score = None
        tag_match = re.search(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>", lowered, flags=re.IGNORECASE)
        if tag_match:
            score = float(tag_match.group(1))

        if score is None:
            line_hits = re.findall(
                r"(?:^|\n)\s*(?:score|likelihood|probability)\s*[:=]\s*([0-9]*\.?[0-9]+)",
                lowered,
                flags=re.IGNORECASE,
            )
            if line_hits:
                score = float(line_hits[-1])

        if score is None:
            floats = [float(item) for item in re.findall(r"\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b", lowered)]
            if floats:
                score = floats[-1]

        parse_error = None
        if score is None:
            score = float(fallback_score)
            parse_error = "unparsed_response"

        decision = None
        decision_match = re.search(r"(?:^|\n)\s*earliest\s*[:=]\s*(yes|no)", lowered, flags=re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).lower()
        elif " yes" in lowered or lowered.strip().endswith("yes"):
            decision = "yes"
        elif " no" in lowered or lowered.strip().endswith("no"):
            decision = "no"

        return {
            "score": max(0.0, min(1.0, float(score))),
            "decision": decision,
            "parse_error": parse_error,
            "raw_response": text,
        }

    def _run_pointwise_rerank(
        self,
        *,
        question: str,
        final_answer: str,
        ground_truth: str,
        history: List[Dict[str, Any]],
        agent_field: str,
        candidate_steps: List[int],
        candidate_score_map: Dict[int, float],
    ) -> Dict[str, Any]:
        participants = sorted({int(step) for step in candidate_steps[: max(1, self.max_tournament_candidates)]})
        if not participants:
            return {"winner_step": 0, "rounds": [], "avg_confidence": 0.0, "mode": "pointwise"}

        eval_steps = participants[: max(1, self.rerank_top_k)]
        rows: List[Dict[str, Any]] = []
        model_scores: List[float] = []

        for rank_idx, step in enumerate(eval_steps):
            prior = float(candidate_score_map.get(step, 0.0))
            context = self._render_local_context(
                history,
                agent_field=agent_field,
                center_step=step,
                radius=max(2, self.local_window + 1),
                max_chars_per_step=320,
            )
            candidate_agent = normalize_agent(history[step].get(agent_field)) if 0 <= step < len(history) else None
            gt_block = f"Reference Correct Task Answer: {ground_truth}\n" if str(ground_truth or "").strip() else ""
            prompt = (
                "You are BLADE-Reranker.\n"
                f"Problem: {question}\n"
                f"{gt_block}"
                f"Observed Final Answer from trajectory: {final_answer}\n"
                f"Candidate Step: {step}\n"
                f"Candidate Agent: {candidate_agent or 'unknown_agent'}\n"
                f"Deterministic Prior Score: {prior:.4f}\n\n"
                f"Local Context:\n{context}\n\n"
                "Evaluate whether this step is the earliest root-cause.\n"
                "Return exactly:\n"
                "Score: <0~1>\n"
                "Earliest: <yes/no>\n"
                "Reason: <one short line>\n"
            )
            raw = self._call_model(
                prompt=prompt,
                system_prompt="Assess one candidate step. Follow output format exactly.",
                max_tokens=self.rerank_max_tokens,
                temperature=0.0,
            )
            parsed = self._parse_pointwise_score(raw, fallback_score=prior)
            model_score = float(parsed["score"])
            model_scores.append(model_score)

            combined = 0.62 * model_score + 0.38 * prior
            if parsed.get("decision") == "yes":
                combined += 0.05
            elif parsed.get("decision") == "no":
                combined -= 0.05

            rows.append(
                {
                    "round": rank_idx,
                    "candidate_step": step,
                    "candidate_agent": candidate_agent,
                    "prior_score": round(prior, 4),
                    "model_score": round(model_score, 4),
                    "combined_score": round(self._clip(combined), 4),
                    "decision": parsed.get("decision"),
                    "parse_error": parsed.get("parse_error"),
                    "raw_response": raw,
                }
            )

        winner_row = max(rows, key=lambda item: (float(item["combined_score"]), -int(item["candidate_step"])))
        return {
            "winner_step": int(winner_row["candidate_step"]),
            "rounds": rows,
            "avg_confidence": mean(model_scores) if model_scores else 0.0,
            "mode": "pointwise",
        }

    @staticmethod
    def _is_terminal_agent(agent: str | None) -> bool:
        if not agent:
            return False
        normalized = normalize_agent(agent)
        return normalized in {"computer_terminal", "terminal", "computerterminal"}

    def _project_terminal_prediction(
        self,
        *,
        pred_agent: str | None,
        pred_step: int | None,
        history: List[Dict[str, Any]],
        agent_field: str,
    ) -> Dict[str, Any]:
        if pred_step is None or not history:
            return {
                "agent": pred_agent,
                "step": pred_step,
                "applied": False,
                "reason": "",
            }
        step = max(0, min(int(pred_step), len(history) - 1))
        current_agent = normalize_agent(history[step].get(agent_field))
        content = str(history[step].get("content", "")).lower()
        looks_execution = "exitcode:" in content or "execution failed" in content or "execution succeeded" in content

        if not (self._is_terminal_agent(pred_agent) or self._is_terminal_agent(current_agent) or looks_execution):
            return {"agent": pred_agent, "step": step, "applied": False, "reason": ""}

        for idx in range(step - 1, -1, -1):
            candidate_agent = normalize_agent(history[idx].get(agent_field))
            candidate_content = str(history[idx].get("content", "")).lower()
            if self._is_terminal_agent(candidate_agent):
                continue
            if "exitcode:" in candidate_content:
                continue
            if candidate_agent is None:
                continue
            return {
                "agent": candidate_agent,
                "step": idx,
                "applied": True,
                "reason": f"projected_from_terminal_step_{step}",
            }

        return {"agent": pred_agent, "step": step, "applied": False, "reason": ""}

    def _estimate_seed_confidence(
        self,
        *,
        seed_step: int,
        tournament_avg_conf: float,
        candidate_score_map: Dict[int, float],
    ) -> float:
        if not candidate_score_map:
            return self._clip(0.36 + 0.34 * float(tournament_avg_conf))

        normalized_scores = {int(step): float(score) for step, score in candidate_score_map.items()}
        ranking = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)
        top_score = ranking[0][1]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        seed_score = float(normalized_scores.get(seed_step, 0.0))
        margin = max(0.0, top_score - second_score)
        seed_gap = max(0.0, top_score - seed_score)

        conf = (
            0.28
            + 0.30 * self._clip(float(tournament_avg_conf))
            + 0.28 * self._clip(seed_score)
            + 0.22 * self._clip(margin)
            - 0.18 * self._clip(seed_gap)
        )
        return self._clip(conf)

    def _run_local_attribution(
        self,
        *,
        question: str,
        final_answer: str,
        ground_truth: str,
        history: List[Dict[str, Any]],
        agent_field: str,
        seed_step: int,
        tournament_avg_conf: float,
        candidate_score_map: Dict[int, float],
    ) -> Dict[str, Any]:
        local_context = self._render_local_context(
            history,
            agent_field=agent_field,
            center_step=seed_step,
            radius=max(2, self.local_window + 1),
            max_chars_per_step=300,
        )
        clipped_seed_step = max(0, min(seed_step, max(0, len(history) - 1)))
        seed_agent = normalize_agent(history[clipped_seed_step].get(agent_field)) if history else None
        seed_reason = f"Selected by BLADE localizer around step {clipped_seed_step}."
        seed_confidence = self._estimate_seed_confidence(
            seed_step=clipped_seed_step,
            tournament_avg_conf=tournament_avg_conf,
            candidate_score_map=candidate_score_map,
        )

        if not self.use_finalize_model:
            return {
                "mistake_agent": seed_agent,
                "mistake_step": clipped_seed_step,
                "reason": seed_reason,
                "confidence": seed_confidence,
                "raw_response": "",
                "local_context": local_context,
                "parse_error": "finalize_disabled",
            }

        prompt = build_finalize_prompt(
            question=question,
            final_answer=final_answer,
            ground_truth=ground_truth,
            focused_step=clipped_seed_step,
            local_context=local_context,
        )
        raw = self._call_model(
            prompt=prompt,
            system_prompt=build_finalize_system_prompt(),
            max_tokens=self.finalize_max_tokens,
            temperature=0.0,
        )
        parsed = parse_final_text(raw)
        parse_error = None

        pred_step = normalize_step(parsed["mistake_step"])
        if pred_step is None:
            pred_step = clipped_seed_step
            parse_error = "finalize_step_unparsed"
        pred_step = max(0, min(pred_step, max(0, len(history) - 1)))

        pred_agent = parsed["mistake_agent"]
        known_agents = {normalize_agent(row.get(agent_field)) for row in history}
        if pred_agent not in known_agents:
            pred_agent = None
        if pred_agent is None and history:
            pred_agent = normalize_agent(history[pred_step].get(agent_field))

        confidence = float(parsed.get("confidence", 0.0))
        if confidence <= 0.0:
            confidence = seed_confidence
            if parse_error is None:
                parse_error = "finalize_confidence_missing"
        else:
            # 融合局部模型置信度与赛程先验，减少单次输出波动。
            confidence = 0.65 * self._clip(confidence) + 0.35 * seed_confidence

        return {
            "mistake_agent": pred_agent,
            "mistake_step": pred_step,
            "reason": parsed["reason"] or seed_reason,
            "confidence": self._clip(confidence),
            "raw_response": raw,
            "local_context": local_context,
            "parse_error": parse_error,
        }

    def _run_escalation(
        self,
        *,
        question: str,
        final_answer: str,
        ground_truth: str,
        local_context: str,
        seed_step: int,
    ) -> Dict[str, Any]:
        if not self.enable_escalation or self.escalation_num_analysts <= 0:
            return {"analyses": [], "final": None}

        analyses: List[Dict[str, Any]] = []
        analyst_count = min(max(1, self.escalation_num_analysts), len(self.ESCALATION_ROLES))
        for idx in range(analyst_count):
            role = self.ESCALATION_ROLES[idx]
            prompt = build_escalation_prompt(
                question=question,
                final_answer=final_answer,
                ground_truth=ground_truth,
                local_context=local_context,
                seed_step=seed_step,
            )
            raw = self._call_model(
                prompt=prompt,
                system_prompt=build_escalation_system_prompt(role),
                max_tokens=self.escalation_max_tokens,
                temperature=max(0.0, min(0.3, self.temperature)),
            )
            payload = extract_json_block(raw)
            parsed = parse_escalation(payload)
            parsed["role"] = role
            analyses.append(parsed)

        vote_score: Dict[Tuple[str | None, int | None], float] = {}
        reason_by_key: Dict[Tuple[str | None, int | None], str] = {}
        for row in analyses:
            key = (row.get("agent"), row.get("step"))
            if key[1] is None and key[0] is None:
                continue
            vote_score[key] = float(vote_score.get(key, 0.0)) + float(row.get("confidence") or 0.0)
            if key not in reason_by_key and row.get("reason"):
                reason_by_key[key] = str(row["reason"])

        if not vote_score:
            return {"analyses": analyses, "final": None}

        best_key = max(vote_score.items(), key=lambda item: item[1])[0]
        avg_conf = vote_score[best_key] / max(1, analyst_count)
        final = {
            "mistake_agent": best_key[0],
            "mistake_step": best_key[1],
            "confidence": self._clip(avg_conf),
            "reason": reason_by_key.get(best_key, ""),
        }
        return {"analyses": analyses, "final": final}

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        sample = load_json(file_path)
        history = sample.get("history") or []
        question = str(sample.get("question", ""))
        gt_agent, gt_step = self._extract_ground_truth(sample)
        task_ground_truth = self._extract_answer(sample)
        prompt_ground_truth = task_ground_truth if self.use_ground_truth else ""
        final_answer = self._resolve_final_answer(sample, history)
        agent_field = self._resolve_agent_field(sample)

        events = self._eventize_history(history, agent_field=agent_field)
        deterministic_hints = self._deterministic_hints(events)
        if self.enable_screening_model:
            screening = self._run_screening(
                question=question,
                final_answer=final_answer,
                ground_truth=prompt_ground_truth,
                events=events,
                deterministic_hints=deterministic_hints,
            )
        else:
            screening = {
                "forward_candidates": [
                    {"step": step, "score": 0.5, "reason": "deterministic_only"}
                    for step in (deterministic_hints.get("forward_steps") or [])[: self.candidate_top_k]
                ],
                "backward_candidates": [
                    {"step": step, "score": 0.5, "reason": "deterministic_only"}
                    for step in (deterministic_hints.get("backward_steps") or [])[: self.candidate_top_k]
                ],
                "global_confidence": 0.0,
                "parse_error": "disabled_by_config",
                "raw_response": "",
                "selected_event_steps": [int(row["step"]) for row in events],
            }
        merged = self._merge_candidates(screening=screening, deterministic_hints=deterministic_hints)
        if self.use_pointwise_reranker:
            tournament = self._run_pointwise_rerank(
                question=question,
                final_answer=final_answer,
                ground_truth=prompt_ground_truth,
                history=history,
                agent_field=agent_field,
                candidate_steps=merged["candidate_steps"],
                candidate_score_map=merged["candidate_score_map"],
            )
        else:
            tournament = self._run_tournament(
                question=question,
                final_answer=final_answer,
                ground_truth=prompt_ground_truth,
                events=events,
                candidate_steps=merged["candidate_steps"],
                candidate_score_map=merged["candidate_score_map"],
            )

        seed_step = normalize_step(tournament.get("winner_step"))
        if seed_step is None:
            seed_step = int(merged["candidate_steps"][0]) if merged["candidate_steps"] else 0
        seed_step = max(0, min(seed_step, max(0, len(history) - 1)))

        local_result = self._run_local_attribution(
            question=question,
            final_answer=final_answer,
            ground_truth=prompt_ground_truth,
            history=history,
            agent_field=agent_field,
            seed_step=seed_step,
            tournament_avg_conf=float(tournament.get("avg_confidence", 0.0)),
            candidate_score_map=merged["candidate_score_map"],
        )
        final_result = dict(local_result)
        final_source = "local"

        escalation = {"analyses": [], "final": None}
        if self.enable_escalation and float(local_result.get("confidence", 0.0)) < self.uncertainty_threshold:
            escalation = self._run_escalation(
                question=question,
                final_answer=final_answer,
                ground_truth=prompt_ground_truth,
                local_context=local_result["local_context"],
                seed_step=seed_step,
            )
            escal_final = escalation.get("final")
            if isinstance(escal_final, dict):
                local_conf = float(local_result.get("confidence", 0.0))
                escal_conf = float(escal_final.get("confidence", 0.0))
                if escal_conf >= local_conf or final_result.get("mistake_agent") is None:
                    final_result["mistake_agent"] = normalize_agent(escal_final.get("mistake_agent"))
                    final_result["mistake_step"] = normalize_step(escal_final.get("mistake_step"))
                    final_result["confidence"] = escal_conf
                    if str(escal_final.get("reason") or "").strip():
                        final_result["reason"] = str(escal_final["reason"])
                    final_source = "escalation"

        pred_agent = normalize_agent(final_result.get("mistake_agent"))
        pred_step = normalize_step(final_result.get("mistake_step"))
        if pred_step is not None and history:
            pred_step = max(0, min(pred_step, len(history) - 1))
            if pred_agent is None:
                pred_agent = normalize_agent(history[pred_step].get(agent_field))
        if self.enable_terminal_projection:
            projection = self._project_terminal_prediction(
                pred_agent=pred_agent,
                pred_step=pred_step,
                history=history,
                agent_field=agent_field,
            )
        else:
            projection = {"agent": pred_agent, "step": pred_step, "applied": False, "reason": "disabled_by_config"}
        pred_agent = normalize_agent(projection.get("agent"))
        pred_step = normalize_step(projection.get("step"))
        if pred_step is not None and history:
            pred_step = max(0, min(pred_step, len(history) - 1))
            if pred_agent is None:
                pred_agent = normalize_agent(history[pred_step].get(agent_field))

        parse_errors: List[str] = []
        if screening.get("parse_error"):
            parse_errors.append(f"screening:{screening.get('parse_error')}")
        for idx, row in enumerate(tournament.get("rounds") or []):
            if row.get("parse_error"):
                parse_errors.append(f"tournament_r{idx}:{row.get('parse_error')}")
        if local_result.get("parse_error"):
            parse_errors.append(f"local:{local_result.get('parse_error')}")
        for idx, row in enumerate(escalation.get("analyses") or []):
            if row.get("parse_error"):
                parse_errors.append(f"escalation_{idx}:{row.get('parse_error')}")
        parse_error = ";".join(parse_errors) if parse_errors else None

        mode_parts = [
            "dual_screen" if self.enable_screening_model else "det_only",
            "pointwise_rank" if self.use_pointwise_reranker else "binary_localize",
            "finalize" if self.use_finalize_model else "seed_local",
        ]
        if self.enable_escalation:
            mode_parts.append("escalation")
        if self.enable_terminal_projection and projection.get("applied"):
            mode_parts.append("terminal_projection")

        acc_agent = int(pred_agent == gt_agent and pred_agent is not None)
        acc_step = int(pred_step == gt_step and pred_step is not None)

        return {
            "file": str(file_path),
            "question": question,
            "gt": {"agent": gt_agent, "step": gt_step},
            "pred": {"agent": pred_agent, "step": pred_step},
            "acc_agent": acc_agent,
            "acc_step": acc_step,
            "blade_events": events,
            "deterministic_hints": deterministic_hints,
            "screening": screening,
            "candidate_merge": merged,
            "tournament": tournament,
            "seed_step": seed_step,
            "local_attribution": local_result,
            "escalation": escalation,
            "final_source": final_source,
            "projection": projection,
            "final_pred": {
                "mistake_agent": pred_agent,
                "mistake_step": pred_step,
                "reason": str(final_result.get("reason") or ""),
                "confidence": self._clip(float(final_result.get("confidence", 0.0))),
            },
            "parse_error": parse_error,
            "request_mode": "+".join(mode_parts),
        }
