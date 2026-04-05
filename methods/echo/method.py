"""ECHO: Error attribution through Contextual Hierarchy and Objective consensus."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.llm import chat_completion
from core.utils import load_json, normalize_agent, normalize_step
from methods.base import BaseMethod

from .context import build_conversation_summary, build_hierarchical_contexts
from .parsers import extract_json_block, normalize_objective_analysis
from .prompts import build_objective_prompt, build_objective_system_prompt
from .voting import aggregate_consensus


class ECHOMethod(BaseMethod):
    ANALYST_ROLES = [
        "conservative",
        "liberal",
        "detail_focused",
        "pattern_focused",
        "skeptical",
        "general",
    ]

    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        self.num_analysts = int(self.params.get("num_analysts", 6))
        self.min_confidence_threshold = float(self.params.get("min_confidence_threshold", 0.3))
        self.context_type = str(self.params.get("context_type", "decision_quality"))
        self.include_ground_truth = bool(self.params.get("use_ground_truth_in_prompt", True))
        self.max_summary_chars = int(self.params.get("max_summary_chars", 12000))

    def _call_model(self, *, prompt: str, system_prompt: str, temperature: float) -> str:
        result = chat_completion(
            self.client,
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        return result.content

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
    def _pick_single_agent(consensus: Dict[str, Any]) -> str | None:
        conclusion = consensus.get("consensus_conclusion") or {}
        attribution = conclusion.get("attribution") or []
        if attribution:
            return normalize_agent(attribution[0])
        summary = consensus.get("agent_evaluations_summary") or {}
        if not isinstance(summary, dict) or not summary:
            return None
        best = max(summary.items(), key=lambda item: float((item[1] or {}).get("avg_error_likelihood", 0.0)))
        return normalize_agent(best[0])

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        sample = load_json(file_path)
        history = sample.get("history") or []
        question = str(sample.get("question", ""))
        ground_truth = str(sample.get("ground_truth") or sample.get("answer") or "")
        final_answer = self._resolve_final_answer(sample, history)
        gt_agent = normalize_agent(sample.get("mistake_agent"))
        gt_step = normalize_step(sample.get("mistake_step"))

        contexts = build_hierarchical_contexts(history, context_type=self.context_type)
        summary = build_conversation_summary(contexts, history, max_chars=self.max_summary_chars)

        objective_analyses: List[Dict[str, Any]] = []
        analyst_count = max(1, self.num_analysts)
        for analyst_index in range(analyst_count):
            role = self.ANALYST_ROLES[analyst_index % len(self.ANALYST_ROLES)]
            system_prompt = build_objective_system_prompt(role)
            prompt = build_objective_prompt(
                query=question,
                ground_truth=ground_truth,
                final_answer=final_answer,
                context_summary=summary,
                include_ground_truth=self.include_ground_truth,
            )
            raw = self._call_model(prompt=prompt, system_prompt=system_prompt, temperature=self.temperature)
            parsed = extract_json_block(raw)
            normalized = normalize_objective_analysis(parsed, role, analyst_index)
            objective_analyses.append(normalized)

        consensus = aggregate_consensus(
            objective_analyses,
            min_confidence_threshold=self.min_confidence_threshold,
            conversation_history=history,
        )

        pred_agent = self._pick_single_agent(consensus)
        pred_step = normalize_step((consensus.get("consensus_conclusion") or {}).get("mistake_step"))
        acc_agent = int(pred_agent == gt_agent and pred_agent is not None)
        acc_step = int(pred_step == gt_step and pred_step is not None)

        return {
            "file": str(file_path),
            "question": question,
            "gt": {"agent": gt_agent, "step": gt_step},
            "pred": {"agent": pred_agent, "step": pred_step},
            "acc_agent": acc_agent,
            "acc_step": acc_step,
            "echo_contexts": contexts,
            "echo_summary": summary,
            "objective_analyses": objective_analyses,
            "consensus_result": consensus,
            "final_pred": {
                "mistake_agent": pred_agent,
                "mistake_step": pred_step,
                "reason": (consensus.get("consensus_conclusion") or {}).get("reasoning", ""),
            },
        }

