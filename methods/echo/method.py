"""ECHO: Error attribution through Contextual Hierarchy and Objective consensus."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List

from core.llm import chat_completion
from core.utils import load_json, normalize_agent, normalize_step
from methods.base import BaseMethod

from .context import build_conversation_summary, build_hierarchical_contexts
from .parsers import extract_json_block, normalize_objective_analysis
from .prompts import build_objective_prompt, build_objective_system_prompt
from .voting import aggregate_consensus, aggregate_decoupled_consensus


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
        self.random_sample_analysts = bool(self.params.get("random_sample_analysts", False))
        self.analyst_seed = self.params.get("analyst_seed")
        self.temperature_strategy = str(self.params.get("temperature_strategy", "fixed"))
        self.temperature_min = float(self.params.get("temperature_min", 0.3))
        self.temperature_max = float(self.params.get("temperature_max", 0.9))
        self.temperature_values = self.params.get("temperature_values")
        self.decoupled_attribution = bool(self.params.get("decoupled_attribution", True))

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

    def _build_rng(self, *, index: int) -> random.Random:
        if self.analyst_seed is None:
            return random.Random()
        try:
            base = int(self.analyst_seed)
        except Exception:
            base = abs(hash(str(self.analyst_seed))) % (2**31)
        return random.Random(base + int(index))

    def _select_analyst_roles(self, *, index: int) -> List[str]:
        pool = list(self.ANALYST_ROLES)
        target = max(1, self.num_analysts)
        if not self.random_sample_analysts:
            return [pool[i % len(pool)] for i in range(target)]

        rng = self._build_rng(index=index)
        if target >= len(pool):
            sampled = list(pool)
            rng.shuffle(sampled)
            return sampled
        return rng.sample(pool, target)

    @staticmethod
    def _linspace(min_value: float, max_value: float, count: int) -> List[float]:
        if count <= 1:
            return [min_value]
        step = (max_value - min_value) / float(count - 1)
        return [min_value + step * i for i in range(count)]

    def _select_temperatures(self, *, index: int, analyst_count: int) -> List[float]:
        count = max(1, analyst_count)
        strategy = self.temperature_strategy.lower()

        if strategy == "list":
            values = self.temperature_values if isinstance(self.temperature_values, list) else []
            parsed: List[float] = []
            for row in values:
                try:
                    parsed.append(float(row))
                except Exception:
                    continue
            if not parsed:
                parsed = [self.temperature]
            if len(parsed) >= count:
                return [max(0.0, min(1.0, parsed[i])) for i in range(count)]
            repeated = [parsed[i % len(parsed)] for i in range(count)]
            return [max(0.0, min(1.0, value)) for value in repeated]

        if strategy == "linspace":
            seq = self._linspace(self.temperature_min, self.temperature_max, count)
            return [max(0.0, min(1.0, value)) for value in seq]

        if strategy == "random_uniform":
            low = min(self.temperature_min, self.temperature_max)
            high = max(self.temperature_min, self.temperature_max)
            rng = self._build_rng(index=index)
            seq = [rng.uniform(low, high) for _ in range(count)]
            return [max(0.0, min(1.0, value)) for value in seq]

        # fixed
        return [max(0.0, min(1.0, float(self.temperature))) for _ in range(count)]

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
        objective_analyses_agent: List[Dict[str, Any]] = []
        objective_analyses_step: List[Dict[str, Any]] = []
        analyst_roles = self._select_analyst_roles(index=index)
        analyst_temperatures = self._select_temperatures(index=index, analyst_count=len(analyst_roles))
        analyst_count = len(analyst_roles)
        for analyst_index in range(analyst_count):
            role = analyst_roles[analyst_index]
            analyst_temperature = analyst_temperatures[analyst_index]
            agent_system_prompt = build_objective_system_prompt(role, attribution_target="agent")
            agent_prompt = build_objective_prompt(
                query=question,
                ground_truth=ground_truth,
                final_answer=final_answer,
                context_summary=summary,
                include_ground_truth=self.include_ground_truth,
                attribution_target="agent",
            )
            agent_raw = self._call_model(
                prompt=agent_prompt,
                system_prompt=agent_system_prompt,
                temperature=analyst_temperature,
            )
            agent_parsed = extract_json_block(agent_raw)
            agent_analysis = normalize_objective_analysis(agent_parsed, role, analyst_index)
            agent_analysis["temperature"] = analyst_temperature
            agent_analysis["attribution_target"] = "agent"
            objective_analyses_agent.append(agent_analysis)

            if self.decoupled_attribution:
                step_system_prompt = build_objective_system_prompt(role, attribution_target="step")
                step_prompt = build_objective_prompt(
                    query=question,
                    ground_truth=ground_truth,
                    final_answer=final_answer,
                    context_summary=summary,
                    include_ground_truth=self.include_ground_truth,
                    attribution_target="step",
                )
                step_raw = self._call_model(
                    prompt=step_prompt,
                    system_prompt=step_system_prompt,
                    temperature=analyst_temperature,
                )
                step_parsed = extract_json_block(step_raw)
                step_analysis = normalize_objective_analysis(step_parsed, role, analyst_index)
                step_analysis["temperature"] = analyst_temperature
                step_analysis["attribution_target"] = "step"
                objective_analyses_step.append(step_analysis)
                objective_analyses.append(
                    {
                        "analyst_id": analyst_index,
                        "analyst_role": role,
                        "temperature": analyst_temperature,
                        "agent_phase": agent_analysis,
                        "step_phase": step_analysis,
                    }
                )
            else:
                objective_analyses.append(agent_analysis)

        if self.decoupled_attribution:
            consensus = aggregate_decoupled_consensus(
                objective_analyses_agent,
                objective_analyses_step,
                min_confidence_threshold=self.min_confidence_threshold,
                conversation_history=history,
            )
        else:
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
            "analyst_plan": {
                "roles": analyst_roles,
                "temperatures": analyst_temperatures,
                "random_sample_analysts": self.random_sample_analysts,
                "temperature_strategy": self.temperature_strategy,
                "temperature_range": [self.temperature_min, self.temperature_max],
                "decoupled_attribution": self.decoupled_attribution,
            },
            "objective_analyses": objective_analyses,
            "objective_analyses_agent": objective_analyses_agent,
            "objective_analyses_step": objective_analyses_step,
            "consensus_result": consensus,
            "final_pred": {
                "mistake_agent": pred_agent,
                "mistake_step": pred_step,
                "reason": (consensus.get("consensus_conclusion") or {}).get("reasoning", ""),
            },
        }
