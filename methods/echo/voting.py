"""ECHO 共识投票。"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def _synthesize_reasoning(votes: List[Dict[str, Any]], avg_confidence: float) -> str:
    if not votes:
        return "No clear consensus reached."
    reasonings = [str(item.get("reasoning") or "") for item in votes if item.get("reasoning")]
    if not reasonings:
        return f"Consensus reached by {len(votes)} analysts with average confidence {avg_confidence:.2f}."
    return (
        f"Consensus reached by {len(votes)} analysts (avg confidence {avg_confidence:.2f}). "
        f"Primary reasoning: {reasonings[0][:220]}"
    )


def _analyze_disagreements(conclusion_votes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    all_confidences: List[float] = []
    for votes in conclusion_votes.values():
        all_confidences.extend(float(vote.get("confidence", 0.0)) for vote in votes)
    spread = max(all_confidences) - min(all_confidences) if all_confidences else 0.0
    num_types = len(conclusion_votes)
    high_disagreement = num_types > 2 and all(len(votes) > 0 for votes in conclusion_votes.values())
    return {
        "high_disagreement": high_disagreement,
        "num_different_conclusions": num_types,
        "confidence_spread": spread,
        "requires_review": high_disagreement or spread > 0.5,
    }


def aggregate_consensus(
    objective_analyses: List[Dict[str, Any]],
    *,
    min_confidence_threshold: float,
    conversation_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not objective_analyses:
        return {
            "consensus_conclusion": {
                "type": "single_agent",
                "attribution": [],
                "mistake_step": None,
                "confidence": 0.0,
                "reasoning": "No objective analyses provided.",
            },
            "voting_details": {
                "conclusion_votes": {},
                "step_votes": {},
                "best_weighted_score": 0.0,
                "disagreement_analysis": {
                    "high_disagreement": False,
                    "num_different_conclusions": 0,
                    "confidence_spread": 0.0,
                    "requires_review": True,
                },
            },
            "agent_evaluations_summary": {},
            "alternative_hypotheses": [],
            "num_analysts": 0,
        }

    primary_conclusions: List[Dict[str, Any]] = []
    all_agent_evaluations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_alternative_hypotheses: List[Dict[str, Any]] = []

    for idx, analysis in enumerate(objective_analyses):
        primary = dict(analysis.get("primary_conclusion") or {})
        primary["analyst_id"] = idx
        primary_conclusions.append(primary)
        for row in analysis.get("agent_evaluations") or []:
            name = row.get("agent_name")
            if not name:
                continue
            all_agent_evaluations[str(name)].append(
                {
                    "error_likelihood": float(row.get("error_likelihood") or 0.0),
                    "reasoning": str(row.get("reasoning") or ""),
                    "evidence": str(row.get("evidence") or ""),
                    "analyst_id": idx,
                }
            )
        for alt in analysis.get("alternative_hypotheses") or []:
            payload = dict(alt)
            payload["analyst_id"] = idx
            all_alternative_hypotheses.append(payload)

    conclusion_votes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for conclusion in primary_conclusions:
        confidence = float(conclusion.get("confidence") or 0.0)
        if confidence < min_confidence_threshold:
            continue
        conclusion_type = str(conclusion.get("type") or "single_agent")
        conclusion_votes[conclusion_type].append(
            {
                "confidence": confidence,
                "attribution": conclusion.get("attribution") or [],
                "mistake_step": conclusion.get("mistake_step"),
                "reasoning": str(conclusion.get("reasoning") or ""),
                "analyst_id": conclusion.get("analyst_id"),
            }
        )

    best_type = None
    best_info = None
    best_weighted_score = 0.0
    for conclusion_type, votes in conclusion_votes.items():
        total_confidence = sum(float(vote["confidence"]) for vote in votes)
        avg_confidence = total_confidence / len(votes) if votes else 0.0
        if total_confidence > best_weighted_score:
            best_weighted_score = total_confidence
            best_type = conclusion_type
            best_info = {"votes": votes, "avg_confidence": avg_confidence, "total_confidence": total_confidence}

    attribution_votes: Dict[str, float] = defaultdict(float)
    step_votes: Dict[int, float] = defaultdict(float)
    if best_info:
        for vote in best_info["votes"]:
            for agent in vote.get("attribution") or []:
                attribution_votes[str(agent)] += float(vote["confidence"])
            step = vote.get("mistake_step")
            if isinstance(step, int):
                step_votes[step] += float(vote["confidence"])

    sorted_agents = sorted(attribution_votes.items(), key=lambda item: item[1], reverse=True)
    if best_type == "single_agent":
        final_attribution = [sorted_agents[0][0]] if sorted_agents else []
    else:
        final_attribution = [agent for agent, score in sorted_agents if score >= min_confidence_threshold]

    validated_steps = [(step, score) for step, score in step_votes.items() if 0 <= step < len(conversation_history)]
    validated_steps.sort(key=lambda item: item[1], reverse=True)
    final_step = validated_steps[0][0] if validated_steps else None

    aggregated_agent = {}
    for name, evaluations in all_agent_evaluations.items():
        scores = [float(row["error_likelihood"]) for row in evaluations]
        aggregated_agent[name] = {
            "avg_error_likelihood": (sum(scores) / len(scores)) if scores else 0.0,
            "num_evaluations": len(evaluations),
            "evaluations": evaluations,
        }

    avg_conf = float(best_info["avg_confidence"]) if best_info else 0.0
    return {
        "consensus_conclusion": {
            "type": best_type or "single_agent",
            "attribution": final_attribution,
            "mistake_step": final_step,
            "confidence": avg_conf,
            "reasoning": _synthesize_reasoning(best_info["votes"], avg_conf) if best_info else "No clear consensus reached.",
        },
        "voting_details": {
            "conclusion_votes": dict(conclusion_votes),
            "step_votes": dict(step_votes),
            "best_weighted_score": best_weighted_score,
            "disagreement_analysis": _analyze_disagreements(conclusion_votes),
        },
        "agent_evaluations_summary": aggregated_agent,
        "alternative_hypotheses": all_alternative_hypotheses[:5],
        "num_analysts": len(objective_analyses),
    }

