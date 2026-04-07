"""评估指标计算模块"""

from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class AttributionResult:
    """归因结果"""
    method: str
    predicted_agent: str
    predicted_step: int
    confidence: float
    token_cost: int
    latency_ms: float
    causal_chain: List[int] = None
    early_stopped: bool = False
    reasoning: str = ""

    def __post_init__(self):
        if self.causal_chain is None:
            self.causal_chain = []


def calculate_accuracy(predicted_agent: str, predicted_step: int,
                       ground_truth_agent: str, ground_truth_step: int,
                       tolerance: int = 0) -> Dict[str, int]:
    """计算归因准确率"""
    agent_hit = int(predicted_agent == ground_truth_agent)
    step_hit = int(abs(predicted_step - ground_truth_step) <= tolerance)

    return {
        "agent_acc": agent_hit,
        "step_acc": step_hit,
        "both_acc": agent_hit * step_hit
    }


def aggregate_metrics(results: List[Dict]) -> Dict:
    """聚合多个结果的指标"""
    if not results:
        return {}

    n = len(results)
    metrics = {
        "total_cases": n,
        "agent_accuracy": sum(r.get("agent_acc", 0) for r in results) / n,
        "step_accuracy": sum(r.get("step_acc", 0) for r in results) / n,
        "both_accuracy": sum(r.get("both_acc", 0) for r in results) / n,
        "avg_token_cost": sum(r.get("token_cost", 0) for r in results) / n,
        "avg_confidence": sum(r.get("confidence", 0) for r in results) / n,
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / n,
        "early_stop_rate": sum(r.get("early_stopped", 0) for r in results) / n,
    }

    return metrics
