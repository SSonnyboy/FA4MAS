"""重构后的 CHIEF 方法。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.llm import chat_completion
from core.utils import load_json, normalize_agent, normalize_step
from methods.base import BaseMethod

from .parsers import (
    parse_agent_edges,
    parse_candidate_set,
    parse_final_prediction,
    parse_subtask_agents,
    parse_subtask_edges,
    parse_subtasks,
)
from .prompts import (
    build_agent_edge_prompt,
    build_agent_prompt,
    build_candidate_prompt,
    build_final_prompt,
    build_subtask_edge_prompt,
    build_subtask_prompt,
)
from .rag import OptionalRAGRetriever, build_rag_text


class CHIEFMethod(BaseMethod):
    """将旧版 CHIEF 流程收敛为一个状态清晰的类。"""

    def __init__(self, client, config) -> None:
        super().__init__(client, config)
        self.retriever = OptionalRAGRetriever()

    def call_model(self, prompt: str) -> str:
        # CHIEF 各阶段都共用同一套调用配置。
        return chat_completion(
            self.client,
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            system_prompt="You are a helpful assistant skilled in analyzing multi-agent conversations.",
        )

    @staticmethod
    def build_dag_graph(
        subtasks_agents: List[Dict[str, Any]],
        subtasks_edges: List[Dict[str, Any]],
        agent_edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # 图结构是后续候选集筛选和最终归因的统一中间表示。
        return {
            "subtasks": subtasks_agents,
            "subtask_edges": subtasks_edges,
            "agent_edges": agent_edges,
        }

    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        sample = load_json(file_path)
        history = sample.get("history", [])
        question = str(sample.get("question", ""))
        ground_truth = str(sample.get("ground_truth", ""))
        gt_agent = normalize_agent(sample.get("mistake_agent"))
        gt_step = normalize_step(sample.get("mistake_step"))

        # Step1: 结合可选 RAG，将原始轨迹切成子任务。
        rag_results = self.retriever.search(question, top_k=2)
        rag_text = build_rag_text(rag_results)
        step1_raw = self.call_model(build_subtask_prompt(history, question, ground_truth, rag_text))
        subtasks = parse_subtasks(step1_raw)

        # Step2: 构造子任务之间的依赖边。
        step2_raw = self.call_model(build_subtask_edge_prompt(history, question, ground_truth, subtasks))
        subtasks_edges = parse_subtask_edges(step2_raw)

        # Step3: 在每个子任务内部抽出 agent 和细粒度数据流。
        step3_raw = self.call_model(build_agent_prompt(history, question, ground_truth, subtasks))
        subtasks_agents = parse_subtask_agents(step3_raw, subtasks)

        # Step4: 构造子任务内部的 agent 因果边。
        step4_raw = self.call_model(build_agent_edge_prompt(history, question, ground_truth, subtasks_agents))
        agent_edges = parse_agent_edges(step4_raw)

        dag_graph = self.build_dag_graph(subtasks_agents, subtasks_edges, agent_edges)

        # Step5: 先缩小到候选责任集合，再做最终判断。
        step5_raw = self.call_model(build_candidate_prompt(history, question, ground_truth, dag_graph))
        candidate_set = parse_candidate_set(step5_raw)

        # Step6: 在候选集合上输出唯一责任 agent 和 step。
        step6_raw = self.call_model(build_final_prompt(history, question, ground_truth, candidate_set, dag_graph))
        final_prediction = parse_final_prediction(step6_raw)

        pred_agent = normalize_agent(final_prediction.get("mistake_agent"))
        pred_step = normalize_step(final_prediction.get("mistake_step"))
        acc_agent = int(pred_agent == gt_agent and pred_agent is not None)
        acc_step = int(pred_step == gt_step and pred_step is not None)

        return {
            "file": str(file_path),
            "question": question,
            "gt": {"agent": gt_agent, "step": gt_step},
            "pred": {"agent": pred_agent, "step": pred_step},
            "acc_agent": acc_agent,
            "acc_step": acc_step,
            "rag_results": rag_results,
            "step1_raw": step1_raw,
            "step2_raw": step2_raw,
            "step3_raw": step3_raw,
            "step4_raw": step4_raw,
            "step5_raw": step5_raw,
            "step6_raw": step6_raw,
            "subtasks": subtasks,
            "subtask_edges": subtasks_edges,
            "subtasks_agents": subtasks_agents,
            "agent_edges": agent_edges,
            "dag_graph": dag_graph,
            "candidate_set": candidate_set,
            "final_pred": final_prediction,
        }
