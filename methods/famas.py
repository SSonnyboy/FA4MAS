# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/31 20:04 
'''
# -*- coding: UTF-8 -*-
import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple

from data.dataset import get_log_steps
from methods.base_method import BaseAttributionMethod

# ════════════════════════════════════════════════════════
# 提示词配置
# ════════════════════════════════════════════════════════
ABSTRACTION_SYSTEM = "You are an expert at analyzing multi-agent system execution logs. Your task is to extract structured information from conversation logs."

ABSTRACTION_PROMPT = """Given the following segment of a multi-agent system execution log, extract each agent action as a structured triple.
For each step in the log segment, output a JSON array where each element has:
- "agent": the name/role of the agent performing the action
- "action": a concise description of the concrete action taken (what the agent did, 2-8 words)
- "state": a concise description of the resulting system state after the action (2-8 words)
- "step_index": the original step index in the conversation (integer)

Rules: Focus on meaningful actions. Use consistent terms. Keep descriptions concise. Output ONLY a valid JSON array.
Log segment:
{chunk}"""

AGENT_CLUSTER_PROMPT = """Identify and group names that refer to the same agent.
Agent names: {agent_names}
Output a JSON object where keys are canonical agent names and values are lists of variant names mapping to it. Output ONLY valid JSON.
Example: {{"WebSurfer": ["WebSurfer", "web_agent"], "Orchestrator": ["Orch", "orchestrator"]}}"""

ACTION_STATE_CLUSTER_PROMPT = """Group semantically equivalent action-state pairs together.
Action-state pairs (indexed):
{pairs}
Output a JSON object where keys are canonical action-state descriptions (format: "action | state") and values are lists of original indices. Output ONLY valid JSON."""


# ════════════════════════════════════════════════════════
# Famas 方法主类
# ════════════════════════════════════════════════════════
class FamasMethod(BaseAttributionMethod):
    def __init__(self, llm_client, lam: float = 0.9, chunk_size: int = 10):
        super().__init__(llm_client)
        self.lam = lam
        self.chunk_size = chunk_size

    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        # 1. 提取所有轨迹 (Main failed history + Replayed histories)
        failed_steps = get_log_steps(instance)
        if not failed_steps:
            return {"responsible_agent": "", "error_step": -1, "reason": "fallback: no structured steps"}

        all_steps_list = [failed_steps]
        outcomes = [0]  # 0 表示失败

        for r in instance.get("replayed_histories", []):
            r_steps = get_log_steps({"history": r.get("history", [])})  # 复用清洗逻辑
            if r_steps:
                all_steps_list.append(r_steps)
                outcomes.append(r.get("success", 0))

        if verbose: print(f"\n[Famas] 提取到 {len(all_steps_list)} 条轨迹 (1 失败, {len(all_steps_list) - 1} 重放)")

        # 2. Phase 1: 轨迹抽象与聚类
        refined_trajs = self._phase1_full(all_steps_list, verbose)
        if not refined_trajs or not refined_trajs[0]:
            return {"responsible_agent": "", "error_step": -1, "reason": "phase1 abstraction failed"}

        # 3. Phase 2: 频谱分析打分
        if verbose: print("[Famas Phase 2] 执行 SBFL 频谱分析计算...")
        ranked = self._phase2_spectrum_analysis(refined_trajs, outcomes, failed_traj_idx=0)

        if not ranked:
            return {"responsible_agent": "", "error_step": -1, "reason": "no candidates from spectrum analysis"}

        top_action_state, top_agent, top_step_idx, top_score = ranked[0]

        # 组装理由
        reason_lines = [f"Famas(λ={self.lam}) Top-3 Suspicious:"]
        for rank, (as_, ag_, si_, sc_) in enumerate(ranked[:3], 1):
            reason_lines.append(f" #{rank} [Score: {sc_:.4f}] Step {si_} ({ag_}): {as_}")

        if verbose: print(f"  -> 锁定真凶: Agent={top_agent}, Step={top_step_idx}, Score={top_score:.4f}")

        return {
            "responsible_agent": top_agent,
            "error_step": top_step_idx,
            "reason": "\n".join(reason_lines),
            "_famas_details": {
                "top_score": top_score,
                "top_action_state": top_action_state
            }
        }

    # ── Phase 1: Trajectory Replay & Abstraction ──────────────────
    def _phase1_full(self, all_steps_list: List[List[Tuple[int, str, str]]], verbose: bool) -> List[List[Dict]]:
        if verbose: print("[Famas Phase 1.1] LLM 提取 Primitive Triples...")
        all_primitive = []
        for i, steps in enumerate(all_steps_list):
            if verbose: print(f"  -> 正在处理轨迹 {i}/{len(all_steps_list) - 1} ({len(steps)} steps)")
            all_primitive.append(self._parse_primitive_trajectory(steps))

        all_triples_flat = [t for traj in all_primitive for t in traj]

        if verbose: print("[Famas Phase 1.2] Hierarchical Clustering (Agent & Action-State)...")
        agent_mapping = self._cluster_agents(all_triples_flat)
        action_state_mapping = self._cluster_action_states(all_triples_flat)

        # 组装 Refined Trajectories
        refined_trajectories = []
        flat_idx = 0
        for primitives in all_primitive:
            traj_as_mapping = {}
            for local_idx in range(len(primitives)):
                traj_as_mapping[local_idx] = action_state_mapping.get(flat_idx, "")
                flat_idx += 1

            refined_trajectories.append(self._refine_trajectory(primitives, agent_mapping, traj_as_mapping))

        return refined_trajectories

    def _parse_primitive_trajectory(self, steps: List[Tuple[int, str, str]]) -> List[Dict]:
        all_triples = []
        chunks = [steps[i:i + self.chunk_size] for i in range(0, len(steps), self.chunk_size)]

        for chunk in chunks:
            chunk_text = "\n".join(f"Step {s[0]} | {s[1]}: {s[2][:800]}" for s in chunk)
            raw = self.llm.call(ABSTRACTION_SYSTEM, ABSTRACTION_PROMPT.format(chunk=chunk_text), max_tokens=1024,
                                temperature=0.2)
            parsed = self.llm.parse_json(raw)

            if isinstance(parsed, list):
                for t in parsed:
                    if isinstance(t, dict) and all(k in t for k in ("agent", "action", "state")):
                        t.setdefault("step_index", chunk[0][0])
                        all_triples.append(t)
            else:
                # Fallback: JSON解析彻底失败时，用原始信息兜底
                for s in chunk:
                    all_triples.append({"agent": s[1], "action": s[2][:50], "state": "unknown", "step_index": s[0]})
        return all_triples

    def _cluster_agents(self, triples: List[Dict]) -> Dict[str, str]:
        agent_names = list(set(t["agent"] for t in triples))
        if len(agent_names) <= 1: return {n: n for n in agent_names}

        raw = self.llm.call(ABSTRACTION_SYSTEM, AGENT_CLUSTER_PROMPT.format(agent_names=json.dumps(agent_names)),
                            max_tokens=512, temperature=0.1)
        parsed = self.llm.parse_json(raw)

        mapping = {name: name for name in agent_names}
        if isinstance(parsed, dict):
            for canonical, variants in parsed.items():
                if isinstance(variants, list):
                    for v in variants: mapping[v] = canonical
        return mapping

    def _cluster_action_states(self, triples: List[Dict], batch_size: int = 30) -> Dict[int, str]:
        if not triples: return {}
        index_to_canonical = {i: f"{t['action']} | {t['state']}" for i, t in enumerate(triples)}  # 默认映射

        pairs_text = [f"{i}: {t['action']} | {t['state']}" for i, t in enumerate(triples)]
        for batch_start in range(0, len(pairs_text), batch_size):
            batch_str = "\n".join(pairs_text[batch_start: batch_start + batch_size])
            raw = self.llm.call(ABSTRACTION_SYSTEM, ACTION_STATE_CLUSTER_PROMPT.format(pairs=batch_str),
                                max_tokens=1024, temperature=0.1)
            parsed = self.llm.parse_json(raw)

            if isinstance(parsed, dict):
                for canonical_key, indices in parsed.items():
                    if isinstance(indices, list):
                        for idx in indices:
                            abs_idx = batch_start + int(idx)
                            if abs_idx < len(triples): index_to_canonical[abs_idx] = canonical_key
        return index_to_canonical

    def _refine_trajectory(self, primitives: List[Dict], agent_mapping: Dict[str, str], as_mapping: Dict[int, str]) -> \
    List[Dict]:
        return [{"agent": agent_mapping.get(t["agent"], t["agent"]),
                 "action_state": as_mapping.get(i, f"{t['action']} | {t['state']}"),
                 "step_index": t["step_index"]} for i, t in enumerate(primitives)]

    # ── Phase 2: Spectrum Analysis ────────────────────────────────
    def _phase2_spectrum_analysis(self, refined_trajs: List[List[Dict]], outcomes: List[int], failed_traj_idx: int) -> \
    List[Tuple[str, str, int, float]]:
        # 1. 构造频率矩阵
        action_state_freq = defaultdict(lambda: defaultdict(int))
        agent_freq = defaultdict(lambda: defaultdict(int))
        for t_idx, traj in enumerate(refined_trajs):
            for triple in traj:
                action_state_freq[triple["action_state"]][t_idx] += 1
                agent_freq[triple["agent"]][t_idx] += 1

        # 2. 锁定失败轨迹中的候选人并计算分数
        candidates = {}
        for triple in refined_trajs[failed_traj_idx]:
            eta, agent, step = triple["action_state"], triple["agent"], triple["step_index"]
            if eta not in candidates:  # 保留首次出现的步骤
                candidates[eta] = (agent, step)

        ranked = []
        for eta, (agent, step_idx) in candidates.items():
            score = self._compute_suspiciousness(eta, agent, action_state_freq, agent_freq, outcomes, failed_traj_idx)
            ranked.append((eta, agent, step_idx, score))

        return sorted(ranked, key=lambda x: x[3], reverse=True)

    def _compute_suspiciousness(self, eta: str, agent: str, as_freq: dict, agt_freq: dict, outcomes: List[int],
                                failed_idx: int) -> float:
        eta_f, agt_f = as_freq.get(eta, {}), agt_freq.get(agent, {})

        # γ (Action Coverage Ratio) & β (Action Frequency Proportion)
        nc_eta, nc_agent = sum(1 for v in eta_f.values() if v > 0), sum(1 for v in agt_f.values() if v > 0)
        gamma = nc_eta / nc_agent if nc_agent > 0 else 0.0
        beta = sum(eta_f.values()) / sum(agt_f.values()) if sum(agt_f.values()) > 0 else 0.0

        # α (Local Frequency Enhancement)
        f_ij = eta_f.get(failed_idx, 0)
        alpha = 1.0 + math.log(f_ij, 1.0 / self.lam) if (f_ij > 0 and 0 < self.lam < 1) else 1.0

        # Kulczynski2λ
        n_cf, n_cs = 0.0, 0.0
        for t_idx, count in eta_f.items():
            if count > 0:
                decay = self.lam ** (count - 1)
                if outcomes[t_idx] == 0:
                    n_cf += decay
                else:
                    n_cs += decay

        n_fail = sum(1 for o in outcomes if o == 0)
        n_uf = n_fail - sum(1 for t_idx, c in eta_f.items() if c > 0 and outcomes[t_idx] == 0)

        kulc = 0.5 * (
                    (n_cf / (n_cf + n_uf) if n_cf + n_uf > 0 else 0) + (n_cf / (n_cf + n_cs) if n_cf + n_cs > 0 else 0))

        return (alpha * kulc) * (1.0 + beta) * (1.0 + gamma)