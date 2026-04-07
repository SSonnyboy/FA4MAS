# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/27 13:52 
'''
"""
Famas: Spectrum-Based Failure Attribution for Multi-Agent Systems
复现论文: "Who is Introducing the Failure? Automatically Attributing Failures of
Multi-Agent Systems via Spectrum Analysis" (arXiv:2509.13782)

两阶段方法:
  Phase 1: Trajectory Replay & Abstraction
    1.1 Primitive Trajectory Replay & Abstraction  (LLM解析log -> <AGENT, ACTION, STATE> triples)
    1.2 Hierarchical Clustering & Trace Refinement (agent聚类 + action-state语义聚类)
  Phase 2: Spectrum Analysis
    2.1 Agent Behavior Group: γ (Action Coverage Ratio), β (Action Frequency Proportion)
    2.2 Action Behavior Group: α (Local Frequency Enhancement), λ-Decay SBFL (Kulczynski2)
    最终 S(ηj) = [α · Kulczynski2λ] · [1+β] · [1+γ]

注意:
  - 本脚本假设数据集 JSON 中已包含多条轨迹 (replayed_histories 字段)，
    若无该字段则退化为仅用原始 history (k=0 额外轨迹)。
  - 若数据集仅有单条 history，Famas 自动降级为使用 LLM 直接做 all-at-once 分析
    并给出提示。
  - 结果实时写入 jsonl 文件，与 baseline 保持一致。
"""

import os
import json
import argparse
import re
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from openai import OpenAI

# ════════════════════════════════════════════════════════
# 1.  LLM 调用层 (与 baseline 保持一致)
# ════════════════════════════════════════════════════════

class LLMClient:
    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        self.model = model
        kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "sk-xxx")}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def chat(self, system: str, user: str, max_tokens: int = 1024,
             temperature: float = 0.6) -> str:
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"    [LLM Error attempt {attempt+1}] {e}")
                time.sleep(2 ** attempt)
        return ""


# ════════════════════════════════════════════════════════
# 2.  数据加载
# ════════════════════════════════════════════════════════

def load_dataset(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    instances = []
    json_files = sorted(
        [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".json"],
        key=lambda f: int("".join(filter(str.isdigit, f.stem)) or 0),
    )
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        data["task_id"] = jf.stem
        instances.append(data)
    return instances


# ════════════════════════════════════════════════════════
# 3.  Phase 1.1: Primitive Trajectory Abstraction
#     将单条 history (log) 解析为 <AGENT, ACTION, STATE> triples
# ════════════════════════════════════════════════════════

ABSTRACTION_SYSTEM = (
    "You are an expert at analyzing multi-agent system execution logs. "
    "Your task is to extract structured information from conversation logs."
)

ABSTRACTION_PROMPT = """
Given the following segment of a multi-agent system execution log, extract each agent action as a structured triple.

For each step in the log segment, output a JSON array where each element has:
- "agent": the name/role of the agent performing the action
- "action": a concise description of the concrete action taken (what the agent did, 2-8 words)
- "state": a concise description of the resulting system state after the action (2-8 words)
- "step_index": the original step index in the conversation (0-based)

Rules:
1. Focus on meaningful actions that change system state
2. Use consistent, canonical terms for similar actions (e.g., always "web_search" not "search the web")
3. Keep action and state descriptions concise and normalized
4. Output ONLY valid JSON array, no markdown, no extra text

Log segment:
{chunk}

Output JSON array:
"""

def chunk_history(history: List[Dict], chunk_size: int = 10) -> List[List[Tuple[int, Dict]]]:
    """将 history 分块，每块包含 (原始index, entry) 对"""
    chunks = []
    for i in range(0, len(history), chunk_size):
        chunk = [(idx, history[idx]) for idx in range(i, min(i + chunk_size, len(history)))]
        chunks.append(chunk)
    return chunks

def format_chunk(chunk: List[Tuple[int, Dict]]) -> str:
    lines = []
    for idx, entry in chunk:
        name = entry.get("name", entry.get("role", "Unknown"))
        content = entry.get("content", "")
        # 截断超长内容，避免 token 爆炸
        if len(content) > 800:
            content = content[:800] + "...[truncated]"
        lines.append(f"Step {idx} | {name}: {content}")
    return "\n".join(lines)

def parse_primitive_trajectory(history: List[Dict], client: LLMClient,
                                chunk_size: int = 10) -> List[Dict]:
    """
    Phase 1.1: 将一条 history log 解析为 primitive <AGENT, ACTION, STATE> triples
    返回 list of {"agent", "action", "state", "step_index"}
    """
    chunks = chunk_history(history, chunk_size)
    all_triples = []

    for chunk in chunks:
        chunk_text = format_chunk(chunk)
        prompt = ABSTRACTION_PROMPT.format(chunk=chunk_text)
        raw = client.chat(ABSTRACTION_SYSTEM, prompt, max_tokens=1024, temperature=0.2)

        # 解析 JSON
        try:
            # 尝试提取 JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                triples = json.loads(match.group())
                for t in triples:
                    if isinstance(t, dict) and "agent" in t and "action" in t and "state" in t:
                        # 确保 step_index 存在且合理
                        if "step_index" not in t:
                            t["step_index"] = chunk[0][0]
                        all_triples.append(t)
        except Exception as e:
            # 解析失败时，退化为直接用 history entry 构建基本 triple
            for idx, entry in chunk:
                name = entry.get("name", entry.get("role", "Unknown"))
                content = entry.get("content", "")[:100]
                all_triples.append({
                    "agent": name,
                    "action": content[:50] if content else "unknown_action",
                    "state": "unknown_state",
                    "step_index": idx,
                })

    return all_triples


# ════════════════════════════════════════════════════════
# 4.  Phase 1.2: Hierarchical Clustering & Trajectory Refinement
#     Step A: Agent Clustering (统一 agent 名称)
#     Step B: Action-State Clustering (语义相似 triples 归为同一类)
# ════════════════════════════════════════════════════════

AGENT_CLUSTER_PROMPT = """
Given the following list of agent names extracted from multi-agent system logs, identify and group names that refer to the same agent.

Agent names: {agent_names}

Output a JSON object where keys are canonical agent names and values are lists of variant names that map to that canonical name.
Use the most common or descriptive name as the canonical form.
Output ONLY valid JSON, no markdown.

Example output:
{{"WebSurfer": ["WebSurfer", "web_agent", "Web"], "Orchestrator": ["Orchestrator", "Orch", "orchestrator"]}}
"""

ACTION_STATE_CLUSTER_PROMPT = """
Given the following list of action-state pairs from a multi-agent system, group semantically equivalent pairs together.

Action-state pairs (indexed):
{pairs}

Two pairs are semantically equivalent if they represent the same type of action with the same type of result, even if worded differently.

Output a JSON object where keys are canonical action-state descriptions (format: "action | state") 
and values are lists of original indices that belong to this cluster.
Output ONLY valid JSON, no markdown.
"""

def cluster_agents(all_triples: List[Dict], client: LLMClient) -> Dict[str, str]:
    """
    Phase 1.2 Step A: 对所有 primitive triples 中的 agent 名称做聚类，
    返回 {variant_name -> canonical_name} 的映射
    """
    agent_names = list(set(t["agent"] for t in all_triples))
    if len(agent_names) <= 1:
        return {name: name for name in agent_names}

    prompt = AGENT_CLUSTER_PROMPT.format(agent_names=json.dumps(agent_names))
    raw = client.chat(ABSTRACTION_SYSTEM, prompt, max_tokens=512, temperature=0.1)

    mapping = {}
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            clusters = json.loads(match.group())
            for canonical, variants in clusters.items():
                for v in variants:
                    mapping[v] = canonical
    except Exception:
        pass

    # 未被映射的 agent 名保持原样
    for name in agent_names:
        if name not in mapping:
            mapping[name] = name
    return mapping

def cluster_action_states(triples: List[Dict], client: LLMClient,
                           batch_size: int = 30) -> Dict[int, str]:
    """
    Phase 1.2 Step B: 对 action-state pair 做语义聚类，
    返回 {triple_list_index -> canonical_action_state_str}
    """
    if not triples:
        return {}

    # 构建 (idx, action|state) 列表
    pairs_text_list = []
    for i, t in enumerate(triples):
        pairs_text_list.append(f"{i}: {t['action']} | {t['state']}")

    # 分批处理（避免 prompt 过长）
    index_to_canonical = {}

    for batch_start in range(0, len(pairs_text_list), batch_size):
        batch = pairs_text_list[batch_start: batch_start + batch_size]
        pairs_str = "\n".join(batch)
        prompt = ACTION_STATE_CLUSTER_PROMPT.format(pairs=pairs_str)
        raw = client.chat(ABSTRACTION_SYSTEM, prompt, max_tokens=1024, temperature=0.1)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                clusters = json.loads(match.group())
                for canonical_key, indices in clusters.items():
                    for idx in indices:
                        # idx 是 batch 内的相对 index
                        abs_idx = batch_start + int(idx)
                        if abs_idx < len(triples):
                            index_to_canonical[abs_idx] = canonical_key
        except Exception:
            pass

    # 未聚类的 triples 使用原始 action|state
    for i, t in enumerate(triples):
        if i not in index_to_canonical:
            index_to_canonical[i] = f"{t['action']} | {t['state']}"

    return index_to_canonical

def refine_trajectory(primitive_triples: List[Dict],
                       agent_mapping: Dict[str, str],
                       action_state_mapping: Dict[int, str]) -> List[Dict]:
    """
    用聚类结果替换 primitive triples 中的 agent/action_state，
    生成 refined trajectory: list of {agent, action_state_canonical, step_index}
    """
    refined = []
    for i, t in enumerate(primitive_triples):
        canonical_agent = agent_mapping.get(t["agent"], t["agent"])
        canonical_as = action_state_mapping.get(i, f"{t['action']} | {t['state']}")
        refined.append({
            "agent": canonical_agent,
            "action_state": canonical_as,
            "step_index": t["step_index"],
        })
    return refined

def phase1_abstract_single_log(history: List[Dict], client: LLMClient,
                                chunk_size: int = 10) -> List[Dict]:
    """
    对单条 history 执行 Phase 1.1 抽象（不含聚类，聚类在所有 log 抽象后统一做）
    返回 primitive triples
    """
    return parse_primitive_trajectory(history, client, chunk_size)

def phase1_full(all_histories: List[List[Dict]], client: LLMClient,
                chunk_size: int = 10) -> Tuple[List[List[Dict]], Dict[str, str]]:
    """
    对所有 histories (含原始 failed log + k 条 replayed logs) 执行完整 Phase 1：
    1. 各自抽象为 primitive triples
    2. 跨所有 triples 做 hierarchical clustering
    3. 用聚类结果 refine 每条轨迹
    返回:
      refined_trajectories: list of refined trajectory (每条是 list of {agent, action_state, step_index})
      agent_mapping: {variant -> canonical}
    """
    print("    [Phase1] 抽象原始日志为 primitive triples...")
    all_primitive = []
    for i, hist in enumerate(all_histories):
        print(f"      Log {i}/{len(all_histories)-1} ({len(hist)} steps)...")
        primitives = phase1_abstract_single_log(hist, client, chunk_size)
        all_primitive.append(primitives)

    # 合并所有 triples 做聚类
    all_triples_flat = [t for traj in all_primitive for t in traj]

    print("    [Phase1] Agent 聚类...")
    agent_mapping = cluster_agents(all_triples_flat, client)

    print("    [Phase1] Action-State 聚类...")
    action_state_mapping = cluster_action_states(all_triples_flat, client)

    # 用聚类结果 refine 每条轨迹
    # 需要把 flat index 映射回各轨迹
    refined_trajectories = []
    flat_idx = 0
    for primitives in all_primitive:
        traj_as_mapping = {}
        for local_idx in range(len(primitives)):
            traj_as_mapping[local_idx] = action_state_mapping.get(flat_idx, "")
            flat_idx += 1
        refined = refine_trajectory(primitives, agent_mapping, traj_as_mapping)
        refined_trajectories.append(refined)

    return refined_trajectories, agent_mapping


# ════════════════════════════════════════════════════════
# 5.  Phase 2: Spectrum Analysis
# ════════════════════════════════════════════════════════

def build_matrices(refined_trajectories: List[List[Dict]],
                   outcomes: List[int]) -> Tuple[
                       Dict[str, Dict[str, int]],   # action_state -> {traj_i: count}
                       Dict[str, Dict[str, int]],   # agent -> {traj_i: count}
                       List[str],                    # unique action_states (UL)
                       List[str],                    # unique agents
                   ]:
    """
    构建论文中的矩阵:
      Cη, Fη (action-state triple 维度)
      Cagent, Fagent (agent 维度)
    返回字典形式（稀疏），避免大矩阵开销
    """
    # action_state_freq[η][traj_idx] = count
    action_state_freq: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # agent_freq[agent][traj_idx] = count
    agent_freq: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for traj_idx, traj in enumerate(refined_trajectories):
        for triple in traj:
            eta = triple["action_state"]
            agent = triple["agent"]
            action_state_freq[eta][traj_idx] += 1
            agent_freq[agent][traj_idx] += 1

    unique_etas = list(action_state_freq.keys())
    unique_agents = list(agent_freq.keys())

    return (dict(action_state_freq), dict(agent_freq),
            unique_etas, unique_agents)

def compute_lambda_decay_counts(freq_dict: Dict[int, int],
                                 outcomes: List[int],
                                 lam: float) -> Tuple[float, float]:
    """
    计算 λ-Decay SBFL 的 n^λ_cf 和 n^λ_cs (公式 5)
    freq_dict: {traj_idx -> count}
    outcomes: [0/1 per traj] (0=fail, 1=success)
    """
    n_cf = 0.0  # failing trajectories that covered ηj
    n_cs = 0.0  # successful trajectories that covered ηj

    for traj_idx, count in freq_dict.items():
        if count > 0:
            decay_val = (lam ** (count - 1))
            if outcomes[traj_idx] == 0:  # fail
                n_cf += decay_val
            else:                         # success
                n_cs += decay_val

    return n_cf, n_cs

def kulczynski2_lambda(n_cf: float, n_cs: float, n_uf: float) -> float:
    """
    λ-Decay 版 Kulczynski2 (公式 7)
    Kulczynski2λ = 0.5 * (n_cf / (n_cf + n_uf) + n_cf / (n_cf + n_cs))
    """
    term1 = n_cf / (n_cf + n_uf) if (n_cf + n_uf) > 0 else 0.0
    term2 = n_cf / (n_cf + n_cs) if (n_cf + n_cs) > 0 else 0.0
    return 0.5 * (term1 + term2)

def compute_suspiciousness(
    eta: str,                              # 目标 action-state triple
    agent: str,                            # 对应 agent
    action_state_freq: Dict[str, Dict[int, int]],
    agent_freq: Dict[str, Dict[int, int]],
    outcomes: List[int],
    lam: float = 0.9,
    failed_traj_idx: int = 0,
) -> float:
    """
    计算 S(ηj) = [α_τ0 · Kulczynski2λ] · [1 + β] · [1 + γ]  (公式 6)
    """
    n_traj = len(outcomes)
    eta_freq = action_state_freq.get(eta, {})
    agt_freq = agent_freq.get(agent, {})

    # ── Agent Behavior Group ──
    # γ = nc_ηj / nc_agenti  (Action Coverage Ratio, 公式 2)
    nc_eta = sum(1 for v in eta_freq.values() if v > 0)          # 含 ηj 的轨迹数
    nc_agent = sum(1 for v in agt_freq.values() if v > 0)        # 含 agenti 的轨迹数
    gamma = nc_eta / nc_agent if nc_agent > 0 else 0.0

    # β = f_ηj / f_agenti  (Action Frequency Proportion, 公式 3)
    f_eta = sum(eta_freq.values())
    f_agent = sum(agt_freq.values())
    beta = f_eta / f_agent if f_agent > 0 else 0.0

    # ── Action Behavior Group ──
    # α = 1 + log_{1/λ}(f_ij)  (Local Frequency Enhancement, 公式 4)
    f_ij = eta_freq.get(failed_traj_idx, 0)
    if f_ij > 0 and lam > 0 and lam < 1:
        log_base = 1.0 / lam  # log base = 1/λ
        alpha = 1.0 + math.log(f_ij, log_base)
    else:
        alpha = 1.0

    # λ-Decay SBFL (公式 5)
    n_cf, n_cs = compute_lambda_decay_counts(eta_freq, outcomes, lam)
    # n_uf: failed trajectories that did NOT cover ηj
    n_fail = sum(1 for o in outcomes if o == 0)
    n_uf = n_fail - sum(1 for traj_idx, cnt in eta_freq.items()
                         if cnt > 0 and outcomes[traj_idx] == 0)

    kulc = kulczynski2_lambda(n_cf, n_cs, n_uf)

    # 最终得分 (公式 6)
    score = (alpha * kulc) * (1.0 + beta) * (1.0 + gamma)
    return score

def phase2_spectrum_analysis(
    refined_trajectories: List[List[Dict]],
    outcomes: List[int],
    lam: float = 0.9,
    failed_traj_idx: int = 0,
) -> List[Tuple[str, str, float]]:
    """
    Phase 2: 对 failed trajectory (index=0) 中的每个 action-state triple 计算 suspiciousness，
    返回按分数降序排列的 [(action_state, agent, score), ...]
    """
    action_state_freq, agent_freq, unique_etas, unique_agents = build_matrices(
        refined_trajectories, outcomes
    )

    # 只对 failed trajectory 中出现的 triples 排序
    failed_traj = refined_trajectories[failed_traj_idx]
    # 去重（保留首次出现位置信息）
    seen = set()
    candidates = []
    for triple in failed_traj:
        eta = triple["action_state"]
        agent = triple["agent"]
        if eta not in seen:
            seen.add(eta)
            candidates.append((eta, agent, triple["step_index"]))

    ranked = []
    for eta, agent, step_idx in candidates:
        score = compute_suspiciousness(
            eta, agent, action_state_freq, agent_freq,
            outcomes, lam, failed_traj_idx
        )
        ranked.append((eta, agent, step_idx, score))

    # 按 score 降序
    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked  # [(action_state, agent, step_index, score)]


# ════════════════════════════════════════════════════════
# 6.  评估 (与 baseline 保持一致)
# ════════════════════════════════════════════════════════

def evaluate(predictions: List[Dict], instances: List[Dict], tolerance: int = 0) -> Dict:
    agent_correct = 0
    step_correct = 0
    total = len(predictions)

    for pred, inst in zip(predictions, instances):
        gt_agent = str(inst.get("mistake_agent", "")).lower().strip()
        try:
            gt_step = int(inst.get("mistake_step", -1))
        except Exception:
            gt_step = -1

        p_agent = str(pred.get("responsible_agent", "")).lower().strip()
        try:
            p_step = int(pred.get("error_step", -1))
        except Exception:
            p_step = -1

        if gt_agent and (gt_agent in p_agent or p_agent in gt_agent):
            agent_correct += 1

        if p_step >= 0 and gt_step >= 0 and abs(p_step - gt_step) <= tolerance:
            step_correct += 1

    return {
        "total": total,
        "agent_accuracy": agent_correct / total if total else 0,
        "step_accuracy": step_correct / total if total else 0,
        "agent_correct": agent_correct,
        "step_correct": step_correct,
    }


# ════════════════════════════════════════════════════════
# 7.  主流程: Famas 推理
# ════════════════════════════════════════════════════════

def famas_predict(instance: Dict, client: LLMClient,
                   lam: float = 0.9, chunk_size: int = 10) -> Dict:
    """
    对单个 instance 运行完整 Famas 流程，返回预测结果 dict
    """
    # ── 收集所有轨迹 ──
    # failed trajectory (必有)
    failed_history = instance.get("history", [])
    if not failed_history:
        return {"responsible_agent": "", "error_step": -1, "reason": "empty history"}

    # 额外的 replayed trajectories (可选，字段名: replayed_histories)
    # 每条为 {"history": [...], "success": 0/1}
    replayed = instance.get("replayed_histories", [])

    all_histories = [failed_history] + [r["history"] for r in replayed]
    # outcomes: 0=fail (index 0 一定是 failed), 其余按 replayed 中的 success 字段
    outcomes = [0] + [r.get("success", 0) for r in replayed]

    # ── Phase 1: Trajectory Replay & Abstraction ──
    refined_trajs, agent_mapping = phase1_full(all_histories, client, chunk_size)

    if not refined_trajs or not refined_trajs[0]:
        return {"responsible_agent": "", "error_step": -1,
                "reason": "phase1 abstraction failed"}

    # ── Phase 2: Spectrum Analysis ──
    print("    [Phase2] 频谱分析...")
    ranked = phase2_spectrum_analysis(refined_trajs, outcomes, lam=lam, failed_traj_idx=0)

    if not ranked:
        return {"responsible_agent": "", "error_step": -1,
                "reason": "no candidates from spectrum analysis"}

    # Top-1 结果
    top_action_state, top_agent, top_step_idx, top_score = ranked[0]

    # 生成 reason（列出 top-3）
    top_k = ranked[:3]
    reason_lines = []
    for rank, (as_, ag_, si_, sc_) in enumerate(top_k, 1):
        reason_lines.append(
            f"Rank{rank}: agent={ag_}, step={si_}, score={sc_:.4f}, action_state={as_}"
        )
    reason = "; ".join(reason_lines)

    return {
        "responsible_agent": top_agent,
        "error_step": top_step_idx,
        "reason": reason,
        "top_score": top_score,
        "top_action_state": top_action_state,
    }


# ════════════════════════════════════════════════════════
# 8.  入口
# ════════════════════════════════════════════════════════

def run(args):
    client = LLMClient(args.model, args.api_key, args.base_url)
    instances = load_dataset(args.data_dir)

    predictions = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "_")
    out_file = output_dir / f"famas_{model_tag}.jsonl"

    print(f"[Famas] 模型={args.model}  λ={args.lam}  数据集={args.data_dir}")
    print(f"[Famas] 实时输出: {out_file}")

    for i, inst in enumerate(instances):
        task_id = inst.get("task_id", str(i))
        print(f"\n[{i+1}/{len(instances)}] task_id={task_id}")

        try:
            pred = famas_predict(inst, client, lam=args.lam, chunk_size=args.chunk_size)
        except Exception as e:
            print(f"  [错误] {e}")
            pred = {"responsible_agent": "", "error_step": -1, "reason": str(e)}

        pred["task_id"] = task_id
        predictions.append(pred)

        # 实时写入 (与 baseline 保持一致)
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(instances)}")

    # ── 评估 ──
    print("\n[评估结果]")
    for tol in [0, 1, 2]:
        metrics = evaluate(predictions, instances, tolerance=tol)
        tag = f" (tolerance={tol})" if tol > 0 else ""
        print(f"  Agent Acc{tag}: {metrics['agent_accuracy']:.1%}"
              f"  ({metrics['agent_correct']}/{metrics['total']})")
        print(f"  Step  Acc{tag}: {metrics['step_accuracy']:.1%}"
              f"  ({metrics['step_correct']}/{metrics['total']})")

    # ── 保存汇总 ──
    dataset_name = Path(args.data_dir).name
    summary = {
        "method": "famas",
        "model": args.model,
        "lam": args.lam,
        "chunk_size": args.chunk_size,
        "n_samples": len(instances),
        "metrics_tol0": evaluate(predictions, instances, 0),
        "metrics_tol1": evaluate(predictions, instances, 1),
    }
    summary_file = output_dir / f"summary_famas_{model_tag}_{dataset_name}.json"
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\n[输出] 详细结果: {out_file}")
    print(f"[输出] 汇总结果: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Famas: Spectrum-Based Failure Attribution for MASs"
    )
    parser.add_argument("--model", default="deepseek-chat",
                        help="LLM model name (OpenAI-compatible API)")
    parser.add_argument("--data_dir", required=True,
                        help="数据集目录，每个 .json 文件为一个 instance")
    parser.add_argument("--output_dir", default="outputs",
                        help="输出目录")
    parser.add_argument("--api_key", default="",
                        help="API key (或设置 OPENAI_API_KEY 环境变量)")
    parser.add_argument("--base_url", default="",
                        help="自定义 API base URL")
    parser.add_argument("--lam", type=float, default=0.9,
                        help="λ decay factor (论文默认 0.9，范围 0.5~1.0)")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Phase 1.1 分块大小 (每块 history 条数)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()