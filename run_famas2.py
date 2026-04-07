"""
Famas: Spectrum-Based Failure Attribution for Multi-Agent Systems
复现论文: arXiv:2509.13782

数据集仅含单条 history（failed log），无 replayed_histories。
解决方案：用 LLM 对同一 task 生成 k 条"假设性"执行轨迹（含成功/失败），
          模拟论文 §4.1 的 trajectory replay，再执行 Phase 2 频谱分析。

两阶段流程：
  Phase 1: Trajectory Replay & Abstraction
    1.0  用 LLM 重放生成 k 条轨迹（替代真实 MAS 重跑）
    1.1  Primitive Abstraction: log -> <AGENT, ACTION, STATE> triples
    1.2  Hierarchical Clustering: agent 归一化 + action-state 语义聚类
  Phase 2: Spectrum Analysis
    2.1  Agent Behavior Group: γ (Action Coverage Ratio), β (Action Frequency Proportion)
    2.2  Action Behavior Group: α (Local Freq Enhancement), λ-Decay Kulczynski2
    S(ηj) = [α · Kulczynski2λ] · [1+β] · [1+γ]          (公式 6)
"""

import os
import json
import argparse
import re
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from openai import OpenAI

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ════════════════════════════════════════════════════════
# 1.  LLM 调用层（与 baseline 保持一致，支持本地模型）
# ════════════════════════════════════════════════════════

class LLMClient:
    def __init__(self, model: str, api_key: str = "", base_url: str = ""):
        self.model = model
        self._local_model = None
        self._tokenizer = None

        if model.startswith("local:"):
            self._init_local(model[6:])
        else:
            kwargs = {"api_key": api_key or os.environ.get("OPENAI_API_KEY", "sk-xxx")}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)

    def _init_local(self, model_path: str):
        assert HAS_TRANSFORMERS, "transformers not installed"
        print(f"[本地模型] 加载 {model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._local_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
        )

    def chat(self, system: str, user: str, max_tokens: int = 1024,
             temperature: float = 0.6) -> str:
        if self._local_model is not None:
            return self._local_chat(system, user, max_tokens, temperature)
        return self._api_chat(system, user, max_tokens, temperature)

    def _api_chat(self, system: str, user: str, max_tokens: int,
                  temperature: float) -> str:
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"    [API重试 {attempt+1}/3] {e}")
                time.sleep(2 ** attempt)
        return ""

    def _local_chat(self, system: str, user: str, max_tokens: int,
                    temperature: float) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._local_model.device)
        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature, do_sample=True)
        return self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True).strip()


# ════════════════════════════════════════════════════════
# 2.  数据加载与评估（与 baseline 完全一致）
# ════════════════════════════════════════════════════════

def load_dataset(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    instances = []
    json_files = sorted(
        [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".json"],
        key=lambda f: int("".join(filter(str.isdigit, f.stem)) or 0),
    )
    for jf in json_files:
        with open(jf, encoding="utf-8") as fh:
            data = json.load(fh)
        data["task_id"] = jf.stem
        instances.append(data)
    return instances


def evaluate(predictions: List[Dict], instances: List[Dict],
             tolerance: int = 0) -> Dict:
    agent_correct = step_correct = 0
    total = len(predictions)
    for pred, inst in zip(predictions, instances):
        gt_agent = str(inst.get("mistake_agent", "")).lower().strip()
        try:    gt_step = int(inst.get("mistake_step", -1))
        except: gt_step = -1

        p_agent = str(pred.get("responsible_agent", "")).lower().strip()
        try:    p_step = int(pred.get("error_step", -1))
        except: p_step = -1

        if gt_agent and (gt_agent in p_agent or p_agent in gt_agent):
            agent_correct += 1
        if p_step >= 0 and gt_step >= 0 and abs(p_step - gt_step) <= tolerance:
            step_correct += 1

    return {
        "total":           total,
        "agent_accuracy":  agent_correct / total if total else 0,
        "step_accuracy":   step_correct  / total if total else 0,
        "agent_correct":   agent_correct,
        "step_correct":    step_correct,
    }


# ════════════════════════════════════════════════════════
# 3.  Phase 1.0 — Trajectory Replay via LLM
#     数据集无真实 replayed logs，用 LLM 模拟 k 次独立执行
# ════════════════════════════════════════════════════════

REPLAY_SYSTEM = "You are an expert at simulating multi-agent system executions."

REPLAY_PROMPT = """\
A multi-agent system attempted to solve the following problem but FAILED:

Problem: {problem}
Agents involved: {agents}

The original (failed) execution had this conversation:
{original_conv}

Now simulate ONE alternative independent execution of the same task by the same agents.
This new run may succeed or fail - vary the agents' behavior naturally.

Output a JSON object with:
- "success": 1 if the task is solved correctly, 0 if it fails
- "history": a list of steps, each with:
  - "name": agent name (must be one of the agents listed above)
  - "content": what the agent says/does (be concise, 1-3 sentences)

Output ONLY valid JSON, no markdown fences.
"""

def format_history_for_prompt(history: List[Dict], max_chars: int = 2000) -> str:
    lines = []
    for i, m in enumerate(history):
        name    = m.get("name", m.get("role", "Unknown"))
        content = m.get("content", "")[:300]
        lines.append(f"Step {i} | {name}: {content}")
    return "\n".join(lines)[:max_chars]


def replay_trajectories(instance: Dict, client: LLMClient,
                         k: int = 20) -> Tuple[List[List[Dict]], List[int]]:
    """
    Phase 1.0: 用 LLM 生成 k 条额外轨迹。
    返回:
      all_histories: [failed_history, replay_1, ..., replay_m]   (m <= k)
      outcomes:      [0, success_1, ..., success_m]
    """
    failed_history = instance.get("history", [])
    problem = instance.get("question", "")
    agents  = list({m.get("name", m.get("role", "Unknown")) for m in failed_history})
    original_conv = format_history_for_prompt(failed_history)

    all_histories: List[List[Dict]] = [failed_history]
    outcomes:      List[int]        = [0]   # index 0 永远是 failed trajectory

    for _ in range(k):
        prompt = REPLAY_PROMPT.format(
            problem=problem,
            agents=", ".join(agents),
            original_conv=original_conv,
        )
        raw = client.chat(REPLAY_SYSTEM, prompt, max_tokens=1024, temperature=0.8)

        try:
            raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            obj = json.loads(raw_clean)
            hist    = obj.get("history", [])
            success = int(obj.get("success", 0))
            if isinstance(hist, list) and len(hist) > 0:
                all_histories.append(hist)
                outcomes.append(success)
        except Exception:
            pass   # 解析失败跳过此次 replay

    return all_histories, outcomes


# ════════════════════════════════════════════════════════
# 4.  Phase 1.1 — Primitive Trajectory Abstraction
#     每条 history -> list of <AGENT, ACTION, STATE> triples
# ════════════════════════════════════════════════════════

ABSTRACTION_SYSTEM = (
    "You are an expert at extracting structured information from multi-agent logs."
)

ABSTRACTION_PROMPT = """\
Extract structured agent-action-state triples from the following multi-agent conversation segment.

For EACH step output a JSON object with:
- "step_index": integer (the Step number shown)
- "agent": agent name exactly as shown
- "action": 3-6 word canonical verb phrase describing what the agent DID
             (e.g. "execute python code", "search web", "verify result")
- "state": 3-6 word canonical noun phrase describing the resulting system state
           (e.g. "code output returned", "search results retrieved", "answer confirmed")

Rules:
- Use consistent canonical phrasing; avoid agent-specific detail
- Output a JSON ARRAY of objects, one per step
- Output ONLY valid JSON array, no markdown

Conversation segment:
{chunk}

JSON array:
"""

def chunk_history(history: List[Dict],
                  chunk_size: int = 8) -> List[List[Tuple[int, Dict]]]:
    chunks = []
    for i in range(0, len(history), chunk_size):
        chunks.append([(idx, history[idx])
                       for idx in range(i, min(i + chunk_size, len(history)))])
    return chunks


def format_chunk(chunk: List[Tuple[int, Dict]]) -> str:
    lines = []
    for idx, entry in chunk:
        name    = entry.get("name", entry.get("role", "Unknown"))
        content = entry.get("content", "")
        if len(content) > 500:
            content = content[:500] + "…"
        lines.append(f"Step {idx} | {name}: {content}")
    return "\n".join(lines)


def abstract_single_log(history: List[Dict], client: LLMClient,
                         chunk_size: int = 8) -> List[Dict]:
    """
    Phase 1.1: 单条 history -> primitive triples
    每个 triple: {"step_index", "agent", "action", "state"}
    """
    all_triples: List[Dict] = []
    for chunk in chunk_history(history, chunk_size):
        prompt = ABSTRACTION_PROMPT.format(chunk=format_chunk(chunk))
        raw = client.chat(ABSTRACTION_SYSTEM, prompt, max_tokens=1024, temperature=0.1)

        try:
            raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            triples = json.loads(raw_clean)
            if isinstance(triples, list):
                for t in triples:
                    if isinstance(t, dict) and "agent" in t \
                            and "action" in t and "state" in t:
                        if "step_index" not in t:
                            t["step_index"] = chunk[0][0]
                        all_triples.append(t)
        except Exception:
            # 解析失败：回退为粗粒度 triple
            for idx, entry in chunk:
                name    = entry.get("name", entry.get("role", "Unknown"))
                content = entry.get("content", "")[:60].replace("\n", " ")
                all_triples.append({
                    "step_index": idx,
                    "agent":      name,
                    "action":     content[:40] if content else "unknown action",
                    "state":      "unknown state",
                })

    return all_triples


# ════════════════════════════════════════════════════════
# 5.  Phase 1.2 — Hierarchical Clustering & Trace Refinement
# ════════════════════════════════════════════════════════

AGENT_CLUSTER_PROMPT = """\
Given these agent name variants from multi-agent system logs, group names referring to the same agent.

Names: {names}

Output a JSON object: canonical_name -> [list of variant names].
Use the most descriptive name as canonical.
Output ONLY valid JSON, no markdown.

Example: {{"WebSurfer": ["WebSurfer", "Web", "web_agent"]}}
"""

ACTION_STATE_CLUSTER_PROMPT = """\
Group semantically equivalent action-state pairs from this list.
Two pairs are equivalent if they describe the same type of agent behavior and outcome.

Indexed pairs:
{pairs}

Output a JSON object: "canonical action | canonical state" -> [list of integer indices].
Output ONLY valid JSON, no markdown.
"""


def cluster_agents(all_triples: List[Dict], client: LLMClient) -> Dict[str, str]:
    """返回 {variant_name -> canonical_name}"""
    names = list({t["agent"] for t in all_triples})
    if len(names) <= 1:
        return {n: n for n in names}

    prompt = AGENT_CLUSTER_PROMPT.format(names=json.dumps(names))
    raw    = client.chat(ABSTRACTION_SYSTEM, prompt, max_tokens=512, temperature=0.1)

    mapping: Dict[str, str] = {}
    try:
        raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        clusters  = json.loads(raw_clean)
        for canonical, variants in clusters.items():
            for v in (variants if isinstance(variants, list) else [variants]):
                mapping[str(v)] = canonical
    except Exception:
        pass

    for n in names:
        if n not in mapping:
            mapping[n] = n
    return mapping


def cluster_action_states(all_triples: List[Dict], client: LLMClient,
                           batch_size: int = 40) -> Dict[int, str]:
    """返回 {flat_index -> canonical_action_state_str}"""
    if not all_triples:
        return {}

    index_to_canonical: Dict[int, str] = {}

    for batch_start in range(0, len(all_triples), batch_size):
        batch = all_triples[batch_start: batch_start + batch_size]
        pairs_lines = [f"{i}: {t['action']} | {t['state']}"
                       for i, t in enumerate(batch)]
        prompt = ACTION_STATE_CLUSTER_PROMPT.format(pairs="\n".join(pairs_lines))
        raw    = client.chat(ABSTRACTION_SYSTEM, prompt, max_tokens=1024, temperature=0.1)

        try:
            raw_clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            clusters  = json.loads(raw_clean)
            for canonical_key, indices in clusters.items():
                for local_idx in (indices if isinstance(indices, list) else [indices]):
                    abs_idx = batch_start + int(local_idx)
                    if 0 <= abs_idx < len(all_triples):
                        index_to_canonical[abs_idx] = str(canonical_key)
        except Exception:
            pass

    for i, t in enumerate(all_triples):
        if i not in index_to_canonical:
            index_to_canonical[i] = f"{t['action']} | {t['state']}"

    return index_to_canonical


def phase1_full(all_histories: List[List[Dict]], client: LLMClient,
                chunk_size: int = 8) -> List[List[Dict]]:
    """
    完整 Phase 1：
      1.1 每条 history -> primitive triples（分块 LLM 解析）
      1.2 跨所有 triples 做 hierarchical clustering（agent + action-state）
      → refine 每条轨迹，返回 refined_trajectories
    """
    # ── 1.1 Primitive abstraction ──
    print("    [Phase1.1] 抽象 primitive triples ...")
    all_primitive: List[List[Dict]] = []
    for i, hist in enumerate(all_histories):
        print(f"      轨迹 {i}/{len(all_histories)-1}  ({len(hist)} steps)")
        all_primitive.append(abstract_single_log(hist, client, chunk_size))

    # 合并（记录各轨迹长度，用于后续拆分）
    flat_triples: List[Dict] = [t for traj in all_primitive for t in traj]
    traj_lengths = [len(p) for p in all_primitive]

    # ── 1.2 Hierarchical clustering ──
    print("    [Phase1.2] Agent 聚类 ...")
    agent_mapping = cluster_agents(flat_triples, client)

    print("    [Phase1.2] Action-State 聚类 ...")
    as_mapping = cluster_action_states(flat_triples, client)

    # ── Refine ──
    refined_trajectories: List[List[Dict]] = []
    flat_idx = 0
    for primitives in all_primitive:
        refined: List[Dict] = []
        for t in primitives:
            canon_agent = agent_mapping.get(t["agent"], t["agent"])
            canon_as    = as_mapping.get(flat_idx, f"{t['action']} | {t['state']}")
            refined.append({
                "agent":        canon_agent,
                "action_state": canon_as,
                "step_index":   t["step_index"],
            })
            flat_idx += 1
        refined_trajectories.append(refined)

    return refined_trajectories


# ════════════════════════════════════════════════════════
# 6.  Phase 2 — Spectrum Analysis
# ════════════════════════════════════════════════════════

def build_frequency_matrices(
    refined_trajectories: List[List[Dict]],
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, int]]]:
    """
    构建:
      eta_freq[action_state][traj_idx]  = count   (对应论文 Fη)
      agent_freq[agent][traj_idx]       = count   (对应论文 Fagent)
    """
    eta_freq:   Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    agent_freq: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for traj_idx, traj in enumerate(refined_trajectories):
        for triple in traj:
            eta_freq[triple["action_state"]][traj_idx]  += 1
            agent_freq[triple["agent"]][traj_idx]        += 1

    return dict(eta_freq), dict(agent_freq)


def lambda_decay_counts(freq_by_traj: Dict[int, int],
                         outcomes: List[int],
                         lam: float) -> Tuple[float, float]:
    """
    公式 5: n^λ_cf, n^λ_cs
    每条覆盖 ηj 的轨迹贡献 λ^(count-1)（而非二元 0/1）
    """
    n_cf = n_cs = 0.0
    for traj_idx, count in freq_by_traj.items():
        if count > 0:
            w = lam ** (count - 1)
            if outcomes[traj_idx] == 0:
                n_cf += w
            else:
                n_cs += w
    return n_cf, n_cs


def kulczynski2_lam(n_cf: float, n_cs: float, n_uf: float) -> float:
    """公式 7: Kulczynski2λ = 0.5 * (n_cf/(n_cf+n_uf) + n_cf/(n_cf+n_cs))"""
    t1 = n_cf / (n_cf + n_uf) if (n_cf + n_uf) > 0 else 0.0
    t2 = n_cf / (n_cf + n_cs) if (n_cf + n_cs) > 0 else 0.0
    return 0.5 * (t1 + t2)


def suspiciousness(
    eta:   str,
    agent: str,
    eta_freq:   Dict[str, Dict[int, int]],
    agent_freq: Dict[str, Dict[int, int]],
    outcomes:   List[int],
    lam:        float = 0.9,
    failed_traj_idx: int = 0,
) -> float:
    """
    公式 6: S(ηj) = [α · Kulczynski2λ] · [1+β] · [1+γ]
    """
    eta_f   = eta_freq.get(eta,   {})
    agent_f = agent_freq.get(agent, {})

    # ── Agent Behavior Group ──
    # γ = nc_ηj / nc_agenti   (公式 2)
    nc_eta   = sum(1 for v in eta_f.values()   if v > 0)
    nc_agent = sum(1 for v in agent_f.values() if v > 0)
    gamma = nc_eta / nc_agent if nc_agent > 0 else 0.0

    # β = f_ηj / f_agenti     (公式 3)
    f_eta   = sum(eta_f.values())
    f_agent = sum(agent_f.values())
    beta = f_eta / f_agent if f_agent > 0 else 0.0

    # ── Action Behavior Group ──
    # α = 1 + log_{1/λ}(f_ij)   (公式 4)
    f_ij = eta_f.get(failed_traj_idx, 0)
    if f_ij > 0 and 0 < lam < 1:
        alpha = 1.0 + math.log(f_ij, 1.0 / lam)
    else:
        alpha = 1.0

    # λ-Decay Kulczynski2   (公式 5, 7)
    n_cf, n_cs = lambda_decay_counts(eta_f, outcomes, lam)
    n_fail = sum(1 for o in outcomes if o == 0)
    n_uf   = n_fail - sum(
        1 for traj_idx, cnt in eta_f.items()
        if cnt > 0 and outcomes[traj_idx] == 0
    )
    kulc = kulczynski2_lam(n_cf, n_cs, max(n_uf, 0))

    return (alpha * kulc) * (1.0 + beta) * (1.0 + gamma)


def phase2_spectrum_analysis(
    refined_trajectories: List[List[Dict]],
    outcomes: List[int],
    lam: float = 0.9,
    failed_traj_idx: int = 0,
) -> List[Tuple[str, str, int, float]]:
    """
    对 failed trajectory 中每个 unique triple 计算 S(ηj)，
    返回按分数降序的 [(action_state, agent, step_index, score)]
    """
    print("    [Phase2] 构建频率矩阵 ...")
    eta_freq, agent_freq = build_frequency_matrices(refined_trajectories)

    failed_traj = refined_trajectories[failed_traj_idx]
    seen: set = set()
    candidates: List[Tuple[str, str, int]] = []
    for triple in failed_traj:
        eta = triple["action_state"]
        if eta not in seen:
            seen.add(eta)
            candidates.append((eta, triple["agent"], triple["step_index"]))

    print(f"    [Phase2] 计算 suspiciousness ({len(candidates)} 个 unique triples) ...")
    ranked: List[Tuple[str, str, int, float]] = []
    for eta, agent, step_idx in candidates:
        score = suspiciousness(
            eta, agent, eta_freq, agent_freq,
            outcomes, lam, failed_traj_idx
        )
        ranked.append((eta, agent, step_idx, score))

    ranked.sort(key=lambda x: x[3], reverse=True)
    return ranked


# ════════════════════════════════════════════════════════
# 7.  Famas 单样本预测
# ════════════════════════════════════════════════════════

def famas_predict(instance: Dict, client: LLMClient,
                   k: int = 20, lam: float = 0.9,
                   chunk_size: int = 8) -> Dict:
    """对单个 instance 执行完整 Famas 流程"""

    # ── Phase 1.0: Trajectory Replay ──
    print(f"    [Phase1.0] LLM 轨迹重放 (k={k}) ...")
    all_histories, outcomes = replay_trajectories(instance, client, k=k)
    n_success = sum(outcomes)
    n_fail    = len(outcomes) - n_success
    print(f"      收集到 {len(all_histories)} 条轨迹 "
          f"(失败={n_fail}, 成功={n_success})")

    # ── Phase 1.1 + 1.2: Abstraction & Clustering ──
    refined_trajs = phase1_full(all_histories, client, chunk_size)

    if not refined_trajs or not refined_trajs[0]:
        return {"responsible_agent": "", "error_step": -1,
                "reason": "Phase1 abstraction returned empty result"}

    # ── Phase 2: Spectrum Analysis ──
    ranked = phase2_spectrum_analysis(refined_trajs, outcomes, lam=lam)

    if not ranked:
        return {"responsible_agent": "", "error_step": -1,
                "reason": "Phase2 returned no ranked candidates"}

    top_as, top_agent, top_step, top_score = ranked[0]

    # reason: top-3
    reason = "; ".join(
        f"Rank{r}: agent={ag}, step={si}, score={sc:.4f}, action_state={aas}"
        for r, (aas, ag, si, sc) in enumerate(ranked[:3], 1)
    )

    return {
        "responsible_agent": top_agent,
        "error_step":        top_step,
        "reason":            reason,
        "top_score":         top_score,
        "top_action_state":  top_as,
        "n_trajectories":    len(all_histories),
        "n_success":         n_success,
        "n_fail":            n_fail,
    }


# ════════════════════════════════════════════════════════
# 8.  主运行逻辑（与 baseline 结构完全对齐）
# ════════════════════════════════════════════════════════

def run(args):
    client    = LLMClient(args.model, args.api_key, args.base_url)
    instances = load_dataset(args.data_dir)

    predictions: List[Dict] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "_")
    out_file  = output_dir / f"famas_{model_tag}.jsonl"

    print(f"[Famas] 模型={args.model}  k={args.k}  λ={args.lam}  数据集={args.data_dir}")
    print(f"[Famas] 实时输出 -> {out_file}\n")

    for i, inst in enumerate(instances):
        task_id = inst.get("task_id", str(i))
        print(f"[{i+1}/{len(instances)}] task_id={task_id}")

        try:
            pred = famas_predict(inst, client,
                                  k=args.k, lam=args.lam,
                                  chunk_size=args.chunk_size)
        except Exception as e:
            print(f"  [错误] {e}")
            pred = {"responsible_agent": "", "error_step": -1, "reason": str(e)}

        pred["task_id"] = task_id
        predictions.append(pred)

        # 实时写入（与 baseline 保持一致）
        with open(out_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(pred, ensure_ascii=False) + "\n")

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
        "method":       "famas",
        "model":        args.model,
        "k":            args.k,
        "lam":          args.lam,
        "chunk_size":   args.chunk_size,
        "n_samples":    len(instances),
        "metrics_tol0": evaluate(predictions, instances, 0),
        "metrics_tol1": evaluate(predictions, instances, 1),
    }
    summary_file = output_dir / f"summary_famas_{model_tag}_{dataset_name}.json"
    with open(summary_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\n[输出] 详细结果: {out_file}")
    print(f"[输出] 汇总结果: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Famas: Spectrum-Based Failure Attribution for MASs"
    )
    parser.add_argument("--model",      default="deepseek-chat",
                        help="LLM 模型名（OpenAI-compatible API 或 local:<path>）")
    parser.add_argument("--data_dir",   required=True,
                        help="数据集目录，每个 .json 为一个 instance")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--api_key",    default="")
    parser.add_argument("--base_url",   default="")
    parser.add_argument("--k",    type=int,   default=20,
                        help="轨迹重放次数（论文默认 k=20）")
    parser.add_argument("--lam",  type=float, default=0.9,
                        help="λ decay factor（论文默认 0.9）")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Phase 1.1 每次分析的 history 步数")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()