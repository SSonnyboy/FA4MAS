# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/31 19:22 
'''
# -*- coding: UTF-8 -*-
import json
import re
from pathlib import Path

from data.dataset import get_query_and_gt, get_log_steps
from methods.base_method import BaseAttributionMethod


# ════════════════════════════════════════════════════════
# CHIEF 专有数据结构
# ════════════════════════════════════════════════════════
class SubtaskNode:
    def __init__(self, sid, name, step_range, agents_id=None, oracle=None):
        self.id = sid
        self.name = name
        self.step_range = step_range
        self.agents_id = agents_id or []
        self.oracle = oracle or {}

    def to_dict(self):
        return {"id": self.id, "name": self.name, "step_range": list(self.step_range), "agents_id": self.agents_id,
                "oracle": self.oracle}


class AgentNode:
    def __init__(self, aid, name, subtask_id, step_ids=None, otar=None):
        self.id = aid
        self.name = name
        self.subtask_id = subtask_id
        self.step_ids = step_ids or []
        self.otar = otar or {}

    def to_dict(self):
        return {"id": self.id, "name": self.name, "subtask_id": self.subtask_id, "step_ids": self.step_ids,
                "otar": self.otar}


class CausalEdge:
    def __init__(self, from_id, to_id, edge_type, level, counterfactual_patterns=None, data_info=None):
        self.from_id = from_id
        self.to_id = to_id
        self.edge_type = edge_type
        self.level = level
        self.counterfactual_patterns = counterfactual_patterns or []
        self.data_info = data_info or {}

    def to_dict(self):
        return {"from": self.from_id, "to": self.to_id, "type": self.edge_type, "level": self.level,
                "counterfactual_patterns": self.counterfactual_patterns, "data_info": self.data_info}


class HierarchicalCausalGraph:
    def __init__(self):
        self.subtask_nodes = []
        self.agent_nodes = []
        self.edges = []

    def to_dict(self):
        return {
            "subtask_nodes": [n.to_dict() for n in self.subtask_nodes],
            "agent_nodes": [n.to_dict() for n in self.agent_nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_text(self):
        lines = ["=== Hierarchical Causal Graph ===", "\n--- Subtask Nodes ---"]
        for s in self.subtask_nodes:
            lines.append(f"[{s.id}] {s.name}  steps={s.step_range}  agents={s.agents_id}")
            if s.oracle:
                lines.extend([f"  Oracle.Goal: {s.oracle.get('Goal', '')}",
                              f"  Oracle.Precondition: {s.oracle.get('Precondition', '')}",
                              f"  Oracle.Key_Evidence: {s.oracle.get('Key_Evidence', '')}",
                              f"  Oracle.Acceptance_Criteria: {s.oracle.get('Acceptance_Criteria', '')}"])
        lines.append("\n--- Agent Nodes ---")
        for a in self.agent_nodes:
            otar_summary = f" | Action: {str(a.otar.get('Action', ''))[:60]}" if a.otar else ""
            lines.append(f"[{a.id}] {a.name} (subtask={a.subtask_id}, steps={a.step_ids}){otar_summary}")
        lines.append("\n--- Causal Edges ---")
        for e in self.edges:
            patterns = ""
            if e.counterfactual_patterns:
                p = e.counterfactual_patterns[0]
                patterns = f" | Bias->Anomaly: {str(p.get('Bias', ''))[:40]} -> {str(p.get('Anomaly', ''))[:40]}"
            if e.level == "step" and e.data_info:
                patterns = f" | data: {e.data_info.get('output_data', '')[:40]} -> {e.data_info.get('input_data', '')[:40]}"
            lines.append(f"[{e.level}] {e.from_id} -> {e.to_id}  ({e.edge_type}){patterns}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════
# RAG 知识库与检索
# ════════════════════════════════════════════════════════
RAG_KNOWLEDGE_BASE = [
    {"source": "AssistantBench", "task_type": "geo_search",
     "question": "Which gyms (not including gymnastics centers) in West Virginia are within 5 miles (by car) of the Mothman Museum?",
     "subtasks": "1. Locate Mothman Museum coordinates. 2. Search nearby gyms within 5 miles by car. 3. Filter out gymnastics centers. 4. Return qualifying gym list."},
    {"source": "GAIA", "task_type": "multi_role_lookup",
     "question": "A 5-man group made up of one tank, one healer, and three DPS is doing a dungeon run. What classes can fill all three roles?",
     "subtasks": "1. Search WoW class role documentation. 2. Identify classes supporting tank+healer+DPS. 3. List qualifying classes alphabetically."},
    {"source": "GAIA", "task_type": "math_computation",
     "question": "What is the total distance in km if a traveler visits cities A, B, C in order using the given distance table?",
     "subtasks": "1. Parse distance table. 2. Sum A→B and B→C distances. 3. Return total."},
]  # 为了演示精简了部分，你可以把原版全粘回来

_RAG_INDEX = None
_FALLBACK_RAG = "[Exemplar 1 | Source: AssistantBench]\nQuestion: Which gyms in West Virginia are within 5 miles of the Mothman Museum?\nSubtask decomposition: 1. Locate Mothman Museum. 2. Search nearby gyms within 5 miles. 3. Filter gymnastics centers. 4. Return list."


def retrieve_rag_examples(query: str, top_k: int = 2) -> str:
    global _RAG_INDEX
    try:
        if _RAG_INDEX is None:
            corpus = [item["question"] for item in RAG_KNOWLEDGE_BASE]
            import math
            from collections import Counter
            tokenized = [q.lower().split() for q in corpus]
            df = Counter(tok for doc in tokenized for tok in set(doc))
            N = len(corpus)
            idf = {tok: math.log((N + 1) / (df[tok] + 1)) for tok in df}
            vecs = [{t: Counter(doc)[t] * idf.get(t, 0) for t in doc} for doc in tokenized]
            _RAG_INDEX = (tokenized, vecs, idf)

        tokenized, vecs, idf = _RAG_INDEX
        q_tokens = query.lower().split()
        q_vec = {t: Counter(q_tokens)[t] * idf.get(t, 0) for t in q_tokens}

        def cosine(a, b):
            return sum(a.get(t, 0) * b.get(t, 0) for t in a) / ((math.sqrt(sum(v ** 2 for v in a.values())) + 1e-9) * (
                        math.sqrt(sum(v ** 2 for v in b.values())) + 1e-9))

        scores = [cosine(q_vec, v) for v in vecs]
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        return "\n\n".join(
            f"[Retrieved exemplar {rank + 1} | Source: {RAG_KNOWLEDGE_BASE[idx]['source']}]\nQuestion: {RAG_KNOWLEDGE_BASE[idx]['question']}\nSubtask decomposition: {RAG_KNOWLEDGE_BASE[idx]['subtasks']}"
            for rank, idx in enumerate(top_indices))
    except Exception as e:
        return _FALLBACK_RAG


# ════════════════════════════════════════════════════════
# CHIEF 方法主类
# ════════════════════════════════════════════════════════
class CHIEFMethod(BaseAttributionMethod):
    def __init__(self, llm_client, save_graph: bool = False, output_dir: str = "outputs"):
        super().__init__(llm_client)
        self.save_graph = save_graph
        self.output_dir = output_dir

    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        query, gt_section = get_query_and_gt(instance, use_ground_truth)

        # 统一使用 get_log_steps 提取洗净后的对话
        steps = get_log_steps(instance)
        if not steps:
            return {"responsible_agent": "", "error_step": -1, "reason": "fallback: no structured steps"}

        history_text = "\n".join(f"[Step {s[0]}] {s[1]}: {s[2]}" for s in steps)
        n_steps = len(steps)

        if verbose: print("\n[CHIEF Stage 1] Constructing Hierarchical Causal Graph...")
        subtasks = self._rag_decompose(query, gt_section, history_text, n_steps, steps)
        agent_nodes = self._otar_parse(query, gt_section, history_text, n_steps, subtasks)
        sub_agent_edges = self._build_subtask_agent_edges(query, gt_section, history_text, n_steps, subtasks,
                                                          agent_nodes)
        step_edges = self._build_step_edges(query, gt_section, history_text, n_steps, subtasks)

        graph = HierarchicalCausalGraph()
        graph.subtask_nodes, graph.agent_nodes = subtasks, agent_nodes
        graph.edges = sub_agent_edges + step_edges

        if verbose: print("[CHIEF Stage 2] Oracle Synthesis & Hierarchical Backtracking...")
        subtasks = self._synthesize_oracles(query, gt_section, history_text, n_steps, steps, subtasks)
        graph.subtask_nodes = subtasks
        candidates = self._hierarchical_backtrack(query, gt_section, history_text, n_steps, graph)

        if verbose: print("[CHIEF Stage 3] Counterfactual Attribution...")
        result = self._counterfactual_attribute(query, gt_section, history_text, n_steps, steps, graph, candidates)

        if self.save_graph:
            graph_dir = Path(self.output_dir) / "chief_graphs"
            graph_dir.mkdir(parents=True, exist_ok=True)
            with open(graph_dir / f"graph_{instance.get('task_id', 'unknown')}.json", "w", encoding="utf-8") as f:
                json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)

        return result

    # ── Stage 1 ─────────────────────────────────────────────
    def _rag_decompose(self, query, gt_section, history_text, n_steps, steps):
        rag_examples = retrieve_rag_examples(query)
        prompt = f"The problem is: {query}\n{gt_section}\nHere is the conversation:\n{history_text}\nThere are total {n_steps} steps.\nHere is the retrieved reference example:\n{rag_examples}\nDecompose the reasoning into semantic subtasks.\nOutput ONLY a JSON array of subtasks:\n[{{\"id\": \"S1\", \"name\": \"...\", \"step_range\": [<start>, <end>]}}]"
        raw = self.llm.call("You are a precise multi-agent system analyst. Output only valid JSON.", prompt,
                            max_tokens=1000)

        subtasks = []
        parsed = self.llm.parse_json(raw)
        if isinstance(parsed, list):
            for item in parsed:
                sr = item.get("step_range", [1, n_steps])
                subtasks.append(SubtaskNode(item.get("id", f"S{len(subtasks) + 1}"), item.get("name", "Subtask"),
                                            (int(sr[0]), int(sr[1]))))

        if not subtasks:
            subtasks = [SubtaskNode("S1", "Full Task", (1, n_steps))]

        for st in subtasks:
            s, e_idx = st.step_range
            seen = []
            for i in range(max(0, s - 1), min(e_idx, n_steps)):
                name = steps[i][1]
                if name and name not in seen: seen.append(name)
            st.agents_id = seen
        return subtasks

    def _otar_parse(self, query, gt_section, history_text, n_steps, subtasks):
        subtasks_text = "\n".join(
            f"[{s.id}] {s.name}  step_range: {s.step_range}  agents: {s.agents_id}" for s in subtasks)
        prompt = f"The problem is: {query}\n{gt_section}\nConversation:\n{history_text}\nSubtasks:\n{subtasks_text}\nSummarize each agent's behavior into Observation / Thought / Action / Result.\nOutput ONLY a JSON array:\n[{{\"subtask_id\": \"S1\", \"agents\": [{{\"name\": \"<agent>\", \"step_ids\": [<ints>], \"Observation\": \"...\", \"Thought\": \"...\", \"Action\": \"...\", \"Result\": \"...\"}}]}}]"
        raw = self.llm.call("You are a precise multi-agent system analyst. Output only valid JSON.", prompt,
                            max_tokens=2000)

        agent_nodes, counter = [], 1
        parsed = self.llm.parse_json(raw)
        if isinstance(parsed, list):
            for item in parsed:
                sid = item.get("subtask_id", "S1")
                for ag in item.get("agents", []):
                    agent_nodes.append(
                        AgentNode(f"A{counter}", ag.get("name", f"Agent{counter}"), sid, ag.get("step_ids", []),
                                  {"Observation": ag.get("Observation", ""), "Thought": ag.get("Thought", ""),
                                   "Action": ag.get("Action", ""), "Result": ag.get("Result", "")}))
                    counter += 1
        return agent_nodes

    def _build_subtask_agent_edges(self, query, gt_section, history_text, n_steps, subtasks, agent_nodes):
        subtasks_agents_text = "\n".join(
            f"[{st.id}] {st.name} agents: {', '.join(a.name for a in agent_nodes if a.subtask_id == st.id)}" for st in
            subtasks)
        prompt = f"Problem: {query}\n{gt_section}\nConversation:\n{history_text}\nSubtasks and agents:\n{subtasks_agents_text}\nConstruct causal edges. Output ONLY a JSON array:\n[{{\"from\": \"<id>\", \"to\": \"<id>\", \"level\": \"subtask or agent\", \"type\": \"...\", \"counterfactual_patterns\": [{{\"Bias\": \"...\", \"Anomaly\": \"...\"}}]}}]"
        raw = self.llm.call("You are a precise causal graph builder. Output only valid JSON.", prompt, max_tokens=2000)

        edges = []
        parsed = self.llm.parse_json(raw)
        if isinstance(parsed, list):
            for item in parsed: edges.append(
                CausalEdge(item.get("from", ""), item.get("to", ""), item.get("type", "data_dependency"),
                           item.get("level", "subtask"), item.get("counterfactual_patterns", [])))
        return edges

    def _build_step_edges(self, query, gt_section, history_text, n_steps, subtasks):
        prompt = f"Problem: {query}\n{gt_section}\nConversation:\n{history_text}\nIdentify step-level data flow edges.\nOutput ONLY a JSON array:\n[{{\"upstream_step\": <int>, \"output_data\": \"...\", \"data_type\": \"...\", \"downstream_step\": <int>, \"input_data\": \"...\"}}]"
        raw = self.llm.call("You are a precise data flow analyst. Output only valid JSON.", prompt, max_tokens=1500)

        edges = []
        parsed = self.llm.parse_json(raw)
        if isinstance(parsed, list):
            for item in parsed:
                up, dn = item.get("upstream_step", 0), item.get("downstream_step", 0)
                if up and dn: edges.append(CausalEdge(f"step_{up}", f"step_{dn}", "data_flow", "step", data_info=item))
        return edges

    # ── Stage 2 ─────────────────────────────────────────────
    def _synthesize_oracles(self, query, gt_section, history_text, n_steps, steps, subtasks):
        rag_examples = retrieve_rag_examples(query)
        generated_oracles = []
        for k, st in enumerate(subtasks):
            prior_constraints = "Previously generated oracles:\n" + "".join(
                f"[{p['subtask_id']}] Goal: {p['Goal']}\n" for p in
                generated_oracles) if generated_oracles else "No prior constraints."

            step_start, step_end = st.step_range
            subtask_steps_text = "\n".join(
                f"[Step {s[0]}] {s[1]}: {s[2]}" for s in steps[max(0, step_start - 1):step_end])

            prompt = f"Synthesizing Virtual Oracle for subtask [{st.id}]: {st.name}.\nProblem: {query}\n{gt_section}\nSteps assigned to this subtask:\n{subtask_steps_text}\nPrior Constraints:\n{prior_constraints}\nOutput ONLY a JSON object:\n{{\"Goal\": \"...\", \"Precondition\": \"...\", \"Key_Evidence\": \"...\", \"Acceptance_Criteria\": \"...\"}}"
            raw = self.llm.call("You are a precise oracle synthesizer. Output only valid JSON.", prompt, max_tokens=600)
            parsed = self.llm.parse_json(raw)
            st.oracle = parsed if isinstance(parsed, dict) else {}
            st.oracle["subtask_id"] = st.id
            generated_oracles.append(st.oracle)
        return subtasks

    def _hierarchical_backtrack(self, query, gt_section, history_text, n_steps, graph):
        prompt = f"Problem: {query}\n{gt_section}\nConversation:\n{history_text}\nCausal Graph:\n{graph.to_text()}\nPerform hierarchical backtracking.\nOutput format exactly:\nCandidate Error Subtasks: [Id1, Id2]\nCandidate Error Agents: [agent1]\nCandidate Error Steps: [1, 2]"
        raw = self.llm.call("You are a precise failure attribution analyst.", prompt, max_tokens=800)

        candidates = {"candidate_subtasks": [], "candidate_agents": [], "candidate_steps": []}
        for line in raw.split("\n"):
            if "Candidate Error Steps:" in line:
                candidates["candidate_steps"] = sorted([int(v) for v in re.findall(r"\d+", line)])
        if not candidates["candidate_steps"]: candidates["candidate_steps"] = list(range(1, n_steps + 1))
        return candidates

    # ── Stage 3 ─────────────────────────────────────────────
    def _counterfactual_attribute(self, query, gt_section, history_text, n_steps, steps, graph, candidates):
        prompt = f"Problem: {query}\n{gt_section}\nConversation:\n{history_text}\nCandidates:\n{json.dumps(candidates)}\nGraph:\n{graph.to_text()}\nIdentify the SINGLE most responsible reasoning mistake.\nOutput format exactly:\nAgent Name: name\nStep Number: int\nIs Reversible: YES/NO\nReason for Mistake: text"
        raw = self.llm.call("You are a precise causal attribution analyst.", prompt, max_tokens=700)

        result = {"responsible_agent": "", "error_step": -1, "reason": "", "is_reversible": False}
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("Agent Name:"):
                result["responsible_agent"] = line.split(":", 1)[1].strip().strip("()")
            elif line.startswith("Step Number:"):
                nums = re.findall(r'\d+', line)
                if nums: result["error_step"] = int(nums[0])
            elif line.startswith("Is Reversible:"):
                result["is_reversible"] = line.split(":", 1)[1].strip().upper().startswith("YES")
            elif line.startswith("Reason for Mistake:"):
                result["reason"] = line.split(":", 1)[1].strip()

        # Stage D 后处理
        if result["is_reversible"] and result["error_step"] >= 0:
            alt_steps = sorted([s for s in candidates.get("candidate_steps", []) if s != result["error_step"]])
            if alt_steps:
                result["error_step"] = alt_steps[0]
                step_idx = result["error_step"] - 1
                if 0 <= step_idx < len(steps): result["responsible_agent"] = steps[step_idx][1]
                result["reason"] = f"[Stage D filtered] {result['reason']}"
        return result