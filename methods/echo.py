# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/31 19:31 
'''
# -*- coding: UTF-8 -*-
import json
import re
import random
from collections import defaultdict
from typing import List, Dict, Any, Optional

from data.dataset import get_query_and_gt, get_log_steps
from methods.base_method import BaseAttributionMethod

# ════════════════════════════════════════════════════════
# 角色与系统提示词配置
# ════════════════════════════════════════════════════════
ANALYST_ROLES = {
    "conservative": "You are a CONSERVATIVE analyst with HIGH confidence thresholds. Only attribute errors when you have strong, clear evidence. Prefer single-agent attributions over multi-agent ones. Be cautious about making attributions without definitive proof.",
    "liberal": "You are a LIBERAL analyst more willing to make attributions based on reasonable evidence. Consider multi-agent error scenarios and subtle errors that might be overlooked. Be open to making attributions even with moderate confidence.",
    "detail_focused": "You are a DETAIL-FOCUSED analyst. Examine specific evidence and exact wording. Identify subtle inconsistencies, minor logical gaps, and precise factual inaccuracies. Prioritize concrete evidence over general patterns.",
    "pattern_focused": "You are a PATTERN-FOCUSED analyst. Recognize broader reasoning chains and systemic issues. Track error propagation patterns, identify recurring themes, and analyze overall reasoning structure. Consider how errors propagate through the conversation.",
    "skeptical": "You are a SKEPTICAL analyst. Question all underlying assumptions. Look for alternative explanations, consider whether apparent errors might be valid reasoning. Challenge conventional attributions and examine if ground truth itself could be questioned.",
    "general": "You are a BALANCED GENERAL analyst with no specific specialization. Approach the analysis with a broad perspective, considering all types of evidence equally. Look for the most obvious and impactful mistakes based on objective evaluation.",
}

ANALYST_SYSTEM = """You are an Objective Analysis Agent conducting an impartial investigation to determine error attribution in a multi-agent conversation.

ANALYST SPECIALIZATION: {focus_instructions}

Your task:
1. Analyze ALL agents in the conversation objectively
2. Determine which agent(s) most likely caused the final wrong answer
3. Determine which step/turn the mistake first occurred
4. Provide confidence scores [0.0-1.0] and reasoning for your conclusions

The full conversation is provided below. A hierarchical context example (Step 0's view)
is appended at the end to illustrate how proximity affects detail level:
- Distance 1 (L1): full content preserved
- Distance 2-3 (L2): key decisions only
- Distance 4-6 (L3): brief summary
- Distance >6 (L4): milestones only
Use this structure as a guide when tracing error propagation — nearby agents have
richer detail, distant agents are compressed. Weight your evidence accordingly.

Possible conclusions:
- single_agent: One specific agent caused the mistake at a specific step
- multi_agent: Multiple agents contributed to the mistake across specific steps

Output your response as valid JSON wrapped in <json></json> tags:
<json>
{{
  "analysis_summary": "Brief overview of your investigation approach and findings",
  "agent_evaluations": [
    {{
      "agent_name": "name",
      "step_index": 0,
      "error_likelihood": 0.0,
      "reasoning": "Why this agent may or may not have caused the error",
      "evidence": "Specific evidence supporting your assessment"
    }}
  ],
  "primary_conclusion": {{
    "type": "single_agent",
    "attribution": ["agent_name"],
    "mistake_step": 0,
    "confidence": 0.0,
    "reasoning": "Detailed explanation including which step the error occurred"
  }},
  "alternative_hypotheses": [
    {{
      "type": "single_agent",
      "attribution": ["agent_name"],
      "mistake_step": 0,
      "confidence": 0.0,
      "reasoning": "Alternative explanation"
    }}
  ]
}}
</json>

Be thorough and objective. Pay special attention to identifying the specific step where the error first occurred."""

MIN_CONFIDENCE = 0.3
ALL_ANALYST_ROLES = list(ANALYST_ROLES.keys())


# ════════════════════════════════════════════════════════
# ECHO 方法主类
# ════════════════════════════════════════════════════════
class ECHOMethod(BaseAttributionMethod):
    def __init__(self, llm_client, k_analysts: int = 3):
        super().__init__(llm_client)
        self.k_analysts = k_analysts

    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        query, gt_section = get_query_and_gt(instance, use_ground_truth)
        steps = get_log_steps(instance)

        if not steps:
            return {"responsible_agent": "", "error_step": -1, "reason": "fallback: no structured steps"}

        # 提取最后一步作为 final_answer
        final_answer = steps[-1][2][:300] if steps else ""
        conv_summary = self._build_conversation_summary(steps)
        n_steps = len(steps)

        # 随机采样 k 个 analyst 及其专属 temperature
        sampled_roles = random.sample(ALL_ANALYST_ROLES, k=min(self.k_analysts, len(ALL_ANALYST_ROLES)))
        temperatures = [round(random.uniform(0.3, 0.9), 1) for _ in sampled_roles]

        if verbose: print(f"\n[ECHO] 采样分析师: {sampled_roles} | 温度: {temperatures}")

        agent_analyses = []
        step_analyses = []

        # 并行/串行执行多专家解耦分析
        for idx, (role, temp) in enumerate(zip(sampled_roles, temperatures)):
            if verbose: print(f"  -> [{role}] 正在执行 Agent-level 分析...")
            a_result = self._run_objective_analyst(role, temp, conv_summary, query, gt_section, final_answer, "agent")
            if a_result:
                agent_analyses.append(a_result)

            if verbose: print(f"  -> [{role}] 正在执行 Step-level 分析...")
            s_result = self._run_objective_analyst(role, temp, conv_summary, query, gt_section, final_answer, "step")
            if s_result:
                step_analyses.append(s_result)

        # 汇总投票
        if verbose: print("[ECHO] 正在进行共识投票...")
        consensus = self._consensus_voting(agent_analyses, step_analyses, n_steps)
        cc = consensus["consensus_conclusion"]

        attribution = cc.get("attribution") or []
        error_step = cc.get("mistake_step")

        return {
            "responsible_agent": attribution[0] if attribution else "",
            "error_step": int(error_step) if error_step is not None else -1,
            "reason": f"ECHO consensus confidence: {cc.get('agent_confidence', 0.0):.2f}. Top agent votes: {dict(sorted(consensus['voting_details']['agent_name_votes'].items(), key=lambda x: -x[1])[:3])}",
            "_echo_details": {
                "sampled_roles": sampled_roles,
                "consensus_full": cc,
            }
        }

    def _build_conversation_summary(self, steps: list) -> str:
        lines = ["=== CONVERSATION AGENTS ==="]
        for sid, name, content in steps:
            lines.extend([f"Step {sid} - {name}:", content, ""])
        return "\n".join(lines)

    def _parse_echo_json(self, raw: str) -> Optional[Dict]:
        match = re.search(r"<json>(.*?)</json>", raw, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        # 回退到框架强大的通用解析器
        return self.llm.parse_json(raw)

    def _run_objective_analyst(self, role: str, temp: float, conv_summary: str, problem: str, gt_section: str,
                               final_answer: str, attr_type: str) -> dict:
        focus = ANALYST_ROLES.get(role, ANALYST_ROLES["general"])
        system = ANALYST_SYSTEM.format(focus_instructions=focus)

        task_focus = "FOCUS ON AGENT-LEVEL ATTRIBUTION: Determine WHICH AGENT is most responsible." if attr_type == "agent" else "FOCUS ON STEP-LEVEL ATTRIBUTION: Determine at WHICH STEP the mistake first occurred."
        user_msg = f"Original Problem: {problem}\n{gt_section}\nFinal (Wrong) Answer: {final_answer}\n\n{conv_summary}\n\n{task_focus}\n\nConduct an objective analysis and output your findings wrapped in <json> tags."

        # ⚠️ 这里调用了 LLM 客户端，并传入了动态的 temperature
        raw = self.llm.call(system, user_msg, max_tokens=1500, temperature=temp)

        result = self._parse_echo_json(raw)
        if not result:
            result = {"primary_conclusion": {"type": "single_agent", "attribution": None, "mistake_step": -1,
                                             "confidence": 0.0, "reasoning": raw[:100]}}

        result["analyst_role"] = role
        result["attribution_type"] = attr_type
        return result

    def _consensus_voting(self, agent_analyses: List[Dict], step_analyses: List[Dict], n_steps: int) -> Dict[str, Any]:
        agent_conclusion_votes = defaultdict(list)
        agent_name_votes = defaultdict(float)

        for analysis in agent_analyses:
            pc = analysis.get("primary_conclusion", {})
            conf = float(pc.get("confidence", 0.0))
            if conf >= MIN_CONFIDENCE:
                agent_conclusion_votes[pc.get("type", "single_agent")].append(pc)
                for name in (pc.get("attribution") or []):
                    if name: agent_name_votes[name] += conf

        best_agent_score, best_agent_type, best_agent_votes = 0.0, "single_agent", []
        for ctype, votes in agent_conclusion_votes.items():
            total_conf = sum(v.get("confidence", 0) for v in votes)
            if total_conf > best_agent_score:
                best_agent_score, best_agent_type, best_agent_votes = total_conf, ctype, votes

        final_agent_attribution = None
        if best_agent_votes and agent_name_votes:
            sorted_agents = sorted(agent_name_votes.items(), key=lambda x: x[1], reverse=True)
            final_agent_attribution = [sorted_agents[0][0]] if best_agent_type == "single_agent" else [n for n, c in
                                                                                                       sorted_agents if
                                                                                                       c >= MIN_CONFIDENCE]

        step_votes_dict = defaultdict(float)
        for analysis in step_analyses:
            pc = analysis.get("primary_conclusion", {})
            conf = float(pc.get("confidence", 0.0))
            mistake_step = pc.get("mistake_step")
            if conf >= MIN_CONFIDENCE and mistake_step is not None:
                try:
                    s = int(mistake_step)
                    # 此处用 n_steps 验证合法性 (步骤从 1 开始)
                    if 0 < s <= n_steps: step_votes_dict[s] += conf
                except:
                    pass

        consensus_step = sorted(step_votes_dict.items(), key=lambda x: x[1], reverse=True)[0][
            0] if step_votes_dict else None
        avg_agent_conf = sum(v.get("confidence", 0) for v in best_agent_votes) / max(1, len(best_agent_votes))

        return {
            "consensus_conclusion": {"type": best_agent_type, "attribution": final_agent_attribution,
                                     "mistake_step": consensus_step, "agent_confidence": avg_agent_conf},
            "voting_details": {"agent_name_votes": dict(agent_name_votes), "step_votes": dict(step_votes_dict)}
        }


# --------------------------------------------------------
# 附录：保留原论文的 L1-L4 分层函数（供后续 I2 扩展使用）
# --------------------------------------------------------
def _extract_key_decision(content, max_words=50, context_type="general"): pass


def _summarize_agent(content, max_words=20, context_type="general"): pass


def _obtain_milestones(content, max_words=15, context_type="general"): pass