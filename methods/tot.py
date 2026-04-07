# -*- coding: UTF-8 -*-
from collections import Counter
from data.dataset import get_log_steps, get_query_and_gt, format_log
from methods.base_method import BaseAttributionMethod, SYSTEM_PROMPT

# =====================================================================
# ToT 专属的 Prompts
# =====================================================================
SYSTEM_PROMPT_SCORE = """You are an expert debugger for LLM-based multi-agent systems.
You will evaluate ONE step from a failed multi-agent task and rate its suspicion level.
Always respond in this exact JSON format (nothing else):
{
  "score": <integer 1-10>,
  "label": "<fine|likely|decisive>",
  "reason": "<one sentence explanation>"
}"""

SYSTEM_PROMPT_CF = """You are an expert debugger for LLM-based multi-agent systems.
You will perform counterfactual analysis on one step of a failed task.
Always respond in this exact JSON format (nothing else):
{
  "would_succeed": <true|false>,
  "correct_action": "<brief description of what should have been done>",
  "reasoning": "<2-3 sentences explaining your conclusion>"
}"""

SUSPICION_PROMPT = """You are scoring one step in a failed multi-agent task.
Task: {query}{gt_section}

=== Step to evaluate ===
Step {step_id} | Agent: {agent}
{content}

=== Recent context ===
{context}

Rate suspicion 1-10:
  1-3  = "fine"     — looks correct
  4-6  = "likely"   — questionable
  7-10 = "decisive" — very likely caused the failure

Respond ONLY with JSON (no extra text):
{{"score": <1-10>, "label": "<fine|likely|decisive>", "reason": "<one sentence>"}}
"""

COUNTERFACTUAL_PROMPT = """Verify whether a step is the decisive error in a failed task.

Task: {query}{gt_section}

=== Full log ===
{full_log}

=== Hypothesis ===
Step {step_id} (Agent: {agent}) is the decisive error.
Suspected issue: {reason}

If agent "{agent}" had taken the CORRECT action at step {step_id}, would the task
have ultimately SUCCEEDED? Consider whether earlier errors prevent recovery.

Respond ONLY with JSON:
{{"would_succeed": <true|false>, "correct_action": "<brief>", "reasoning": "<2-3 sentences>"}}"""

VOTE_PROMPT = """choose the most likely decisive error from the candidates below.
Task: {query}{gt_section}
=== Candidates ===
{candidates}
Which is the ACTUAL decisive error? 
If a clear causal chain exists between candidates (earlier error directly caused later mistakes), prefer the earliest. 
Otherwise prefer the highest suspicion score. Add "causal_chain":true/false to your JSON.

Respond in the standard JSON format:
{{
  "responsible_agent": "<agent name>",
  "error_step": <step number as integer>,
  "reason": "<brief explanation>",
  "causal_chain": <true|false>
}}
"""


# =====================================================================
# ToT 算法类实现
# =====================================================================
class ToTMethod(BaseAttributionMethod):
    def __init__(self, llm_client, top_k: int, vote_rounds: int, threshold: int):
        super().__init__(llm_client)
        self.top_k = top_k
        self.vote_rounds = vote_rounds
        self.threshold = threshold

    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        query, gt_section = get_query_and_gt(instance, use_ground_truth)
        all_steps = get_log_steps(instance)

        if not all_steps:
            log_text = format_log(instance)
            user = f"Task: {query}{gt_section}\n\nLog:\n{log_text}\n\nIdentify responsible agent and decisive error step."
            result = self.llm.parse_json(self.llm.call(SYSTEM_PROMPT, user))
            result.setdefault("responsible_agent", "")
            result.setdefault("error_step", -1)
            result.setdefault("reason", "fallback: no structured steps")
            return result

        if verbose:
            print(f"\n{'=' * 65}")
            print(f"  Q: {query[:100]}{'...' if len(query) > 100 else ''}")
            print(f"  Steps: {len(all_steps)}")
            print(f"{'=' * 65}")

        # Phase 1
        if verbose: print(f"[Phase 1] Scoring {len(all_steps)} steps...")
        scores = []
        for step_id, agent, content in all_steps:
            s = self._score_step(step_id, agent, content, all_steps, query, gt_section)
            try:
                score_val = int(float(str(s.get("score", 0))))
            except (ValueError, TypeError):
                score_val = 0
            score_val = max(0, min(10, score_val))
            scores.append({"step_id": step_id, "agent": agent,
                           "score": score_val, "label": s.get("label", ""),
                           "reason": s.get("reason", "")})
            if verbose:
                bar = "█" * score_val + "░" * (10 - score_val)
                print(f"  [{bar}] {score_val:2d}/10  Step {step_id:2d} | {agent[:26]:26s} | {s.get('label', '')}")
                print(f"         {s.get('reason', '')}")

        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        top_suspects = [s for s in sorted_scores if s["score"] >= self.threshold][:self.top_k]
        if not top_suspects:
            top_suspects = sorted_scores[:1]
        if verbose:
            print(f"\n  -> {len(top_suspects)} suspect(s) (threshold >= {self.threshold})")

        # Phase 2
        if verbose: print(f"\n[Phase 2] Counterfactual verification...")
        verified = []
        for s in top_suspects:
            v = self._verify_step(s["step_id"], s["agent"], s["reason"],
                                  all_steps, query, gt_section)
            would = bool(v.get("would_succeed", False))
            verified.append((s["step_id"], s["agent"], would,
                             s["score"], v.get("reasoning", s["reason"])))
            if verbose:
                status = "DECISIVE ✓" if would else "not decisive ✗"
                print(f"  Step {s['step_id']:2d} | {s['agent'][:26]:26s} | {status}")
                print(f"         {v.get('reasoning', '')}")

        # Phase 3
        if verbose: print(f"\n[Phase 3] Voting ({self.vote_rounds} rounds)...")
        all_false = all(not v[2] for v in verified)
        if all_false:
            best = max(verified, key=lambda v: v[3])
            final = {
                "responsible_agent": best[1],
                "error_step": best[0],
                "reason": best[4],
                "_confidence": 0.0,
            }
        else:
            final = self._vote(verified, query, gt_section)

        result = {
            "responsible_agent": final.get("responsible_agent", ""),
            "error_step": final.get("error_step", -1),
            "reason": final.get("reason", ""),
            "_confidence": final.get("_confidence", 0.0),
            "_phase_details": {
                "all_scores": [(s["step_id"], s["agent"], s["score"], s["label"]) for s in scores],
                "top_suspects": [(s["step_id"], s["agent"], s["score"]) for s in top_suspects],
                "counterfactuals": [(v[0], v[1], v[2]) for v in verified],
            },
        }
        if verbose:
            print(f"\n{'─' * 65}")
            print(f"  Result: agent='{result['responsible_agent']}' "
                  f"step={result['error_step']} conf={result['_confidence']:.0%}")
            print(f"{'=' * 65}")
        return result

    # ─────────────────────────────────────────────────────────────────
    # 私有辅助方法：调用 self.llm 实例
    # ─────────────────────────────────────────────────────────────────
    def _score_step(self, step_id, agent, content, all_steps, query, gt_section):
        prior = [s for s in all_steps if s[0] < step_id][-5:]#这里是取当前步骤的前五条
        context = "\n".join(
            f"[Step {s[0]}] {s[1]}: {s[2][:200]}{'...' if len(s[2]) > 200 else ''}"
            for s in prior
        ) or "(no prior steps)"
        user = (SUSPICION_PROMPT
                .replace("{query}", query)
                .replace("{gt_section}", gt_section)
                .replace("{step_id}", str(step_id))
                .replace("{agent}", agent)
                .replace("{context}", context))

        # 使用类的 LLM 客户端
        response = self.llm.call(SYSTEM_PROMPT_SCORE, user, max_tokens=200)
        return self.llm.parse_json(response)

    def _verify_step(self, step_id, agent, reason, all_steps, query, gt_section):
        full_log = "\n".join(
            f"[Step {s[0]}] {s[1]}: {s[2][:300]}{'...' if len(s[2]) > 300 else ''}"
            for s in all_steps
        )
        user = (COUNTERFACTUAL_PROMPT
                .replace("{query}", query)
                .replace("{gt_section}", gt_section)
                .replace("{full_log}", full_log)
                .replace("{step_id}", str(step_id))
                .replace("{agent}", agent)
                .replace("{reason}", reason))

        response = self.llm.call(SYSTEM_PROMPT_CF, user, max_tokens=300)
        return self.llm.parse_json(response)

    def _vote(self, candidates, query, gt_section):
        passed = [c for c in candidates if c[2]] or candidates
        cand_text = "\n".join(
            f"- Step {c[0]} | Agent: {c[1]} | "
            f"Counterfactual: {'yes' if c[2] else 'no'} | Suspicion: {c[3]}/10\n  {c[4]}"
            for c in passed
        )
        user = (VOTE_PROMPT
                .replace("{query}", query)
                .replace("{gt_section}", gt_section)
                .replace("{candidates}", cand_text))

        # 替换 VOTE_ROUNDS 为 self.vote_rounds，并使用类的 LLM 客户端
        votes = []
        for _ in range(self.vote_rounds):
            resp = self.llm.call(SYSTEM_PROMPT, user, max_tokens=200)
            votes.append(self.llm.parse_json(resp))

        vote_counts = Counter(
            (v.get("responsible_agent", ""), v.get("error_step", -1)) for v in votes
        )
        winner, win_count = vote_counts.most_common(1)[0]

        # 修复了生成器为空可能引发的 StopIteration 异常风险
        winner_vote = next(
            (v for v in votes if v.get("responsible_agent") == winner[0] and v.get("error_step") == winner[1]),
            votes[0] if votes else {}
        )

        no_chain = sum(1 for v in votes if not v.get("causal_chain", True)) > len(votes) / 2
        if no_chain:
            best = max(passed, key=lambda c: c[3])
            winner = (best[1], best[0])
            winner_vote = next((v for v in votes if v.get("responsible_agent") == winner[0]), winner_vote)

        winner_vote["_confidence"] = win_count / self.vote_rounds if self.vote_rounds > 0 else 0
        return winner_vote