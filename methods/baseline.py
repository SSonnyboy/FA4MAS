# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/30 20:54 
'''
# -*- coding: UTF-8 -*-
from data.dataset import format_log
from methods.base_method import BaseAttributionMethod, SYSTEM_PROMPT


class AllAtOnceMethod(BaseAttributionMethod):
    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        log_text = format_log(instance)
        query = instance.get("question") or instance.get("query") or instance.get("task") or ""

        gt = ""
        if use_ground_truth:
            answer = instance.get("ground_truth") or instance.get("answer") or ""
            if answer: gt = f"\n\nGround truth answer: {answer}"

        user_msg = f"Task query: {query}\n\nFailure log:\n{log_text}\n{gt}\n\nIdentify the responsible agent and decisive error step."

        if verbose: print(f"[All-at-Once] 分析完整日志中...")
        response = self.llm.call(SYSTEM_PROMPT, user_msg, max_tokens=300)
        return self.llm.parse_json(response)


class StepByStepMethod(BaseAttributionMethod):
    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        log = instance.get("history")
        # print(f"【DEBUG】log 类型: {type(log)}, 内容预览: {str(log)[:300]}")
        query = instance.get("question")
        gt = ""
        if use_ground_truth:
            answer = instance.get("ground_truth")
            if answer: gt = f"\nGround truth answer: {answer}"

        # 如果没有结构化日志，降级为 All-at-Once
        if not isinstance(log, list) or len(log) == 0:
            if verbose: print("[Step-by-Step] 缺少结构化步骤，降级为 All-at-Once")
            fallback_method = AllAtOnceMethod(self.llm)
            return fallback_method.run_attribution(instance, use_ground_truth, verbose)

        history = []
        for i, msg in enumerate(log):
            name = msg.get("name")
            content = msg.get("content")
            history.append(f"[Step {i + 1}] {name}: {content}")

            user_msg = f"Task query: {query}{gt}\n\nConversation so far:\n{chr(10).join(history)}\n\nHas a decisive error occurred at Step {i + 1}?\n- If YES: respond in JSON with responsible_agent, error_step={i + 1}, reason\n- If NO: respond exactly with {{\"decisive_error_found\": false}}"

            if verbose: print(f"[Step-by-Step] 检查 Step {i + 1}...")
            response = self.llm.call(SYSTEM_PROMPT, user_msg, max_tokens=200)

            if "decisive_error_found" not in response.lower():
                result = self.llm.parse_json(response)
                if result.get("responsible_agent"):
                    if verbose: print(f"  -> 发现错误！锁定在 Step {i + 1}")
                    return result

        if verbose: print("  -> 遍历完毕未发现错误，归咎于最后一步")
        return {"responsible_agent": name, "error_step": len(log), "reason": "Error at final step"}


class BinarySearchMethod(BaseAttributionMethod):
    def run_attribution(self, instance: dict, use_ground_truth: bool = False, verbose: bool = True) -> dict:
        log = instance.get("history")
        query = instance.get("question")
        gt = ""
        if use_ground_truth:
            answer = instance.get("ground_truth") or instance.get("answer") or ""
            if answer: gt = f"\nGround truth answer: {answer}"

        if not isinstance(log, list) or len(log) == 0:
            if verbose: print("[Binary-Search] 缺少结构化步骤，降级为 All-at-Once")
            return AllAtOnceMethod(self.llm).run_attribution(instance, use_ground_truth, verbose)

        steps = [(i + 1, msg.get("name"),
                  msg.get("content")) for i, msg in enumerate(log)]
        lo, hi = 0, len(steps) - 1

        while lo < hi:
            mid = (lo + hi) // 2
            segment = steps[lo: mid + 1]
            seg_text = "\n".join(f"[Step {s}] {r}: {c}" for s, r, c in segment)

            user_msg = f"Task query: {query}{gt}\n\nExamine Steps {steps[lo][0]} to {steps[mid][0]}:\n{seg_text}\n\nDoes the decisive error occur in this segment (Steps {steps[lo][0]}-{steps[mid][0]})?\nRespond ONLY with: {{\"in_segment\": true}} or {{\"in_segment\": false}}"

            if verbose: print(f"[Binary-Search] 检查区间 Step {steps[lo][0]} - {steps[mid][0]} ...")
            response = self.llm.call(SYSTEM_PROMPT, user_msg, max_tokens=50)

            if "true" in response.lower():
                hi = mid
            else:
                lo = mid + 1

        final_step, final_agent, final_content = steps[lo]
        log_text = "\n".join(f"[Step {s}] {r}: {c}" for s, r, c in steps)
        user_msg = f"Task query: {query}{gt}\n\nFull log:\n{log_text}\n\nThe decisive error is at Step {final_step}. Confirm the responsible agent and reason.\nRespond in JSON: responsible_agent, error_step, reason"

        if verbose: print(f"[Binary-Search] 锁定 Step {final_step}，提取最终原因...")
        response = self.llm.call(SYSTEM_PROMPT, user_msg, max_tokens=200)
        result = self.llm.parse_json(response)
        if not result.get("error_step") or result.get("error_step") == -1:
            result["error_step"] = final_step
        return result