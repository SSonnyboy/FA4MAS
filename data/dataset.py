# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/30 19:33 
'''
import os, sys, json, re, time, argparse
from pathlib import Path
from typing import List, Dict

def load_dataset(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    instances = []

    single_file = data_dir / "instances.json"
    if single_file.exists():
        with open(single_file, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, list) else raw.get("data", [])

    json_files = sorted(
        [f for f in data_dir.iterdir() if f.is_file() and f.suffix == ".json"],
        key=lambda f: int(f.stem) if f.stem.isdigit() else 0
    )
    if json_files:
        for jf in json_files:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            data["task_id"] = jf.stem
            instances.append(data)
        print(f"[数据] 格式A（平铺json）：读取 {len(instances)} 个文件")
        return instances

    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for fname in ("log.json", "data.json", "history.json"):
            log_file = task_dir / fname
            if log_file.exists():
                with open(log_file, encoding="utf-8") as f:
                    data = json.load(f)
                data["task_id"] = task_dir.name
                instances.append(data)
                break

    if not instances:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到数据。请确认目录结构（参考 baseline 注释）。"
        )
    print(f"[数据] 格式B（子目录）：读取 {len(instances)} 条")
    return instances


def format_log(instance: dict) -> str:
    """格式化日志为字符串，过滤 Computer_terminal 和 TERMINATE（与 get_log_steps 一致）"""
    steps = get_log_steps(instance)
    if steps:
        return "\n".join(f"[Step {sid}] {name}: {content}" for sid, name, content in steps)
    # fallback：无结构化 history 时直接用原始文本
    raw = (instance.get("log") or instance.get("failure_log") or "")
    return str(raw)


def get_log_steps(instance: dict) -> list:
    """
    返回 (step_1indexed, agent_name, content) 列表。
    过滤规则（与数据格式对齐）：
      - Computer_terminal：代码执行器，不是决策 agent，不计入步骤
      - TERMINATE 消息：终止信令，不是实质性动作
    过滤后 step_id 连续重编（从 1 开始），与 evaluate 的 mistake_step+1 对齐。
    """
    SKIP_AGENT_NAMES = {"computer_terminal"}  # 小写匹配

    log = instance.get("history")
    if not isinstance(log, list):
        return []
    steps = []
    step_id = 0
    for msg in log:
        name    = msg.get("name")
        role    = msg.get("role")
        content = (msg.get("content") or msg.get("message") or "").strip()

        # 过滤 Computer_terminal
        # if name.lower().replace(" ", "_") in SKIP_AGENT_NAMES:
        #     continue
        # 过滤纯 TERMINATE 消息
        if content == "TERMINATE":
            continue

        step_id += 1
        steps.append((step_id, name, content))
    return steps


def get_query_and_gt(instance: dict, use_ground_truth: bool) -> tuple:
    query = (instance.get("question") or instance.get("query")
             or instance.get("task") or "")
    gt_section = ""
    if use_ground_truth:
        answer = instance.get("ground_truth") or instance.get("answer") or ""
        if answer:
            gt_section = f"\nGround truth answer: {answer}"
    return query, gt_section

def _parse_json(raw: str) -> dict:
    """
    通用 JSON 提取：先去 markdown 包裹，再用贪婪正则找最外层 {}，
    解析失败则用正则逐字段回退（兼容 responsible_agent 和 score 两种格式）。
    """
    if not raw:
        return {}
    # 去掉 ```json ... ``` 或 ``` ... ``` 包裹
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    # 贪婪匹配最外层 JSON 对象
    try:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    # 正则回退：兼容两种响应格式
    result = {}
    # Phase 3 / fallback 格式
    for pat, key in [
        (r'"responsible_agent"\s*:\s*"([^"]+)"', "responsible_agent"),
        (r'"reason"\s*:\s*"([^"]*)"',            "reason"),
    ]:
        m = re.search(pat, cleaned)
        if m:
            result[key] = m.group(1)
    step_m = re.search(r'"error_step"\s*:\s*(\d+)', cleaned)
    if step_m:
        result["error_step"] = int(step_m.group(1))
    # Phase 1 格式
    score_m = re.search(r'"score"\s*:\s*"?(\d+)"?', cleaned)
    if score_m:
        result["score"] = int(score_m.group(1))
    label_m = re.search(r'"label"\s*:\s*"([^"]+)"', cleaned)
    if label_m:
        result["label"] = label_m.group(1)
    # Phase 2 格式
    succ_m = re.search(r'"would_succeed"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    if succ_m:
        result["would_succeed"] = succ_m.group(1).lower() == "true"
    reas_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', cleaned)
    if reas_m:
        result["reasoning"] = reas_m.group(1)
    if not result:
        result["reason"] = cleaned[:150]
    return result