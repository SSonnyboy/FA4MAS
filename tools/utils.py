"""通用辅助函数"""

import json
import re
from typing import Any, Optional, List, Dict


def format_history(history: List[Dict]) -> str:
    """将对话历史格式化为可读文本"""
    lines = []
    for i, msg in enumerate(history):
        name = msg.get("name") or msg.get("agent") or msg.get("role") or f"Agent{i}"
        content = msg.get("content") or msg.get("message") or str(msg)
        lines.append(f"[Step {i}] {name}: {content[:200]}")
    return "\n".join(lines)


def parse_json_safe(text: str) -> Optional[Any]:
    """安全解析JSON，支持从LLM输出中提取JSON"""
    if not text:
        return None

    cleaned = re.sub(r'```(?:json)?', '', text).strip().strip('`')

    try:
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    try:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return None


def extract_error_keywords(content: str) -> List[str]:
    """提取内容中的错误关键词"""
    keywords = []
    error_patterns = [
        r'error', r'exception', r'fail', r'wrong', r'incorrect',
        r'bug', r'issue', r'problem', r'invalid'
    ]

    for pattern in error_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            keywords.append(pattern)

    return keywords


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的简单相似度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)
