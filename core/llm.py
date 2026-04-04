"""与 OpenAI 兼容接口交互的轻量封装。"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing multi-agent conversations."


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError("Missing OPENAI_API_KEY. Please set it before running.")

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and base_url.strip():
        return OpenAI(api_key=api_key.strip(), base_url=base_url.strip())
    return OpenAI(api_key=api_key.strip())


def chat_completion(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int | None = None,
) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    request: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request)
    message = response.choices[0].message
    content = getattr(message, "content", None)

    if content is None or (isinstance(content, str) and not content.strip()):
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            content = reasoning

    return (content or "").strip()

