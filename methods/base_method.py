# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/30 19:43 
'''
from abc import ABC, abstractmethod
from core.llm_client import LLMClient

# 这是所有归因方法（基线、CoT、ToT）都通用的主系统提示词
SYSTEM_PROMPT = """You are an expert debugger for LLM-based multi-agent systems.
Your task is to identify:
1. Which agent caused the task failure (agent-level attribution)
2. At which step the decisive error occurred (step-level attribution)

The decisive error is the EARLIEST action whose correction would flip the outcome from failure to success.

Always respond in this exact JSON format:
{
  "responsible_agent": "<agent name>",
  "error_step": <step number as integer>,
  "reason": "<brief explanation>"
}"""

class BaseAttributionMethod(ABC):
    def __init__(self, llm_client: LLMClient):
        """
        所有归因算法基类，强制要求传入实例化的 LLMClient。
        """
        self.llm = llm_client

    @abstractmethod
    def run_attribution(self, instance: dict, use_ground_truth: bool, verbose: bool = True) -> dict:
        """
        子类必须实现此方法。
        """
        pass


