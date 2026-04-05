"""方法抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

from core.config import ExperimentConfig


class BaseMethod(ABC):
    def __init__(self, client: OpenAI, config: ExperimentConfig) -> None:
        self.client = client
        self.config = config
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def params(self) -> Dict[str, Any]:
        return self.config.method_params

    @abstractmethod
    def process_sample(self, file_path: Path, *, index: int) -> Dict[str, Any]:
        raise NotImplementedError

