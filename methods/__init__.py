"""方法注册表。"""
from __future__ import annotations

from typing import Dict, Type

from openai import OpenAI

from core.config import ExperimentConfig
from .base import BaseMethod
from .baselines.all_at_once import AllAtOnceBaselineMethod
from .baselines.baseline import FullTrajectoryBaselineMethod
from .baselines.binary_search import BinarySearchBaselineMethod
from .baselines.step_by_step import StepByStepBaselineMethod
from .blade.method import BLADEMethod
from .chief.method import CHIEFMethod
from .echo.method import ECHOMethod


METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {
    "baseline": FullTrajectoryBaselineMethod,
    "all_at_once": AllAtOnceBaselineMethod,
    "binary_search": BinarySearchBaselineMethod,
    "step_by_step": StepByStepBaselineMethod,
    "blade": BLADEMethod,
    "chief": CHIEFMethod,
    "echo": ECHOMethod,
}


def create_method(client: OpenAI, config: ExperimentConfig) -> BaseMethod:
    method_key = config.method.lower()
    if method_key not in METHOD_REGISTRY:
        supported = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unsupported method: {config.method}. Available methods: {supported}")
    method_cls = METHOD_REGISTRY[method_key]
    return method_cls(client=client, config=config)
