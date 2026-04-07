# -*- coding: UTF-8 -*-
'''
@Author  ：陈彩怡
@Date    ：2026/3/30 19:36
'''
import time
import json
import re
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LLMClient:
    def __init__(self, model: str, api_key: str = "", base_url: str = None, temperature: float = 0.0):
        """
        初始化 LLM 客户端，支持 OpenAI 兼容 API 和本地 HuggingFace 模型。
        """
        self.model = model
        self.temperature = temperature
        self._local_model = None
        self._tokenizer = None

        # 智能路由：判断是否走本地推理
        if model.startswith("local:"):
            self._init_local(model[6:])  # 去掉 "local:" 前缀，传入实际路径
        else:
            if OpenAI is None:
                raise ImportError("请先安装 openai: pip install openai")
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _init_local(self, model_path: str):
        """加载本地 HuggingFace 模型（如 Qwen2.5-7B）"""
        if not HF_AVAILABLE:
            raise ImportError("请先安装: pip install transformers torch accelerate")
        print(f"[本地模型] 正在加载 {model_path} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("[本地模型] 加载完成！")

    # 增加 temperature=None 参数
    def call(self, system: str, user: str, max_tokens: int = 1024, temperature: float = None) -> str:
        # 如果调用时没传，就用默认的 self.temperature；传了就用传进来的
        active_temp = temperature if temperature is not None else self.temperature

        if self._local_model is not None:
            return self._local_chat(system, user, max_tokens)
        else:
            return self._api_chat(system, user, max_tokens, active_temp)  # 传给底层

    # 接收 active_temp
    def _api_chat(self, system: str, user: str, max_tokens: int, active_temp: float) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=active_temp,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [API 调用报错，准备重试 {attempt + 1}/3] {e}")
                time.sleep(2 ** attempt)
        return ""

    def _local_chat(self, system: str, user: str, max_tokens: int) -> str:
        """本地模型推理逻辑"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._local_model.device)
        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                # 如果 temperature 设为 0，Transformers 库可能会报错，所以给个极小值 0.01 模拟贪婪解码
                temperature=0.01 if self.temperature == 0.0 else self.temperature,
                do_sample=False if self.temperature == 0.0 else True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        out_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    @staticmethod
    def parse_json(raw: str) -> dict:
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