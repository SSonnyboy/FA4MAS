# FA4MAS

将 `CHIEF` 的实验执行逻辑重构到 `FA4MAS` 下，核心目标是：

- `core/` 只负责配置、LLM、结果写出和运行编排
- `methods/` 只负责具体方法实现
- `methods/baselines/` 聚合不同 baseline
- `methods/chief/` 保留 CHIEF 的多阶段推理流程

## 目录结构

```text
FA4MAS
├── core
├── methods
│   ├── baselines
│   └── chief
├── configs
├── data
└── run_experiment.py
```

## 运行方式

```bash
cd /home/chenyu/workplace/FA4MAS
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY（不要写到 configs/*.json）
set -a && . ./.env && set +a
python run_experiment.py --config configs/chief.json
python run_experiment.py --config configs/echo.json
python run_experiment.py --config configs/baseline.json
python run_experiment.py --config configs/all_at_once.json
python run_experiment.py --config configs/binary_search.json
python run_experiment.py --config configs/step_by_step.json
```

## 密钥安全

- API Key 请仅通过环境变量传入（`OPENAI_API_KEY`）
- 不要在 `configs/*.json` 中填写真实密钥
- `.env` 已被 `.gitignore` 忽略，可安全用于本地开发

## 方法名

- `baseline`
- `all_at_once`
- `binary_search`
- `step_by_step`
- `chief`
- `echo`

## 说明

`chief` 方法中的 RAG 检索器为可选组件：

- 如果本地存在旧版 `CHIEF/rag` 索引且依赖齐全，会自动启用
- 如果缺少 `faiss`、`sentence_transformers` 或索引文件，则自动退化为无检索模式

框架默认会在 summary 中统计 `Step-Level with Tolerance`：

- `step_accuracy_with_tolerance` 默认包含 `±1` 到 `±5`
- 可通过配置 `method_params.step_tolerance_max` 或 `method_params.step_tolerances` 自定义

Debug 模式默认会随机抽样 `debug_limit` 条样本（无需额外配置）：

```json
{
  "debug_mode": true,
  "debug_limit": 10
}
```
