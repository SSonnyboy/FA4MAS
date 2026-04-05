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
python run_experiment.py --config configs/chief.json
python run_experiment.py --config configs/baseline.json
python run_experiment.py --config configs/all_at_once.json
python run_experiment.py --config configs/binary_search.json
python run_experiment.py --config configs/step_by_step.json
```

## 方法名

- `baseline`
- `baseline_all_at_once`
- `baseline_binary_search`
- `baseline_step_by_step`
- `chief`

## 说明

`chief` 方法中的 RAG 检索器为可选组件：

- 如果本地存在旧版 `CHIEF/rag` 索引且依赖齐全，会自动启用
- 如果缺少 `faiss`、`sentence_transformers` 或索引文件，则自动退化为无检索模式

