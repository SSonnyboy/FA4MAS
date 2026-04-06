# FA4MAS 错误归因方法源码报告（含 Baseline 全景）

本文基于当前仓库源码撰写，重点解释多智能体错误归因方法如何处理输入日志并输出责任 agent/step。核心方法注册见 `methods/__init__.py`：`baseline`、`all_at_once`、`binary_search`、`step_by_step`、`chief`、`echo`。

## 1. 统一任务定义与输入输出接口

### 1.1 输入样本结构（所有方法共享）

各方法都从单条 JSON 样本读取以下字段（字段名允许一定兼容）：

- `question`: 原问题
- `history`: 多智能体对话轨迹（按 step 顺序）
- `ground_truth`/`answer`: 任务正确答案（部分方法会选择是否注入提示词）
- `mistake_agent`, `mistake_step`（或 `gt.agent`, `gt.step`）: 标注真值

公共抽象在：

- `methods/base.py`（方法基类）
- `methods/baselines/common.py`（baseline 共享的 GT 提取、输出解析、记录组装）
- `core/runner.py`（逐样本调度 + 指标统计）

### 1.2 输出目标

所有方法最终都会归一化到：

- 预测责任体：`pred.agent`
- 预测出错步：`pred.step`
- 评估：`acc_agent`, `acc_step`

并写入统一结果文件（`core/results.py`）。

---

## 2. 统一“拟造测试样例”用于方法讲解

下面用一个最小可解释日志模拟输入，展示每种方法如何处理。

### 2.1 样例任务

**问题**：用户要计算“订单总额（不含税）”。  
**正确答案**：`100`  
**多智能体最终答案**：`108`（错误）  
**真值标注**：`mistake_agent = "Planner_Agent"`, `mistake_step = 0`

### 2.2 样例 history（简化）

```json
[
  {"role": "assistant", "name": "Planner_Agent", "content": "总额=商品小计+8%税。先按含税口径算。"},
  {"role": "assistant", "name": "DB_Agent", "content": "数据库查到商品小计为 100。"},
  {"role": "assistant", "name": "Calc_Agent", "content": "按 Planner 公式，总额=100*1.08=108。"},
  {"role": "assistant", "name": "Reviewer_Agent", "content": "计算链路自洽，结果 108。"},
  {"role": "assistant", "name": "Final_Agent", "content": "最终答案：108。"}
]
```

这个样例里，**根因是 step0 的任务定义偏差**：题目要求“不含税”，但 Planner 引入了税。

---

## 3. Baseline 方法详解

## 3.1 `baseline`（Full Trajectory, 单次全轨迹）

源码：`methods/baselines/baseline.py`

### 主要思路

一次性把完整 `history` 喂给模型，让模型直接输出：

- `Agent Name`
- `Step Number`
- `Reason for Mistake`

### 在样例上的处理

模型看到全链路后，通常会直接将责任定位到 step0（Planner 引入错误口径）。

### 特点

- **优点**：实现最简单、调用次数最少（通常 1 次）
- **缺点**：长上下文下易受噪声影响；没有中间结构，排障可解释性较弱

---

## 3.2 `all_at_once`（两阶段：先分任务再归因）

源码：`methods/baselines/all_at_once.py`

### 主要思路

分两跳：

1. `generate_subtasks`：把长轨迹压缩为语义子任务（含 `Step Range` + `Oracle`）
2. `predict_error`：基于“原始 history + 子任务摘要”做最终定位

### 在样例上的处理（示意）

阶段 1 可能产出：

- S1: 理解计价口径（step 0）
- S2: 抽取数据（step 1）
- S3: 执行计算（step 2）
- S4: 复核与输出（step 3-4）

阶段 2 再据此判断：S1 的口径错误导致后续传播，最终预测 `Planner_Agent@step0`。

### 特点

- **优点**：先压缩再归因，较适合中长轨迹
- **缺点**：如果第一阶段分段错误，第二阶段会被误导（误差级联）

---

## 3.3 `binary_search`（二分定位 step）

源码：`methods/baselines/binary_search.py`

### 主要思路

在 `[start, end]` 区间内反复询问模型：关键错误更可能在 **upper half** 还是 **lower half**，直到收敛到单个 step。

### 在样例上的处理（示意轮次）

1. 区间 `[0,4]`，mid=2，模型选 upper -> `[0,2]`
2. 区间 `[0,2]`，mid=1，模型选 upper -> `[0,1]`
3. 区间 `[0,1]`，mid=0，模型选 upper -> `[0,0]`
4. 收敛，输出 `step0` 对应 agent = `Planner_Agent`

### 特点

- **优点**：步定位效率高（对话很长时轮数约 `O(log n)`）
- **缺点**：每轮只做“上/下半段”粗判断，若模型在中间某轮不稳定，会触发保守 fallback

---

## 3.4 `step_by_step`（在线逐步审查）

源码：`methods/baselines/step_by_step.py`

### 主要思路

从 step0 开始逐步扩展上下文，每一轮只问“当前最新步是否存在会导致偏航的错误（Yes/No）”，一旦 Yes 立即停止并输出当前步。

### 在样例上的处理

- step0 判断：Yes（口径已偏）
- 立即停机，预测 `Planner_Agent@step0`

### 特点

- **优点**：天然偏向“最早责任步”，符合根因定位直觉；过程日志可解释性高
- **缺点**：最坏需要线性轮数 `O(n)`；对 Yes/No 格式依赖较强

---

## 4. 非 Baseline 方法（用于全景理解）

## 4.1 `chief`（结构化多阶段图推理）

源码：`methods/chief/method.py` + `methods/chief/parsers.py` + `methods/chief/prompts.py`

### 主要思路

6 阶段链路：

1. 子任务分解（可选 RAG 辅助）
2. 子任务间因果边
3. 子任务内 agent/action/data_flow 抽取
4. 子任务内 agent 因果边
5. 候选责任集合筛选（subtask/agent/step）
6. 在候选集上输出唯一责任归因

### 在样例上的处理

会把 step0 识别为“错误口径源头”，并通过边关系解释错误如何传导到最终答案 108。

### 特点

- **优点**：结构化中间产物丰富（图、边、候选集），最利于分析与诊断
- **缺点**：调用链最长、成本最高；解析格式对提示词稳定性较敏感

---

## 4.2 `echo`（多分析员共识）

源码：`methods/echo/method.py` + `context.py` + `voting.py`

### 主要思路

1. 先构建分层上下文摘要（immediate/nearby/distant/milestones）
2. 多个“客观分析员角色”并行独立给出 JSON 归因
3. 聚合投票（按置信度阈值）得到共识结论

### 在样例上的处理

多数分析员会给出 `Planner_Agent@step0`，最终共识输出该结果；若分歧高，会在 `disagreement_analysis` 中体现。

### 特点

- **优点**：通过多视角投票抑制单次随机性
- **缺点**：调用次数随分析员数量线性增加，成本和延迟更高

---

## 5. 方法对比总结（面向落地选择）

| 方法 | 核心机制 | 典型调用形态 | 可解释性 | 成本 |
|---|---|---|---|---|
| baseline | 全轨迹一次归因 | 1 次主调用 | 低-中 | 低 |
| all_at_once | 子任务分解 + 二阶段归因 | 2 次主调用 | 中 | 中 |
| binary_search | 区间二分找 step | 多轮（log n） | 中（有轮次日志） | 中 |
| step_by_step | 逐步 Yes/No 审查 | 多轮（最坏 n） | 高（逐步日志） | 中-高 |
| chief | DAG 化多阶段推理 | 6 阶段 | 很高 | 高 |
| echo | 多分析员 + 共识投票 | k 次分析 + 聚合 | 高 | 高 |

---

## 6. 关键实现细节与注意点

1. **step 编号是 0-based**（如 `Step 0`），报告和评估时要保持一致。  
2. `BaselineMethodBase.parse_final_prediction` 对输出格式有硬约束，字段名必须匹配 `Agent Name / Step Number / Reason for Mistake`。  
3. `core/runner.py` 会统一统计 `step_accuracy_with_tolerance`（默认 `±0..±5`）。  
4. `chief` 的 RAG 是可选能力（资源缺失自动退化为空检索，不阻塞流程）。  
5. `echo` 的鲁棒性较依赖 `num_analysts`、`min_confidence_threshold` 与上下文摘要质量。

---

## 7. 结论

如果目标是**快速建立可用基线**，优先 `baseline / all_at_once`；  
如果目标是**更稳的 earliest-step 定位**，优先 `step_by_step / binary_search`；  
如果目标是**高可解释、可分析的研究型归因**，优先 `chief / echo`。

以上结论是对当前源码实现机制的归纳，不等同于具体数据集上的绝对性能排名（性能仍需以同配置实验结果为准）。
