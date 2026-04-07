"""
ToT-style Failure Attribution — Who&When 本地数据集版
=====================================================
输出格式与 baseline (baseline_reproduction.py) 完全对齐：
  - 推理阶段逐条写入 .jsonl，字段：task_id / responsible_agent / error_step / reason
  - evaluate() 与 baseline 完全相同：mistake_step 0-indexed 转 1-indexed
  - 汇总 JSON 结构与 baseline summary 一致（method / model / metrics_tol0 / metrics_tol1）
  - CLI 参数名与 baseline 完全对齐

克隆数据:
    git clone https://github.com/ag2ai/Agents_Failure_Attribution.git

依赖:
    pip install openai

环境变量:
    export DEEPSEEK_API_KEY=sk-...

使用示例:
    python tot_failure_attribution.py \\
        --data_dir "./Agents_Failure_Attribution/Who&When/Algorithm-Generated" \\
        --model deepseek-ai/DeepSeek-V3 \\
        --max_samples 5

    python tot_failure_attribution.py \\
        --data_dir "./Agents_Failure_Attribution/Who&When/Algorithm-Generated" \\
        --model deepseek-ai/DeepSeek-V3 \\
        --use_ground_truth
"""
import os, sys, json, re, time, argparse
from pathlib import Path
from openai import OpenAI
from data.dataset import load_dataset
from core.evaluator import evaluate
from core.llm_client import LLMClient
from methods.famas import FamasMethod
from methods.tot import ToTMethod
from methods.baseline import AllAtOnceMethod, StepByStepMethod, BinarySearchMethod
from methods.chief import CHIEFMethod
from methods.echo import ECHOMethod


def run(args):
    MODEL           = args.model
    method          = args.method
    api_key  = args.api_key
    base_url = args.base_url
    if "deepseek" in args.model.lower() and not base_url:
        base_url = "https://api.deepseek.com"
    _client = OpenAI(api_key=api_key, base_url=base_url or None)
    # 1. 初始化模型客户端
    llm = LLMClient(model=args.model, api_key=args.api_key, base_url=args.base_url, temperature=args.temperature)

    # 2. 核心路由：根据参数选择实例化的算法类
    if args.method == "tot":
        method_runner = ToTMethod(llm, top_k=args.top_k, vote_rounds=args.vote_rounds, threshold=args.threshold)
    elif args.method == "all_at_once":
        method_runner = AllAtOnceMethod(llm)
    elif args.method == "step_by_step":
        method_runner = StepByStepMethod(llm)
    elif args.method == "binary_search":
        method_runner = BinarySearchMethod(llm)
    elif args.method == "chief":
        method_runner = CHIEFMethod(llm, save_graph=args.save_graph, output_dir=args.output_dir)
    elif args.method == "echo":
        method_runner = ECHOMethod(llm, k_analysts=args.k_analysts)
    elif args.method == "famas":
        method_runner = FamasMethod(llm, lam=args.lam, chunk_size=args.chunk_size)
    else:
        raise ValueError(f"不支持的方法: {args.method}")
    print(f"[模型]  {MODEL}")
    print(f"[API]   {base_url or 'default'}")

    print(f"[数据]  加载 {args.data_dir} ...")
    instances = load_dataset(args.data_dir)
    if args.max_samples > 0:
        instances = instances[:args.max_samples]
    print(f"[数据]  共 {len(instances)} 条样本")

    if instances:
        first = instances[0]
        print(f"[数据]  第一条字段: {list(first.keys())}")
        for key in ["question", "query", "task", "mistake_agent", "mistake_step"]:
            if key in first:
                val = str(first[key])
                print(f"  {key}: {val[:80]}{'...' if len(val)>80 else ''}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = Path(args.data_dir).name
    out_file = output_dir / f"tot_{MODEL.replace('/', '_')}_{dataset_name}.jsonl"

    print(f"\n[推理]  方法={method}  模型={MODEL}  use_gt={args.use_ground_truth}")

    predictions = []
    for i, inst in enumerate(instances):
        task_id = inst.get("task_id", str(i))
        try:
            pred = method_runner.run_attribution(inst, args.use_ground_truth, verbose=True)
        except Exception as e:
            print(f"  [错误] task {task_id}: {e}")
            pred = {"responsible_agent": "", "error_step": -1, "reason": str(e)}

        pred["task_id"] = task_id
        predictions.append(pred)

        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(instances)}")

    print("\n[评估结果]")
    for tol in [0, 1, 2]:
        metrics = evaluate(predictions, instances, tolerance=tol)
        tag = f"(tolerance={tol})" if tol > 0 else ""
        print(f"  Agent Acc {tag}: {metrics['agent_accuracy']:.1%}  "
              f"({metrics['agent_correct']}/{metrics['total']})")
        print(f"  Step  Acc {tag}: {metrics['step_accuracy']:.1%}  "
              f"({metrics['step_correct']}/{metrics['total']})")

    summary = {
        "method":           method,
        "model":            MODEL,
        "use_ground_truth": args.use_ground_truth,
        "n_samples":        len(instances),
        "metrics_tol0":     evaluate(predictions, instances, 0),
        "metrics_tol1":     evaluate(predictions, instances, 1),
    }
    summary_file = output_dir / f"tot_{MODEL.replace('/', '_')}_{dataset_name}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[输出]  详细结果: {out_file}")
    print(f"[输出]  汇总结果: {summary_file}")
    return summary


# ════════════════════════════════════════════════════════════════════════════════
# 6. CLI（参数名与 baseline 完全对齐）
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Failure Attribution Benchmark")

    # 1. 基础配置参数
    parser.add_argument("--data_dir",default="./data/repo/Who&When/Algorithm-Generated", help="数据集路径")
    parser.add_argument("--output_dir",       default="outputs")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3", help="使用的LLM模型")
    parser.add_argument("--api_key", default="sk-xgtseopileqvesiolifchlgdqdccntxcieilpkhwxbdxsgag", help="API Key (如果不填则读取环境变量)")
    parser.add_argument("--base_url", default="https://api.siliconflow.cn/v1", help="API Base URL")
    parser.add_argument("--method", default="chief", choices=["tot", "all_at_once", "step_by_step", "binary_search", "chief"])
    parser.add_argument("--max_samples", type=int, default=-1, metavar="N")
    parser.add_argument("--use_ground_truth", action="store_true")

    # 2. 算法特有的超参数 (Hyperparameters)
    parser.add_argument("--top_k", type=int, default=3, help="[ToT] 验证阶段保留的最大嫌疑步骤数 (默认: 3)")
    parser.add_argument("--vote_rounds", type=int, default=3, help="[ToT] 最终投票的轮数 (默认: 3)")
    parser.add_argument("--threshold", type=int, default=5, help="[ToT] 进入验证阶段的最低嫌疑分数阈值 (默认: 5)")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM 采样温度 (默认: 0.0, 保证确定性)")

    # 3.CHIEF 的特有参数
    parser.add_argument("--save_graph", action="store_true", help="[CHIEF] 保存每条样本的因果图 JSON")

    #4.echo特有的参数
    parser.add_argument("--k_analysts", type=int, default=3, help="[ECHO] 每次随机采样的分析师数量 (默认: 3)")

    #5.famas的参数
    parser.add_argument("--lam", type=float, default=0.9, help="[Famas] λ decay factor (默认 0.9)")
    parser.add_argument("--chunk_size", type=int, default=10, help="[Famas] 分块大小 (默认 10)")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()