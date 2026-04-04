"""FA4MAS 统一实验入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_experiment_config
from core.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 FA4MAS 方法实验。")
    parser.add_argument("--config", type=Path, required=True, help="JSON 配置文件路径。")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    runner = ExperimentRunner(config)
    summary = runner.run()

    print("实验完成")
    print(f"Method: {summary['method']} @ {summary['model']}")
    print(f"Dataset: {summary['dataset']} ({summary['total_samples']} samples)")
    print(f"Agent Accuracy: {summary['agent_accuracy'] * 100:.2f}%")
    print(f"Step Accuracy: {summary['step_accuracy'] * 100:.2f}%")
    print(f"Samples: {summary['per_sample_path']}")
    print(f"Summary: {summary['summary_path']}")
    print(f"Badcases: {summary['badcase_dir']}")


if __name__ == "__main__":
    main()

