"""FA4MAS 统一实验入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_experiment_config
from core.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FA4MAS method experiments.")
    parser.add_argument("--config", type=Path, required=True, help="JSON config file for the experiment.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    runner = ExperimentRunner(config)
    summary = runner.run()

    print("experiment completed!")
    print(f"Method: {summary['method']} @ {summary['model']}")
    print(f"Dataset: {summary['dataset']} ({summary['total_samples']} samples)")
    print(f"Agent Accuracy: {summary['agent_accuracy'] * 100:.2f}%")
    print(f"Step Accuracy: {summary['step_accuracy'] * 100:.2f}%")
    tolerance_stats = summary.get("step_accuracy_with_tolerance", {})
    if isinstance(tolerance_stats, dict) and tolerance_stats:
        for key, value in tolerance_stats.items():
            print(f"Step Accuracy {key}: {float(value) * 100:.2f}%")
    print(f"Samples: {summary['per_sample_path']}")
    print(f"Summary: {summary['summary_path']}")
    print(f"Badcases: {summary['badcase_dir']}")


if __name__ == "__main__":
    main()
