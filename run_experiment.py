"""FA4MAS 统一实验入口。"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_experiment_config
from core.runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FA4MAS method experiments.")
    parser.add_argument("--config", type=Path, required=True, help="JSON config file for the experiment.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run directory instead of starting a new timestamp.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run timestamp to resume/use, e.g. 20260406_235712.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    runner = ExperimentRunner(config, resume=bool(args.resume), run_id=args.run_id)
    summary = runner.run()

    print("experiment completed!")
    print(f"Method: {summary['method']} @ {summary['model']}")
    print(f"Dataset: {summary['dataset']} ({summary['total_samples']} samples)")
    print(f"Run ID: {summary.get('run_timestamp', '-')}")
    print(f"Resumed: {summary.get('resumed', False)}")
    print(
        "Progress: "
        f"existing={summary.get('existing_samples_before_run', 0)}, "
        f"newly_processed={summary.get('newly_processed_samples', 0)}, "
        f"completed={summary.get('completed_samples', 0)}, "
        f"missing={summary.get('missing_sample_count', 0)}"
    )
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
