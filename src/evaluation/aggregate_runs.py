#!/usr/bin/env python3
"""
Aggregate Results Across Multiple Experiment Runs

Computes mean and standard deviation for all metrics across multiple runs,
enabling reliable confidence interval estimation for bar charts.

Usage:
    python -m src.evaluation.aggregate_runs \
        --run_dirs dir1 dir2 dir3 \
        --output_dir aggregated_results
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
import math


def load_summary(summary_path: Path) -> Dict[str, Any]:
    """Load a summary JSON file."""
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)


def mean(values: List[float]) -> Optional[float]:
    """Compute mean, handling empty lists."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def std(values: List[float]) -> Optional[float]:
    """Compute population standard deviation."""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    variance = sum((x - m) ** 2 for x in valid) / len(valid)
    return math.sqrt(variance)


def stderr(values: List[float]) -> Optional[float]:
    """Compute standard error of the mean."""
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    s = std(valid)
    if s is None:
        return None
    return s / math.sqrt(len(valid))


def aggregate_metric(values: List[float]) -> Dict[str, Optional[float]]:
    """Aggregate a list of metric values into mean, std, stderr, min, max."""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": None, "std": None, "stderr": None, "min": None, "max": None, "n": 0}
    return {
        "mean": mean(valid),
        "std": std(valid),
        "stderr": stderr(valid),
        "min": min(valid),
        "max": max(valid),
        "n": len(valid),
    }


def aggregate_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple summary dictionaries into one with mean/std.

    Input: List of summary dicts from multiple runs
    Output: Single summary dict with mean/std for each metric
    """
    if not summaries:
        return {}

    # Collect all dataset keys
    all_keys = set()
    for s in summaries:
        all_keys.update(s.keys())

    aggregated = {}

    for key in all_keys:
        if key == "_overall":
            continue  # Handle separately

        # Collect values for each metric across runs
        plan_metrics = defaultdict(list)
        tool_metrics = defaultdict(list)
        answer_metrics = defaultdict(list)
        meta_metrics = defaultdict(list)

        for s in summaries:
            data = s.get(key, {})
            if not data:
                continue

            # Plan metrics
            plan = data.get("plan", {})
            for m in [
                "node_f1",
                "span_node_f1",
                "strict_node_f1",
                "edge_f1",
                "semantic_edge_f1",
                "dw_order_f1",
                "planning_score",
                "order_precision",
                "order_recall",
                "order_f1",
                "node_label_similarity",
                "ssi",
            ]:
                if plan.get(m) is not None:
                    plan_metrics[m].append(plan[m])

            # Tool metrics
            tool = data.get("tool", {})
            for m in [
                "tool_name_f1",
                "param_name_f1",
                "type_aware_value_f1",
                "tool_usage_score",
                "gt_aligned_tool_success_rate",
                "pred_exec_success_rate",
                "observation_support_rate",
            ]:
                if tool.get(m) is not None:
                    tool_metrics[m].append(tool[m])

            # Answer metrics
            answer = data.get("answer", {}) or {}
            for m in ["exact_match", "token_f1"]:
                if answer.get(m) is not None:
                    answer_metrics[m].append(answer[m])

            # Meta metrics
            if data.get("num_samples") is not None:
                meta_metrics["num_samples"].append(data["num_samples"])
            if data.get("parse_error_rate") is not None:
                meta_metrics["parse_error_rate"].append(data["parse_error_rate"])

        # Build aggregated entry
        aggregated[key] = {
            "num_runs": len(summaries),
            "num_samples": aggregate_metric(meta_metrics.get("num_samples", [])),
            "parse_error_rate": aggregate_metric(meta_metrics.get("parse_error_rate", [])),
            "plan": {
                m: aggregate_metric(plan_metrics.get(m, []))
                for m in [
                    "node_f1",
                    "span_node_f1",
                    "strict_node_f1",
                    "edge_f1",
                    "semantic_edge_f1",
                    "dw_order_f1",
                    "planning_score",
                    "order_precision",
                    "order_recall",
                    "order_f1",
                    "node_label_similarity",
                    "ssi",
                ]
            },
            "tool": {
                m: aggregate_metric(tool_metrics.get(m, []))
                for m in [
                    "tool_name_f1",
                    "param_name_f1",
                    "type_aware_value_f1",
                    "tool_usage_score",
                    "gt_aligned_tool_success_rate",
                    "pred_exec_success_rate",
                    "observation_support_rate",
                ]
            },
            "answer": {
                m: aggregate_metric(answer_metrics.get(m, []))
                for m in ["exact_match", "token_f1"]
            } if answer_metrics else None,
        }

    return aggregated


def find_summary_files(run_dir: Path) -> List[Path]:
    """Find all summary JSON files in a run directory."""
    summaries = []
    # Check direct children
    for f in run_dir.glob("summary.*.json"):
        summaries.append(f)
    # Check model subdirectories (answer mode structure)
    for model_dir in run_dir.iterdir():
        if model_dir.is_dir():
            for f in model_dir.glob("summary.*.json"):
                summaries.append(f)
    return summaries


def aggregate_runs(run_dirs: List[Path], output_dir: Path) -> None:
    """
    Aggregate results from multiple run directories.

    Args:
        run_dirs: List of directories containing run results
        output_dir: Directory to write aggregated results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group summaries by model
    model_summaries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[WARN] Run directory not found: {run_dir}")
            continue

        summary_files = find_summary_files(run_dir)

        for sf in summary_files:
            # Extract model name from filename (summary.MODEL_NAME.json)
            model_name = sf.stem.replace("summary.", "")
            summary = load_summary(sf)
            if summary:
                model_summaries[model_name].append(summary)

    if not model_summaries:
        print("[ERROR] No summary files found in any run directory")
        return

    # Aggregate each model's results
    for model_name, summaries in model_summaries.items():
        print(f"[INFO] Aggregating {len(summaries)} runs for {model_name}")

        aggregated = aggregate_summaries(summaries)

        # Save aggregated summary
        output_path = output_dir / f"aggregated.{model_name}.json"
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved: {output_path}")

    # Create combined summary with all models
    combined = {
        "metadata": {
            "num_runs": len(run_dirs),
            "run_dirs": [str(d) for d in run_dirs],
            "models": list(model_summaries.keys()),
        },
        "models": {}
    }

    for model_name in model_summaries.keys():
        agg_path = output_dir / f"aggregated.{model_name}.json"
        if agg_path.exists():
            with open(agg_path) as f:
                combined["models"][model_name] = json.load(f)

    combined_path = output_dir / "aggregated_all_models.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved combined: {combined_path}")

    # Generate summary table
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 70)

    for model_name, summaries in model_summaries.items():
        aggregated = aggregate_summaries(summaries)
        print(f"\n### {model_name} ({len(summaries)} runs) ###")

        for dataset_key, data in aggregated.items():
            if dataset_key == "_overall":
                continue

            plan = data.get("plan", {})
            tool = data.get("tool", {})
            answer = data.get("answer", {})

            node_f1 = plan.get("node_f1", {})
            tool_f1 = tool.get("tool_name_f1", {})
            em = answer.get("exact_match", {}) if answer else {}

            print(f"  {dataset_key}:")
            if node_f1.get("mean") is not None:
                print(f"    Span-Node F1: {node_f1['mean']:.4f} ± {node_f1.get('std', 0) or 0:.4f}")
            if tool_f1.get("mean") is not None:
                print(f"    Tool F1: {tool_f1['mean']:.4f} ± {tool_f1.get('std', 0) or 0:.4f}")
            if em.get("mean") is not None:
                print(f"    Exact Match: {em['mean']:.4f} ± {em.get('std', 0) or 0:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results across multiple experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="List of run directories to aggregate",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for aggregated results",
    )

    args = parser.parse_args()

    aggregate_runs(args.run_dirs, args.output_dir)


if __name__ == "__main__":
    main()
