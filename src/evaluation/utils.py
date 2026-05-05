#!/usr/bin/env python3
"""
Evaluation Utilities

Helper functions for evaluation metrics, formatting, and I/O.
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional
from dataclasses import asdict, fields


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Load records from a JSONL file."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def mean(xs: List[Optional[float]]) -> Optional[float]:
    """Compute mean, handling None values."""
    valid = [x for x in xs if x is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def std(xs: List[Optional[float]]) -> Optional[float]:
    """Compute standard deviation, handling None values."""
    valid = [x for x in xs if x is not None]
    if len(valid) < 2:
        return None
    m = sum(valid) / len(valid)
    return (sum((x - m) ** 2 for x in valid) / len(valid)) ** 0.5


def format_metric(value: Optional[float], decimals: int = 4) -> str:
    """Format a metric value, handling None as N/A."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def write_csv(results: List[Any], path: Path, result_class):
    """
    Write per-sample results to CSV with proper None handling.

    Args:
        results: List of dataclass instances
        path: Output CSV path
        result_class: The dataclass type (used to get field names)
    """
    if not results:
        return

    fieldnames = [f.name for f in fields(result_class)]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            # Convert None to "N/A" for CSV clarity
            for key, value in row.items():
                if value is None:
                    row[key] = "N/A"
            writer.writerow(row)


def print_summary(summary: Dict[str, Any]):
    """Print formatted summary to console."""
    print("=" * 70)
    print("EVALUATION SUMMARY (Dataset-Aware)")
    print("=" * 70)

    for key in sorted(summary.keys()):
        data = summary[key]
        print(f"\n>>> {key}")
        print(f"    Samples: {data['num_samples']}")
        print(f"    Parse Errors: {data['parse_errors']} ({data['parse_error_rate']:.1%})")

        if "samples_with_arguments" in data:
            print(f"    With Arguments: {data['samples_with_arguments']}")
            print(f"    Without Arguments: {data['samples_without_arguments']}")
        elif "has_argument_samples" in data:
            print(f"    Has Argument Samples: {data['has_argument_samples']}")

        print(f"    Plan Metrics:")
        if "planning_score" in data["plan"]:
            print(f"      Planning Score:{format_metric(data['plan']['planning_score'])}")
        print(f"      Span-Node F1: {format_metric(data['plan']['node_f1'])}")
        if "strict_node_f1" in data["plan"]:
            print(f"      Strict Node F1:{format_metric(data['plan']['strict_node_f1'])}")
        print(f"      Edge F1:       {format_metric(data['plan']['edge_f1'])}")
        if "raw_edge_f1" in data["plan"]:
            print(f"      Raw Edge F1:   {format_metric(data['plan']['raw_edge_f1'])}")
        if "dw_order_f1" in data["plan"]:
            print(f"      DW-Order F1:  {format_metric(data['plan']['dw_order_f1'])}")
        if "order_precision" in data["plan"]:
            print(f"      Order Precision:{format_metric(data['plan']['order_precision'])}")
        if "order_recall" in data["plan"]:
            print(f"      Order Recall: {format_metric(data['plan']['order_recall'])}")
        print(f"      Node Label Sim:{format_metric(data['plan']['node_label_similarity'])}")
        print(f"      SSI:           {format_metric(data['plan']['ssi'])}")

        print(f"    Tool Metrics:")
        print(f"      Tool F1:       {format_metric(data['tool']['tool_name_f1'])}")
        print(f"      Param F1 (t):  {format_metric(data['tool']['param_name_f1'])}")
        if "type_aware_value_f1" in data["tool"]:
            print(f"      Value F1:      {format_metric(data['tool']['type_aware_value_f1'])}")
        if "strict_type_aware_value_f1" in data["tool"]:
            print(f"      Strict Value:  {format_metric(data['tool']['strict_type_aware_value_f1'])}")
        if "normalized_type_aware_value_f1" in data["tool"]:
            print(f"      Norm. Value:   {format_metric(data['tool']['normalized_type_aware_value_f1'])}")
        if "tool_usage_score" in data["tool"]:
            print(f"      Tool Usage:    {format_metric(data['tool']['tool_usage_score'])}")
        if "gt_aligned_tool_success_rate" in data["tool"]:
            print(f"      GT-Tool Succ:  {format_metric(data['tool']['gt_aligned_tool_success_rate'])}")
        if "pred_exec_success_rate" in data["tool"]:
            print(f"      Exec Success:  {format_metric(data['tool']['pred_exec_success_rate'])}")
        if "observation_support_rate" in data["tool"]:
            print(f"      OSR:           {format_metric(data['tool']['observation_support_rate'])}")

        if data.get("answer"):
            ans = data["answer"]
            print(f"    Answer Metrics ({ans['num_with_answer']} samples):")
            print(f"      Exact Match:   {format_metric(ans.get('exact_match'))}")
            print(f"      Token F1:      {format_metric(ans.get('token_f1'))}")
            if "alias_match" in ans:
                print(f"      Alias Match:   {format_metric(ans.get('alias_match'))}")
            if "llm_judge_score" in ans:
                print(f"      LLM Judge:     {format_metric(ans.get('llm_judge_score'))}")

    print("\n" + "=" * 70)
