#!/usr/bin/env python3
# generate_facet_scores.py
"""
Unified LLM Planning Evaluation Visualization

This script generates all visualization figures for the LLM Planning evaluation.
It consolidates functionality from the previous generate_facet_scores.py and generate_figures.py.

Facet A: Planning Fidelity = avg(SpanNodeF1, DW-Order F1)
Facet B: Tool Usage Accuracy
    - If has_arguments=True: avg(Tool F1, Param F1, Value F1)
    - If has_arguments=False (Delta): Tool F1 only
Facet C: Answer Accuracy = avg(Exact Match, Token F1) [only for GAIA]

Total Score:
- UltraTool/TaskBench: (Facet A + Facet B) / 2
- Delta (no args): (Facet A + Tool F1) / 2
- GAIA: Facet C only

Figures Generated:
1. Per-dataset facet score bar charts
2. Detailed metrics bar charts
3. Radar charts (with SSI and without SSI variants)
4. Model scaling analysis (line chart)
5. Parse error impact analysis
6. Overall comparison bar chart

Usage:
    python generate_facet_scores.py --results_dir ./results
    python generate_facet_scores.py --results_dir ./delta_new_results --output_dir ./figures
"""

import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import pi

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# ============================================================================
# Model Metadata
# ============================================================================

MODEL_INFO = {
    # Qwen2.5 Family (hyphen format)
    "qwen2.5-0.5b": {"family": "Qwen", "size_b": 0.5, "display": "Qwen2.5\n0.5B", "short": "0.5B", "order": 1},
    "qwen2.5-1.5b": {"family": "Qwen", "size_b": 1.5, "display": "Qwen2.5\n1.5B", "short": "1.5B", "order": 2},
    "qwen2.5-3b": {"family": "Qwen", "size_b": 3, "display": "Qwen2.5\n3B", "short": "3B", "order": 2.5},
    "qwen2.5-7b": {"family": "Qwen", "size_b": 7, "display": "Qwen2.5\n7B", "short": "7B", "order": 3},
    "qwen2.5-14b": {"family": "Qwen", "size_b": 14, "display": "Qwen2.5\n14B", "short": "14B", "order": 4},
    "qwen2.5-32b": {"family": "Qwen", "size_b": 32, "display": "Qwen2.5\n32B", "short": "32B", "order": 5},
    "qwen2.5-72b": {"family": "Qwen", "size_b": 72, "display": "Qwen2.5\n72B", "short": "72B", "order": 6},
    # Qwen2.5 Family (colon format - API naming convention)
    "qwen2.5:0.5b": {"family": "Qwen", "size_b": 0.5, "display": "Qwen2.5\n0.5B", "short": "0.5B", "order": 1},
    "qwen2.5:1.5b": {"family": "Qwen", "size_b": 1.5, "display": "Qwen2.5\n1.5B", "short": "1.5B", "order": 2},
    "qwen2.5:3b": {"family": "Qwen", "size_b": 3, "display": "Qwen2.5\n3B", "short": "3B", "order": 2.5},
    "qwen2.5:7b": {"family": "Qwen", "size_b": 7, "display": "Qwen2.5\n7B", "short": "7B", "order": 3},
    "qwen2.5:14b": {"family": "Qwen", "size_b": 14, "display": "Qwen2.5\n14B", "short": "14B", "order": 4},
    "qwen2.5:32b": {"family": "Qwen", "size_b": 32, "display": "Qwen2.5\n32B", "short": "32B", "order": 5},
    "qwen2.5:72b": {"family": "Qwen", "size_b": 72, "display": "Qwen2.5\n72B", "short": "72B", "order": 6},
    "Toolchestra": {"family": "Toolchestra", "size_b": 0, "display": "Toolchestra", "short": "Toolchestra", "order": 0},
    # Qwen3 Family (API models)
    "qwen3-4b": {"family": "Qwen", "size_b": 4, "display": "Qwen3\n4B", "short": "3-4B", "order": 7},
    "qwen3-4b-thinking": {"family": "Qwen", "size_b": 4, "display": "Qwen3-Think\n4B", "short": "3-4B-T", "order": 8},
    "qwen2.5-7b-api": {"family": "Qwen", "size_b": 7, "display": "Qwen2.5\n7B-API", "short": "7B-API", "order": 9},
    # Mistral Family
    "mistral-7b": {"family": "Mistral", "size_b": 7, "display": "Mistral\n7B", "short": "7B", "order": 20},
    # Vicuna Family
    "vicuna-7b": {"family": "Vicuna", "size_b": 7, "display": "Vicuna\n7B", "short": "7B", "order": 30},
    "vicuna-13b": {"family": "Vicuna", "size_b": 13, "display": "Vicuna\n13B", "short": "13B", "order": 31},
    # LLaMA Family
    "llama3.1-8b": {"family": "LLaMA", "size_b": 8, "display": "LLaMA3.1\n8B", "short": "8B", "order": 40},
    "llama3-8b": {"family": "LLaMA", "size_b": 8, "display": "LLaMA3\n8B", "short": "8B", "order": 41},
    # Phi Family
    "phi4": {"family": "Phi", "size_b": 14, "display": "Phi4", "short": "4", "order": 50},
    "phi4-mini": {"family": "Phi", "size_b": 3.8, "display": "Phi4\nMini", "short": "4-Mini", "order": 51},
    "phi3-mini": {"family": "Phi", "size_b": 3.8, "display": "Phi3\nMini", "short": "3-Mini", "order": 52},
    # DeepSeek Family
    "deepseek-r1-8b": {"family": "DeepSeek", "size_b": 8, "display": "DeepSeek\nR1-8B", "short": "R1-8B", "order": 60},
    # Gemma Family
    "gemma2-9b": {"family": "Gemma", "size_b": 9, "display": "Gemma2\n9B", "short": "2-9B", "order": 69},
    "gemma3-27b": {"family": "Gemma", "size_b": 27, "display": "Gemma3\n27B", "short": "27B", "order": 70},
    "gemma3-vl-27b": {"family": "Gemma", "size_b": 27, "display": "Gemma3-VL\n27B", "short": "VL-27B", "order": 71},
    # GPT-OSS
    "gpt-oss-20b": {"family": "GPT-OSS", "size_b": 20, "display": "GPT-OSS\n20B", "short": "20B", "order": 80},
    "gpt-oss-120b": {"family": "GPT-OSS", "size_b": 120, "display": "GPT-OSS\n120B", "short": "120B", "order": 81},
    "gptoss": {"family": "GPT-OSS", "size_b": 20, "display": "GPT-OSS\n20B", "short": "20B", "order": 80},
    # New Models (2026-02-05)
    "Google-Gemma-3-27B": {"family": "Gemma", "size_b": 27, "display": "Gemma3\n27B", "short": "3-27B", "order": 72},
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {"family": "LLaMA", "size_b": 17, "display": "Llama4\nMaverick 17B", "short": "4-17B", "order": 44},
    "Llama-3.1-405B-Instruct-FP8": {"family": "LLaMA", "size_b": 405, "display": "Llama3.1\n405B", "short": "405B", "order": 45},
    "Llama-3.3-70B-Instruct": {"family": "LLaMA", "size_b": 70, "display": "Llama3.3\n70B", "short": "3.3-70B", "order": 46},
    "Mistral-Small-3.2-24B-Instruct-2506": {"family": "Mistral", "size_b": 24, "display": "Mistral\nSmall 3.2", "short": "Small-24B", "order": 25},
}

FAMILY_COLORS = {
    "Qwen": "#1E88E5",      # Blue
    "Mistral": "#D32F2F",   # Dark Red
    "Vicuna": "#607D8B",    # Blue Grey
    "LLaMA": "#AB47BC",     # Magenta/Purple
    "Phi": "#43A047",       # Green
    "DeepSeek": "#EAF2EB",  # Pink
    "Gemma": "#FF9800",     # Amber/Gold
    "GPT-OSS": "#FFD700",   # Yellow/Gold
    "Unknown": "#746B73",   # Gray fallback
}

# Model colors for radar chart - HIGH CONTRAST distinct colors
# Designed for clear visual distinction between all models
MODEL_COLORS = {
    # Qwen family - hyphen format (blue shades - distinct from each other)
    "qwen2.5-0.5b": "#E3F2FD",  # Very Light blue
    "qwen2.5-1.5b": "#BBDEFB",  # Light blue
    "qwen2.5-3b": "#64B5F6",    # Light-medium blue
    "qwen2.5-7b": "#2196F3",    # Blue
    "qwen2.5-14b": "#0D47A1",   # Dark blue
    "qwen2.5-32b": "#01579B",   # Very Dark blue
    # "qwen2.5-72b": "#002171",   # Darkest navy
    # Qwen family - colon format (API naming convention)
    "qwen2.5:0.5b": "#E3F2FD",  # Very Light blue
    "qwen2.5:1.5b": "#BBDEFB",  # Light blue
    "qwen2.5:3b": "#64B5F6",    # Light-medium blue
    "qwen2.5:7b": "#2196F3",    # Blue
    "qwen2.5:14b": "#0D47A1",   # Dark blue
    "qwen2.5:32b": "#01579B",   # Very Dark blue
    "qwen3-4b": "#2196F3",      # Standard blue
    "qwen3-4b-thinking": "#1976D2",  # Deep blue
    "qwen2.5-7b-api": "#0277BD",  # Cerulean
    
    # Toolchestra
    "Toolchestra": "#424242",   # Dark Grey

    # GPT-OSS family - gold shades
    "gptoss20b": "#FDD835",
    "gpt-oss-20b": "#FDD835",
    "gpt-oss-120b": "#FFB300",

    # Mistral - Bright Orange (distinct from all)
    "mistral-7b": "#D32F2F",    # Dark red
    # Vicuna (red shades)
    # "vicuna-7b": "#EF5350",     # Light red
    # "vicuna-13b": "#D32F2F",    # Dark red
    # LLaMA - Magenta/Purple (NOT close to blue)
    "llama3.1-8b": "#AB47BC",   # Magenta purple
    "llama3-8b": "#8E24AA",     # Deep purple
    # Phi - Green (clearly different)
    "phi4": "#2E7D32",          # Dark green
    "phi4-mini": "#43A047",     # Medium green
    "phi3-mini": "#66BB6A",     # Light green
    # DeepSeek - Pink
    "deepseek-r1-8b": "#E91E63",  # Pink
    # Gemma - Gold/Amber (warm, distinct from orange)
    "gemma2-9b": "#FFB300",     # Amber light
    "gemma3-27b": "#FF9800",    # Amber
    "gemma3-vl-27b": "#F57C00", # Dark orange
    # Vicuna - Blue Grey
    "vicuna-7b": "#607D8B",     # Blue Grey
    "vicuna-13b": "#455A64",    # Dark Blue Grey
    # GPT-OSS - Yellow/Gold
    "gptoss": "#FDD835",
    
    "Toolchestra": "#424242",   # Dark Grey
    
    # New Models (Distinct High-Contrast Colors)
    "Google-Gemma-3-27B": "#FF8F00",                # Amber 800 - Bright Orange/Gold
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "#B39DDB",  # Light LLaMA lavender
    "Llama-3.1-405B-Instruct-FP8": "#5E35B1",       # Deep LLaMA purple
    "Llama-3.3-70B-Instruct": "#7E57C2",            # Mid LLaMA purple
    "Mistral-Small-3.2-24B-Instruct-2506": "#D50000", # Red A700 - Bright Red
}

# Dataset configurations
DATASET_INFO = {
    # Delta
    "delta/mcp_tools": {"display": "Delta (MCP)", "short": "Delta", "has_answer": False, "color": "#8B5CF6"},
    
    # UltraTool
    "ultratool/test": {"display": "UltraTool", "short": "UltraTool", "has_answer": False, "color": "#E63946"},
    
    # TaskBench variants
    "taskbench/multimedia": {"display": "TaskBench-MM", "short": "TB-MM", "has_answer": False, "color": "#457B9D"},
    "taskbench/huggingface": {"display": "TaskBench-HF", "short": "TB-HF", "has_answer": False, "color": "#1D3557"},
    "taskbench/dailylifeapis": {"display": "TaskBench-DL", "short": "TB-DL", "has_answer": False, "color": "#2A9D8F"},
    
    # GAIA
    "gaia/val_2023": {"display": "GAIA", "short": "GAIA", "has_answer": True, "color": "#F4A261"},
}


def load_summaries(results_dir: Path):
    """Load all summary JSON files."""
    summaries = {}
    
    # Try different naming patterns
    patterns = ["summary.*.json", "summary_*.json"]
    
    for pattern in patterns:
        for f in results_dir.glob(pattern):
            # Extract model name
            stem = f.stem
            if stem.startswith("summary."):
                model_name = stem.replace("summary.", "")
            elif stem.startswith("summary_"):
                model_name = stem.replace("summary_", "")
            else:
                model_name = stem
            
            # Normalize model name (handle variations like qwen2_5-7b -> qwen2.5-7b)
            model_name = model_name.replace("_", ".")
            # Fix double dots
            model_name = model_name.replace("..", ".")
            # Handle special cases
            if "qwen2.5" in model_name:
                # Normalize to qwen2.5-Xb format
                model_name = model_name.replace("qwen2.5.", "qwen2.5-")
                model_name = model_name.replace("qwen2.5-.", "qwen2.5-")

            # Unify Llama-3.3-70B variants (Gaudi3, MI210) to single display name
            MODEL_NAME_MAPPING = {
                "Llama-3.3-70B-Instruct-Gaudi3": "Llama-3.3-70B-Instruct",
                "Llama-3.3-70B-Instruct-MI210": "Llama-3.3-70B-Instruct",
            }
            model_name = MODEL_NAME_MAPPING.get(model_name, model_name)

            try:
                with open(f) as fp:
                    summaries[model_name] = json.load(fp)
                print(f"  Loaded: {f.name} -> {model_name}")
            except Exception as e:
                print(f"  [WARN] Failed to load {f}: {e}")
    
    return summaries


def get_sorted_models(summaries):
    """Sort models by family first, then by order (groups Qwen models together)."""
    def sort_key(model_name):
        info = MODEL_INFO.get(model_name, {})
        family = info.get("family", "Unknown")
        order = info.get("order", 99)
        # Family order: Qwen=1, Mistral=2, Vicuna=3, LLaMA=4, Gemma=5, others=9
        family_order = {
            "Qwen": 1,
            "Mistral": 2,
            "Vicuna": 3,
            "LLaMA": 4,
            "GPT-OSS": 5,
            "Gemma": 6,
            "Phi": 7,
            "DeepSeek": 8,
        }.get(family, 9)
        return (family_order, order)
    return sorted(summaries.keys(), key=sort_key)


def get_available_datasets(summaries):
    """Get list of datasets that have data in the summaries."""
    available = set()
    for model_summary in summaries.values():
        for key in model_summary.keys():
            if key != "_overall":
                available.add(key)
    return sorted(available)


def compute_facet_scores(summary, dataset_key):
    """
    Compute facet scores for a single model on a single dataset.
    
    FIXED: Properly handle Delta dataset which has no arguments.
    """
    data = summary.get(dataset_key, {})
    if not data:
        return None
    
    plan = data.get("plan", {})
    tool = data.get("tool", {})
    answer = data.get("answer", {})
    
    # Check if dataset has arguments
    # FIXED: Derive from actual values - if param_f1 is None, no arguments
    has_arguments = tool.get("has_arguments")
    if has_arguments is None:
        # Infer from param_name_f1 - if it's None, this dataset has no arguments
        has_arguments = tool.get("param_name_f1") is not None
    
    # Facet A: Planning Fidelity
    node_f1 = plan.get("span_node_f1", plan.get("node_f1", 0)) or 0
    edge_f1 = plan.get("edge_f1", 0) or 0
    dw_order_f1 = plan.get("dw_order_f1")
    ssi = plan.get("ssi", 0) or 0
    facet_a = plan.get("planning_score")
    if facet_a is None:
        dw_order_f1 = dw_order_f1 if dw_order_f1 is not None else 0
        facet_a = (node_f1 + dw_order_f1) / 2
    
    # Facet B: Tool Usage Accuracy
    # FIXED: Handle has_arguments and None values properly
    tool_f1 = tool.get("tool_name_f1", 0) or 0
    param_f1 = tool.get("param_name_f1")  # Keep as None if not available
    value_f1 = tool.get("type_aware_value_f1")
    explicit_tool_usage = tool.get("tool_usage_score")

    # Determine which components are available and compute Facet B accordingly
    if explicit_tool_usage is not None:
        facet_b = explicit_tool_usage
        facet_b_components = 3 if has_arguments else 1
    elif has_arguments:
        # Count available components for proper averaging
        available_components = [tool_f1]
        if param_f1 is not None:
            available_components.append(param_f1)
        if value_f1 is not None:
            available_components.append(value_f1)

        facet_b_components = len(available_components)
        facet_b = sum(available_components) / facet_b_components if facet_b_components > 0 else 0.0
    else:
        # Tool selection only (Delta or datasets without arguments)
        facet_b = tool_f1
        facet_b_components = 1
    
    # Facet C: Answer Accuracy (only for GAIA)
    facet_c = None
    if answer:
        em = answer.get("exact_match")
        token_f1 = answer.get("token_f1")
        if em is not None and token_f1 is not None:
            facet_c = (em + token_f1) / 2
        elif em is not None:
            facet_c = em
    
    # Total Score
    dataset_info = DATASET_INFO.get(dataset_key, {"has_answer": False})
    if dataset_info.get("has_answer") and facet_c is not None:
        total = facet_c
    else:
        total = (facet_a + facet_b) / 2
    
    return {
        "facet_a": facet_a,
        "facet_b": facet_b,
        "facet_b_components": facet_b_components,
        "facet_c": facet_c,
        "total": total,
        "node_f1": node_f1,
        "edge_f1": edge_f1,
        "dw_order_f1": dw_order_f1,
        "node_label_similarity": plan.get("node_label_similarity", 0) or 0,
        "ssi": ssi,
        "tool_f1": tool_f1,
        "param_f1": param_f1 if param_f1 is not None else None,
        "value_f1": value_f1 if value_f1 is not None else None,
        "has_arguments": has_arguments,
        "exact_match": answer.get("exact_match") if answer else None,
        "token_f1": answer.get("token_f1") if answer else None,
        "llm_judge_score": answer.get("llm_judge_score") if answer else None,
        # Additional metrics
        "parse_error_rate": data.get("parse_error_rate", 0) or 0,
        "num_samples": data.get("num_samples", 0),
    }


# ============================================================================
# Metrics Tables
# ============================================================================

TABLE_COLUMNS = [
    "scope",
    "dataset",
    "model",
    "family",
    "num_samples",
    "planning_score",
    "node_f1",
    "edge_f1",
    "node_label_similarity",
    "ssi",
    "tool_usage_score",
    "tool_name_f1",
    "param_name_f1",
    "type_aware_value_f1",
    "parse_error_rate",
    "exact_match",
    "token_f1",
    "llm_judge_score",
]


def _format_table_value(value):
    """Format values for markdown tables while preserving readability."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_metrics_row(model_name, dataset_key, scores):
    """Convert computed scores into a flat table row."""
    info = MODEL_INFO.get(model_name, {"family": "Unknown"})
    dataset_info = DATASET_INFO.get(dataset_key, {"display": dataset_key})
    dataset_display = "OVERALL" if dataset_key == "_overall" else dataset_info.get("display", dataset_key)

    return {
        "scope": "overall" if dataset_key == "_overall" else "dataset",
        "dataset": dataset_display,
        "model": model_name,
        "family": info.get("family", "Unknown"),
        "num_samples": scores.get("num_samples"),
        "planning_score": scores.get("facet_a"),
        "node_f1": scores.get("node_f1"),
        "edge_f1": scores.get("edge_f1"),
        "node_label_similarity": scores.get("node_label_similarity"),
        "ssi": scores.get("ssi"),
        "tool_usage_score": scores.get("facet_b"),
        "tool_name_f1": scores.get("tool_f1"),
        "param_name_f1": scores.get("param_f1"),
        "type_aware_value_f1": scores.get("value_f1"),
        "parse_error_rate": scores.get("parse_error_rate"),
        "exact_match": scores.get("exact_match"),
        "token_f1": scores.get("token_f1"),
        "llm_judge_score": scores.get("llm_judge_score"),
    }


def build_metrics_table_rows(summaries, datasets):
    """Build flat rows for overall and per-dataset metrics export."""
    rows = []
    models = get_sorted_models(summaries)

    for model in models:
        overall_scores = compute_facet_scores(summaries[model], "_overall")
        if overall_scores is not None:
            rows.append(_build_metrics_row(model, "_overall", overall_scores))

        for dataset in datasets:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is not None:
                rows.append(_build_metrics_row(model, dataset, scores))

    return rows


def export_metrics_tables(rows, output_dir: Path):
    """Export metrics rows as CSV and Markdown tables."""
    if not rows:
        return

    overall_rows = [row for row in rows if row["scope"] == "overall"]
    dataset_rows = [row for row in rows if row["scope"] == "dataset"]

    for name, table_rows in [
        ("overall_metrics_table", overall_rows),
        ("per_dataset_metrics_table", dataset_rows),
    ]:
        csv_path = output_dir / f"{name}.csv"
        md_path = output_dir / f"{name}.md"

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(table_rows)

        with md_path.open("w") as f:
            f.write("| " + " | ".join(TABLE_COLUMNS) + " |\n")
            f.write("| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |\n")
            for row in table_rows:
                f.write("| " + " | ".join(_format_table_value(row[col]) for col in TABLE_COLUMNS) + " |\n")

        print(f"[OK] Saved: {csv_path.name}, {md_path.name}")


# ============================================================================
# Figure: Radar Charts (With SSI and Without SSI variants)
# ============================================================================

def plot_radar_with_ssi(summaries, output_dir, datasets):
    """
    Generate radar chart WITH SSI metric.

    For Delta (no arguments): Show Tool F1, Edge F1, Node F1, SSI, Node Label Sim (5 metrics)
    For datasets with arguments: Show full metrics including Param F1, Value F1
    """
    models = get_sorted_models(summaries)

    # Filter to planning datasets only
    planning_datasets = [ds for ds in datasets
                        if not DATASET_INFO.get(ds, {}).get("has_answer", False)]

    valid_datasets = [ds for ds in planning_datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]

    if not valid_datasets:
        print("[WARN] No valid datasets for radar chart")
        return

    for dataset in valid_datasets:
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset, "short": dataset.replace("/", "_")})
        dataset_display = dataset_info.get("display", dataset)
        dataset_short = dataset_info.get("short", dataset.replace("/", "_"))

        # Determine which metrics to show based on has_arguments
        first_model_scores = None
        for m in models:
            first_model_scores = compute_facet_scores(summaries[m], dataset)
            if first_model_scores:
                break

        if not first_model_scores:
            continue

        has_arguments = first_model_scores.get("has_arguments", False)


        # -----------------------------
        # Build metric defs (candidate)
        # -----------------------------
        metric_defs = [
            ("tool_f1", "Tool F1"),
            ("edge_f1", "Edge F1"),
            ("node_f1", "Node F1"),
            ("ssi", "SSI"),
        ]
        if has_arguments:
            metric_defs += [("param_f1", "Param F1"), ("value_f1", "Value F1")]

        def _is_nan(x):
            return isinstance(x, (float, np.floating)) and np.isnan(x)

        def _is_valid_axis_value(v):
            
            if v is None or _is_nan(v):
                return False
            return v > 0

        def _to_plot_value(v):
            if v is None or _is_nan(v):
                return 0.0
            return float(v)

        # -----------------------------
        # Collect per-model metric map
        # -----------------------------
        model_metric_maps = {}

        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue

            # data = summaries[model].get(dataset, {})
            # plan = data.get("plan", {})
            # node_label_sim = plan.get("node_label_similarity", 0)
            # if node_label_sim is None or _is_nan(node_label_sim):
            #     node_label_sim = 0.0

            model_metric_maps[model] = {
                "tool_f1": scores.get("tool_f1"),
                "edge_f1": scores.get("edge_f1"),
                "node_f1": scores.get("node_f1"),
                "ssi": scores.get("ssi"),
                "param_f1": scores.get("param_f1"),
                "value_f1": scores.get("value_f1"),
            }

        if not model_metric_maps:
            continue

        # -----------------------------
        # Drop Param/Value axes if ALL models are 0/NA
        # -----------------------------
        if has_arguments:
            has_param_axis = any(_is_valid_axis_value(mm.get("param_f1")) for mm in model_metric_maps.values())
            has_value_axis = any(_is_valid_axis_value(mm.get("value_f1")) for mm in model_metric_maps.values())

            metric_defs = [
                (k, lbl) for (k, lbl) in metric_defs
                if (k != "param_f1" or has_param_axis) and (k != "value_f1" or has_value_axis)
            ]

        metrics = [k for k, _ in metric_defs]
        metric_labels = [lbl for _, lbl in metric_defs]
        n_metrics = len(metrics)

        # -----------------------------
        # Build final model_data aligned with filtered metrics
        # -----------------------------
        model_data = {}
        for model, mm in model_metric_maps.items():
            values = [_to_plot_value(mm.get(k)) for k in metrics]
            model_data[model] = values

        if not model_data:
            continue

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Compute angles for each metric
        angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
        angles += angles[:1]  # Complete the loop

        # Plot each model
        for model, values in model_data.items():
            values_closed = values + values[:1]  # Complete the loop
            color = MODEL_COLORS.get(model, FAMILY_COLORS.get(
                MODEL_INFO.get(model, {}).get("family", "Unknown"), "#999999"))
            info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
            fam = info.get('family', 'Unknown')
            sh = info.get('short', model)
            label = f"{fam}-{sh}" if fam != sh else fam

            ax.plot(angles, values_closed, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values_closed, alpha=0.15, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=11, fontweight='bold')

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Title and legend - indicate if param/value are N/A
        title_suffix = " (No Args)" if not has_arguments else ""
        ax.set_title(f'{dataset_display} - With SSI{title_suffix}', fontweight='bold', fontsize=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()

        filename = f"fig_radar_with_ssi_{dataset_short}"
        plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"[OK] Saved: {filename}.png/pdf (metrics: {n_metrics}, has_arguments={has_arguments})")


def plot_radar_without_ssi(summaries, output_dir, datasets):
    """
    Generate radar chart WITHOUT SSI metric.

    For Delta (no arguments): Show only Tool F1, Edge F1, Node F1 (3 metrics)
    For datasets with arguments: Show full metrics including Param F1, Value F1
    """
    models = get_sorted_models(summaries)

    # Filter to planning datasets only
    planning_datasets = [ds for ds in datasets
                        if not DATASET_INFO.get(ds, {}).get("has_answer", False)]

    valid_datasets = [ds for ds in planning_datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]

    if not valid_datasets:
        print("[WARN] No valid datasets for radar chart")
        return

    for dataset in valid_datasets:
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset, "short": dataset.replace("/", "_")})
        dataset_display = dataset_info.get("display", dataset)
        dataset_short = dataset_info.get("short", dataset.replace("/", "_"))

        # Determine which metrics to show based on has_arguments
        first_model_scores = None
        for m in models:
            first_model_scores = compute_facet_scores(summaries[m], dataset)
            if first_model_scores:
                break

        if not first_model_scores:
            continue

        has_arguments = first_model_scores.get("has_arguments", False)
        # -----------------------------
        # Build metric defs (candidate)
        # -----------------------------
        if has_arguments:
            metric_defs = [
                ("tool_f1", "Tool F1"),
                ("edge_f1", "Edge F1"),
                ("node_f1", "Node F1"),
                ("param_f1", "Param F1"),
                ("value_f1", "Value F1"),
            ]
        else:
            metric_defs = [
                ("tool_f1", "Tool F1"),
                ("edge_f1", "Edge F1"),
                ("node_f1", "Node F1"),
            ]

        def _is_nan(x):
            return isinstance(x, (float, np.floating)) and np.isnan(x)

        def _is_valid_axis_value(v):
            if v is None or _is_nan(v):
                return False
            return v > 0

        def _to_plot_value(v):
            if v is None or _is_nan(v):
                return 0.0
            return float(v)

        # -----------------------------
        # Collect per-model metric map
        # -----------------------------
        model_metric_maps = {}

        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue

            # data = summaries[model].get(dataset, {})
            # plan = data.get("plan", {})
            # node_label_sim = plan.get("node_label_similarity", 0)
            # if node_label_sim is None or _is_nan(node_label_sim):
            #     node_label_sim = 0.0

            model_metric_maps[model] = {
                "tool_f1": scores.get("tool_f1"),
                "edge_f1": scores.get("edge_f1"),
                "node_f1": scores.get("node_f1"),
                "param_f1": scores.get("param_f1"),
                "value_f1": scores.get("value_f1"),
            }

        if not model_metric_maps:
            continue

        # -----------------------------
        # Drop Param/Value axes if ALL models are 0/NA (only when has_arguments=True)
        # -----------------------------
        if has_arguments:
            has_param_axis = any(_is_valid_axis_value(mm.get("param_f1")) for mm in model_metric_maps.values())
            has_value_axis = any(_is_valid_axis_value(mm.get("value_f1")) for mm in model_metric_maps.values())

            metric_defs = [
                (k, lbl) for (k, lbl) in metric_defs
                if (k != "param_f1" or has_param_axis) and (k != "value_f1" or has_value_axis)
            ]

        metrics = [k for k, _ in metric_defs]
        metric_labels = [lbl for _, lbl in metric_defs]
        n_metrics = len(metrics)

        model_data = {}
        for model, mm in model_metric_maps.items():
            values = [_to_plot_value(mm.get(k)) for k in metrics]
            model_data[model] = values



        if not model_data:
            continue

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Compute angles for each metric
        angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
        angles += angles[:1]  # Complete the loop

        # Plot each model
        for model, values in model_data.items():
            values_closed = values + values[:1]  # Complete the loop
            color = MODEL_COLORS.get(model, FAMILY_COLORS.get(
                MODEL_INFO.get(model, {}).get("family", "Unknown"), "#999999"))
            info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
            fam = info.get('family', 'Unknown')
            sh = info.get('short', model)
            label = f"{fam}-{sh}" if fam != sh else fam

            ax.plot(angles, values_closed, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values_closed, alpha=0.15, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=11, fontweight='bold')

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Title and legend
        title_suffix = " (No Args)" if not has_arguments else ""
        ax.set_title(f'{dataset_display} - Without SSI{title_suffix}', fontweight='bold', fontsize=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()

        filename = f"fig_radar_without_ssi_{dataset_short}"
        plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"[OK] Saved: {filename}.png/pdf (metrics: {n_metrics}, has_arguments={has_arguments})")


def plot_radar_combined(summaries, output_dir, datasets):
    """
    Generate a combined radar chart comparing all models on core metrics.
    Creates subplots for each level (level1, level2, level3) if available.
    Uses: Node F1, Edge F1, SSI, Tool F1, Total Score
    """
    models = get_sorted_models(summaries)

    # Find all valid datasets (include all levels)
    valid_datasets = [ds for ds in datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]

    if not valid_datasets:
        return

    # Core metrics that work for all datasets
    metrics = ["node_f1", "edge_f1", "ssi", "tool_f1", "total"]
    metric_labels = ["Node F1", "Edge F1", "SSI", "Tool F1", "Total"]
    n_metrics = len(metrics)

    # Compute angles
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]

    # Determine subplot layout based on number of datasets
    n_datasets = len(valid_datasets)
    if n_datasets == 1:
        n_cols, n_rows = 1, 1
        figsize = (12, 10)
    elif n_datasets == 2:
        n_cols, n_rows = 2, 1
        figsize = (20, 8)
    else:
        n_cols, n_rows = min(3, n_datasets), (n_datasets + 2) // 3
        figsize = (8 * n_cols, 8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              subplot_kw=dict(polar=True))

    # Flatten axes for easy iteration
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Plot for each dataset
    for idx, dataset in enumerate(valid_datasets):
        if idx >= len(axes):
            break

        ax = axes[idx]
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset})
        dataset_display = dataset_info.get("display", dataset)

        # Collect data for this dataset
        model_data = {}
        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue
            values = [scores.get(m, 0) or 0 for m in metrics]
            model_data[model] = values

        if not model_data:
            ax.set_visible(False)
            continue

        # Plot each model
        for model, values in model_data.items():
            values_closed = values + values[:1]
            color = MODEL_COLORS.get(model, "#999999")
            info = MODEL_INFO.get(model, {"short": model, "family": ""})
            fam = info.get('family', '')
            sh = info.get('short', model)
            label = f"{fam}-{sh}" if fam and fam != sh else sh

            ax.plot(angles, values_closed, 'o-', linewidth=2, label=label,
                    color=color, markersize=6)
            ax.fill(angles, values_closed, alpha=0.1, color=color)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f'{dataset_display}', fontweight='bold', fontsize=14, y=1.08)

    # Hide unused subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=min(5, len(models)), fontsize=10)

    fig.suptitle('Multi-Level Model Comparison', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    plt.savefig(output_dir / "fig_radar_combined.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "fig_radar_combined.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: fig_radar_combined.png/pdf")


# ============================================================================
# Figure: 3-Facet Radar Chart (Total / Planning Fidelity / Tool Usage)
# ============================================================================

def plot_radar_3facet(summaries, output_dir, datasets):
    """
    Generate a 3-facet radar chart showing:
    - Total Score
    - Planning Fidelity (Facet A)
    - Tool Usage (Facet B)
    """
    models = get_sorted_models(summaries)

    # Filter to planning datasets only
    planning_datasets = [ds for ds in datasets
                        if not DATASET_INFO.get(ds, {}).get("has_answer", False)]

    valid_datasets = [ds for ds in planning_datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]

    if not valid_datasets:
        print("[WARN] No valid datasets for 3-facet radar chart")
        return

    for dataset in valid_datasets:
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset, "short": dataset.replace("/", "_")})
        dataset_display = dataset_info.get("display", dataset)
        dataset_short = dataset_info.get("short", dataset.replace("/", "_"))

        # Define 3 facets
        metrics = ["total", "facet_a", "facet_b"]
        metric_labels = ["Total", "Planning Fidelity", "Tool Usage"]
        n_metrics = len(metrics)

        # Collect data for each model
        model_data = {}

        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue

            values = [
                scores.get("total", 0) or 0,
                scores.get("facet_a", 0) or 0,
                scores.get("facet_b", 0) or 0,
            ]
            model_data[model] = values

        if not model_data:
            continue

        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Compute angles for each metric
        angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
        angles += angles[:1]  # Complete the loop

        # Plot each model
        for model, values in model_data.items():
            values_closed = values + values[:1]  # Complete the loop
            color = MODEL_COLORS.get(model, FAMILY_COLORS.get(
                MODEL_INFO.get(model, {}).get("family", "Unknown"), "#999999"))
            info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
            fam = info.get('family', 'Unknown')
            sh = info.get('short', model)
            label = f"{fam}-{sh}" if fam != sh else fam

            ax.plot(angles, values_closed, 'o-', linewidth=2, label=label, color=color)
            ax.fill(angles, values_closed, alpha=0.15, color=color)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=12, fontweight='bold')

        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Title and legend
        ax.set_title(f'{dataset_display} - 3-Facet Evaluation', fontweight='bold', fontsize=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()

        filename = f"fig_radar_3facet_{dataset_short}"
        plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"[OK] Saved: {filename}.png/pdf")


# ============================================================================
# Figure: Per-Dataset Facet Scores
# ============================================================================

def plot_per_dataset_facet_scores(summaries, output_dir, datasets):
    """Generate per-dataset facet score comparison."""
    models = get_sorted_models(summaries)
    
    valid_datasets = [ds for ds in datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]
    
    if not valid_datasets:
        print("[WARN] No valid datasets to plot")
        return
    
    n_datasets = len(valid_datasets)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 7))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(valid_datasets):
        ax = axes[idx]
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset, "has_answer": False})
        dataset_display = dataset_info.get("display", dataset)
        has_answer = dataset_info.get("has_answer", False)
        
        # Collect data
        model_names = []
        facet_a_vals = []
        facet_b_vals = []
        facet_c_vals = []
        total_vals = []
        has_args_list = []
        colors = []
        
        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue
            
            info = MODEL_INFO.get(model, {"display": model, "family": "Unknown"})
            model_names.append(info.get("display", model))
            facet_a_vals.append(scores["facet_a"])
            facet_b_vals.append(scores["facet_b"])
            facet_c_vals.append(scores["facet_c"] if scores["facet_c"] is not None else 0)
            total_vals.append(scores["total"])
            has_args_list.append(scores.get("has_arguments", True))
            colors.append(FAMILY_COLORS.get(info.get("family"), "#999999"))
        
        if not model_names:
            ax.set_title(f"{dataset_display}\n(No data)", fontweight='bold')
            continue
        
        n = len(model_names)
        x = np.arange(n)
        
        # Determine Facet B label based on has_arguments
        has_args = has_args_list[0] if has_args_list else True
        facet_b_label = 'Facet B: Tool Usage' if has_args else 'Facet B: Tool Selection'
        
        if has_answer:
            # GAIA: Show Facet C (Answer) only
            width = 0.6
            bars = ax.bar(x, facet_c_vals, width, label='Answer Accuracy', 
                         color='#2A9D8F', alpha=0.85, edgecolor='black', linewidth=0.8)
            
            for bar, val in zip(bars, facet_c_vals):
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Answer Accuracy', fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
        else:
            # Planning datasets: Show Facet A, Facet B, and Total
            width = 0.25
            
            bars1 = ax.bar(x - width, facet_a_vals, width, label='Facet A: Planning', 
                          color='#E63946', alpha=0.85, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x, facet_b_vals, width, label=facet_b_label,
                          color='#457B9D', alpha=0.85, edgecolor='black', linewidth=0.5)
            bars3 = ax.bar(x + width, total_vals, width, label='Total Score',
                          color='#2A9D8F', alpha=0.85, edgecolor='black', linewidth=0.5)
            
            # Annotate total scores
            for bar, val in zip(bars3, total_vals):
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
            
            ax.set_ylabel('Score', fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
        
        ax.set_title(dataset_display, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        # Use single-line short names to prevent crowding
        short_names = [name.replace('\n', ' ') for name in model_names]
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Dual-Facet Evaluation by Dataset", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_dir / "fig1_facet_by_dataset.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "fig1_facet_by_dataset.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: fig1_facet_by_dataset.png/pdf")


# ============================================================================
# Figure: Detailed Metrics Bar Chart
# ============================================================================

def plot_detailed_metrics_bar(summaries, output_dir, datasets):
    """Generate detailed metrics bar chart (replacement for heatmap)."""
    models = get_sorted_models(summaries)
    
    planning_datasets = [ds for ds in datasets 
                        if not DATASET_INFO.get(ds, {}).get("has_answer", False)]
    
    valid_datasets = [ds for ds in planning_datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]
    
    if not valid_datasets:
        return
    
    for dataset in valid_datasets:
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset, "short": dataset.replace("/", "_")})
        dataset_display = dataset_info.get("display", dataset)
        dataset_short = dataset_info.get("short", dataset.replace("/", "_"))
        
        # Check if has arguments
        first_scores = None
        for m in models:
            first_scores = compute_facet_scores(summaries[m], dataset)
            if first_scores:
                break
        
        if not first_scores:
            continue
        
        has_args = first_scores.get("has_arguments", False)
        
        if has_args:
            metrics = ["node_f1", "edge_f1", "ssi", "tool_f1", "param_f1", "value_f1"]
            metric_labels = ["Node F1", "Edge F1", "SSI", "Tool F1", "Param F1", "Value F1"]
            colors = ['#E63946', '#E63946', '#E63946', '#457B9D', '#457B9D', '#457B9D']
        else:
            metrics = ["node_f1", "edge_f1", "ssi", "tool_f1"]
            metric_labels = ["Node F1", "Edge F1", "SSI", "Tool F1"]
            colors = ['#E63946', '#E63946', '#E63946', '#457B9D']
        
        # Collect data
        model_names = []
        data_rows = []
        
        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue
            
            info = MODEL_INFO.get(model, {"short": model, "family": ""})
            model_names.append(f"{info.get('family', '')}-{info.get('short', model)}")
            
            row = []
            for metric in metrics:
                val = scores.get(metric)
                row.append(val if val is not None else 0)
            data_rows.append(row)
        
        if not data_rows:
            continue
        
        # Create grouped bar chart
        n_models = len(model_names)
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 6))
        
        x = np.arange(n_models)
        width = 0.8 / n_metrics
        
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
            offset = (i - (n_metrics - 1) / 2) * width
            values = [row[i] for row in data_rows]
            alpha = 0.7 + 0.3 * (i % 3) / 3  # Vary alpha slightly
            bars = ax.bar(x + offset, values, width, label=label, color=color, 
                         alpha=alpha, edgecolor='black', linewidth=0.3)
        
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_title(f'{dataset_display} - Detailed Metrics', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"fig2_detailed_{dataset_short}"
        plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"[OK] Saved: {filename}.png/pdf")


# ============================================================================
# Figure: Model Scaling Analysis
# ============================================================================

def plot_model_scaling(summaries, output_dir, datasets):
    """Plot how metrics scale with model size (for Qwen2.5 family)."""
    models = get_sorted_models(summaries)
    
    # Filter to Qwen2.5 models
    qwen_models = [m for m in models if m.startswith("qwen2.5")]
    
    if len(qwen_models) < 2:
        print("[INFO] Not enough Qwen2.5 models for scaling analysis")
        return
    
    valid_datasets = [ds for ds in datasets if any(
        ds in summaries.get(m, {}) for m in qwen_models
    )]
    
    if not valid_datasets:
        return
    
    dataset = valid_datasets[0]
    dataset_info = DATASET_INFO.get(dataset, {"display": dataset})
    dataset_display = dataset_info.get("display", dataset)
    
    # Collect data
    sizes = []
    node_f1s = []
    edge_f1s = []
    tool_f1s = []
    totals = []
    
    for model in qwen_models:
        scores = compute_facet_scores(summaries[model], dataset)
        if scores is None:
            continue
        
        info = MODEL_INFO.get(model, {})
        size = info.get("size_b", 0)
        
        sizes.append(size)
        node_f1s.append(scores["node_f1"])
        edge_f1s.append(scores["edge_f1"])
        tool_f1s.append(scores["tool_f1"])
        totals.append(scores["total"])
    
    if len(sizes) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sizes, node_f1s, 'o-', linewidth=2, markersize=10, label='Node F1', color='#E63946')
    ax.plot(sizes, edge_f1s, 's-', linewidth=2, markersize=10, label='Edge F1', color='#F4A261')
    ax.plot(sizes, tool_f1s, '^-', linewidth=2, markersize=10, label='Tool F1', color='#457B9D')
    ax.plot(sizes, totals, 'D-', linewidth=3, markersize=12, label='Total', color='#2A9D8F')
    
    ax.set_xlabel('Model Size (Billion Parameters)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Qwen2.5 Model Scaling - {dataset_display}', fontweight='bold', fontsize=14)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add model size labels
    for size, total in zip(sizes, totals):
        ax.annotate(f'{size}B', xy=(size, total), xytext=(0, 10),
                   textcoords="offset points", ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    plt.savefig(output_dir / "fig3_model_scaling.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "fig3_model_scaling.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: fig3_model_scaling.png/pdf")


# ============================================================================
# Figure: Parse Error vs Performance
# ============================================================================

def plot_parse_error_impact(summaries, output_dir, datasets):
    """Plot relationship between parse error rate and performance."""
    models = get_sorted_models(summaries)
    
    valid_datasets = [ds for ds in datasets if any(
        ds in summaries.get(m, {}) for m in models
    )]
    
    if not valid_datasets:
        return
    
    dataset = valid_datasets[0]
    dataset_info = DATASET_INFO.get(dataset, {"display": dataset})
    dataset_display = dataset_info.get("display", dataset)
    
    # Collect data
    parse_errors = []
    totals = []
    model_labels = []
    colors = []
    
    for model in models:
        scores = compute_facet_scores(summaries[model], dataset)
        if scores is None:
            continue
        
        info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
        parse_errors.append(scores.get("parse_error_rate", 0) * 100)  # Convert to %
        totals.append(scores["total"])
        model_labels.append(f"{info.get('family', '')}-{info.get('short', model)}")
        colors.append(MODEL_COLORS.get(model, "#999999"))
    
    if len(parse_errors) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(parse_errors, totals, c=colors, s=200, alpha=0.8, edgecolors='black', linewidth=1)
    
    for i, label in enumerate(model_labels):
        ax.annotate(label, xy=(parse_errors[i], totals[i]), xytext=(5, 5),
                   textcoords="offset points", fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Parse Error Rate (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Total Score', fontweight='bold', fontsize=12)
    ax.set_title(f'Parse Error Impact on Performance - {dataset_display}', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-2, max(parse_errors) + 5 if parse_errors else 10)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(output_dir / "fig4_parse_error_impact.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "fig4_parse_error_impact.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: fig4_parse_error_impact.png/pdf")


# ============================================================================
# Figure: Overall Comparison Bar Chart (from generate_figures.py)
# ============================================================================

def plot_overall_bar_chart(summaries, output_dir):
    """
    Generate overall comparison bar chart using the current planning/tool stack.

    The chart now follows the paper-facing metric grouping:
    - Planning Score
    - SpanNodeF1
    - DW-OrderF1
    - Tool Usage Score
    - Tool Name F1
    - Tool Param F1
    - Tool Value F1
    - Parse Error Rate
    - Exact Match / Token F1 when answer metrics are available
    """
    models = get_sorted_models(summaries)

    # Extract metrics
    data = {
        'models': [],
        'display_names': [],
        'families': [],
        'planning_score': [],
        'node_f1': [],
        'dw_order_f1': [],
        'tool_usage_score': [],
        'tool_name_f1': [],
        'param_f1': [],
        'value_f1': [],
        'parse_error_rate': [],
        'exact_match': [],
        'token_f1': [],
    }

    for model in models:
        summary = summaries[model]
        overall = summary.get("_overall", {})
        if not overall:
            continue

        info = MODEL_INFO.get(model, {"family": "Unknown", "display": model})
        plan = overall.get("plan", {})
        tool = overall.get("tool", {})
        answer = overall.get("answer", {}) or {}

        data['models'].append(model)
        data['display_names'].append(info.get("display", model))
        data['families'].append(info.get("family", "Unknown"))
        data['planning_score'].append(plan.get("planning_score", 0) or 0)
        data['node_f1'].append(plan.get("node_f1", 0) or 0)
        data['dw_order_f1'].append(plan.get("dw_order_f1", 0) or 0)
        data['tool_usage_score'].append(tool.get("tool_usage_score", 0) or 0)
        data['tool_name_f1'].append(tool.get("tool_name_f1", 0) or 0)
        data['parse_error_rate'].append((overall.get("parse_error_rate", 0) or 0) * 100.0)
        data['exact_match'].append(answer.get("exact_match"))
        data['token_f1'].append(answer.get("token_f1"))

        # Keep None as None for proper N/A detection
        param_val = tool.get("param_name_f1")
        value_val = tool.get("type_aware_value_f1")

        data['param_f1'].append(param_val)
        data['value_f1'].append(value_val)

    if not data['models']:
        print("[WARN] No models with _overall data for overall bar chart")
        return

    n_models = len(data['models'])
    # Use MODEL_COLORS if available (for distinct Qwen blues), else fall back to family color
    colors = [
        MODEL_COLORS.get(m, FAMILY_COLORS.get(f, '#999999')) 
        for m, f in zip(data['models'], data['families'])
    ]

    # Check if param_f1 and value_f1 have valid data (not all None or 0)
    def has_valid_data(values):
        """Check if any value is not None and > 0"""
        for v in values:
            if v is not None and v > 0:
                return True
        return False

    has_param_f1 = has_valid_data(data['param_f1'])
    has_value_f1 = has_valid_data(data['value_f1'])
    has_answer_scores = has_valid_data(data['exact_match']) or has_valid_data(data['token_f1'])

    # Determine which metrics to show
    metric_specs = [
        ('planning_score', 'Planning Score', 'score'),
        ('node_f1', 'SpanNodeF1', 'score'),
        ('dw_order_f1', 'DW-OrderF1', 'score'),
        ('tool_usage_score', 'Tool Usage Score', 'score'),
        ('tool_name_f1', 'Tool Name F1', 'score'),
        ('parse_error_rate', 'Parse Error Rate (%)', 'percent'),
    ]

    if has_param_f1:
        metric_specs.append(('param_f1', 'Tool Param F1', 'score'))
    if has_value_f1:
        metric_specs.append(('value_f1', 'Tool Value F1', 'score'))
    if has_answer_scores:
        metric_specs.extend([
            ('exact_match', 'Exact Match', 'score'),
            ('token_f1', 'Token F1', 'score'),
        ])

    n_metrics = len(metric_specs)

    # Determine grid layout
    if n_metrics <= 4:
        n_rows, n_cols = 2, 2
    elif n_metrics <= 8:
        n_rows, n_cols = 2, 4
    elif n_metrics <= 10:
        n_rows, n_cols = 2, 5
    else:
        n_rows, n_cols = 3, 3

    # Create figure with dynamic size
    fig_width = max(12, n_cols * 5)
    fig_height = max(8, n_rows * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    x = np.arange(n_models)

    for idx, (metric_key, metric_title, metric_kind) in enumerate(metric_specs):
        ax = axes[idx]
        values = data[metric_key]

        # Convert None to 0 for plotting
        plot_values = [v if v is not None else 0 for v in values]

        bars = ax.bar(x, plot_values, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

        ax.set_ylabel(metric_title, fontweight='bold')
        ax.set_xticks(x)
        # Use single-line names to prevent alignment drift
        single_line_names = [name.replace('\n', ' ') for name in data['display_names']]
        ax.set_xticklabels(single_line_names, rotation=45, ha='right', fontsize=8)
        if metric_kind == 'percent':
            max_val = max(plot_values) if plot_values else 0
            upper = max(5, min(100, max_val * 1.20 + 1))
            ax.set_ylim(0, upper)
            ax.axhline(y=5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        else:
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, plot_values):
            height = bar.get_height()
            label = f'{val:.1f}' if metric_kind == 'percent' else f'{val:.2f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Add legend with all unique families
    unique_families = list(dict.fromkeys(data['families']))
    handles = [mpatches.Patch(color=FAMILY_COLORS.get(f, '#999999'), label=f, alpha=0.85)
               for f in unique_families]
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 6),
               bbox_to_anchor=(0.5, 0.98), fontsize=11, frameon=True)

    plt.suptitle("Overall Model Comparison on the Current Evaluation Stack",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    plt.savefig(output_dir / "fig_overall_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / "fig_overall_comparison.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(
        "[OK] Saved: fig_overall_comparison.png/pdf "
        f"(metrics: {n_metrics}, has_param_f1={has_param_f1}, "
        f"has_value_f1={has_value_f1}, has_answer_scores={has_answer_scores})"
    )


# ============================================================================
# Print Summary Table
# ============================================================================

def print_facet_summary(summaries, datasets):
    """Print facet scores summary table with correct scoring logic."""
    models = get_sorted_models(summaries)
    
    print("\n" + "=" * 110)
    print("DUAL-FACET EVALUATION SUMMARY (FIXED SCORING)")
    print("=" * 110)
    print(f"\nFacet A (Planning Fidelity) = PlanningScore = avg(SpanNodeF1, DW-Order F1)")
    print(f"Facet B (Tool Usage):")
    print(f"  - If has_arguments=True:  avg(Tool F1, Param F1, Value F1)")
    print(f"  - If has_arguments=False: Tool F1 only  ← Delta uses this!")
    print(f"Total = (Facet A + Facet B) / 2")
    
    for dataset in datasets:
        dataset_info = DATASET_INFO.get(dataset, {"display": dataset, "has_answer": False})
        dataset_display = dataset_info.get("display", dataset)
        has_answer = dataset_info.get("has_answer", False)
        
        has_data = any(dataset in summaries.get(m, {}) for m in models)
        if not has_data:
            continue
        
        print(f"\n{'-' * 110}")
        print(f"Dataset: {dataset_display}")
        print(f"{'-' * 110}")
        
        # Get has_arguments from first available model
        first_scores = None
        for m in models:
            first_scores = compute_facet_scores(summaries[m], dataset)
            if first_scores:
                break
        
        has_args = first_scores.get("has_arguments", True) if first_scores else True
        
        if has_args:
            header = f"{'Model':<22} | {'Planning':<8} | {'Node F1':<8} | {'DW-Ord':<8} | {'Tool F1':<8} | {'Param F1':<8} | {'Value F1':<8} | {'ToolUse':<8} | {'Total':<8}"
        else:
            header = f"{'Model':<22} | {'Planning':<8} | {'Node F1':<8} | {'DW-Ord':<8} | {'Tool F1':<8} | {'ToolUse':<8} | {'Total':<8} | {'Parse%':<8}"
        
        print(header)
        print("-" * 110)
        
        for model in models:
            scores = compute_facet_scores(summaries[model], dataset)
            if scores is None:
                continue
            
            info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
            name = f"{info.get('family', 'Unknown')}-{info.get('short', model)}"
            
            if has_args:
                param_f1 = scores['param_f1'] if scores['param_f1'] is not None else 0
                value_f1 = scores['value_f1'] if scores['value_f1'] is not None else 0
                print(f"{name:<22} | {scores['facet_a']:<8.4f} | {scores['node_f1']:<8.4f} | {scores['dw_order_f1']:<8.4f} | {scores['tool_f1']:<8.4f} | {param_f1:<8.4f} | {value_f1:<8.4f} | {scores['facet_b']:<8.4f} | {scores['total']:<8.4f}")
            else:
                parse_pct = scores.get('parse_error_rate', 0) * 100
                print(f"{name:<22} | {scores['facet_a']:<8.4f} | {scores['node_f1']:<8.4f} | {scores['dw_order_f1']:<8.4f} | {scores['tool_f1']:<8.4f} | {scores['facet_b']:<8.4f} | {scores['total']:<8.4f} | {parse_pct:<8.1f}")
    
    print("=" * 110)


# ============================================================================
# Final Answer Visualization (Facet C)
# ============================================================================

def plot_final_answer_metrics(summaries: dict, output_dir: Path, datasets: list = None):
    """
    Plot final answer metrics (Exact Match and Token F1) for GAIA-like datasets.
    Only plots for datasets with has_answer=True.
    """
    models = sorted(summaries.keys(), key=lambda m: MODEL_INFO.get(m, {}).get("order", 999))

    if datasets is None:
        datasets = get_available_datasets(summaries)

    # Filter to datasets with final answer
    answer_datasets = [ds for ds in datasets
                       if DATASET_INFO.get(ds, {}).get("has_answer", False)]

    # Also check for gaia subsets dynamically
    for model in models:
        for ds in summaries.get(model, {}).keys():
            if ds.startswith("gaia/") and ds not in answer_datasets:
                answer_datasets.append(ds)

    if not answer_datasets:
        print("[INFO] No datasets with final answer metrics found, skipping final answer plot")
        return

    # Collect data
    all_data = {}
    for model in models:
        model_data = summaries.get(model, {})
        all_data[model] = {}

        for ds in answer_datasets:
            if ds in model_data:
                answer = model_data[ds].get("answer", {})
                exact_match = answer.get("exact_match", 0) or 0
                token_f1 = answer.get("token_f1", 0) or 0
                num_samples = answer.get("num_with_answer", 0)
                all_data[model][ds] = {
                    "exact_match": exact_match,
                    "token_f1": token_f1,
                    "num_samples": num_samples
                }

    # Plot 1: Overall Final Answer Comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    # Aggregate across all answer datasets
    exact_matches = []
    token_f1s = []
    correct_counts = []
    total_counts = []

    for model in models:
        c_count = 0
        t_count = 0
        tf_sum = 0
        
        for ds in answer_datasets:
            if ds in all_data[model]:
                data = all_data[model][ds]
                em = data.get("exact_match", 0)
                tf = data.get("token_f1", 0)
                ns = data.get("num_samples", 0)
                
                c_count += int(round(em * ns))
                t_count += ns
                tf_sum += tf * ns

        exact_matches.append(c_count / t_count if t_count > 0 else 0)
        token_f1s.append(tf_sum / t_count if t_count > 0 else 0)
        correct_counts.append(c_count)
        total_counts.append(t_count)

    bars1 = ax.bar(x - width/2, exact_matches, width, label='Exact Match', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, token_f1s, width, label='Token F1', color='#3498db', alpha=0.8)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, exact_matches)):
        if val > 0 or total_counts[i] > 0:
            label = f'{val:.3f}\n({correct_counts[i]}/{total_counts[i]})'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   label, ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, token_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Facet C: Final Answer Accuracy (GAIA)')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_INFO.get(m, {}).get("display", m).replace("\n", " ") for m in models],
                       rotation=45, ha='right')
    ax.legend()
    # Adjust ylim to make room for text
    ax.set_ylim(0, max(max(exact_matches) if exact_matches else 0.1,
                       max(token_f1s) if token_f1s else 0.1) * 1.3 + 0.05)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        fig.savefig(output_dir / f'fig_final_answer_comparison.{ext}',
                   dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] fig_final_answer_comparison.png/pdf")

    # Plot 2: Per-level breakdown (if multiple GAIA levels)
    gaia_levels = [ds for ds in answer_datasets if ds.startswith("gaia/")]

    if len(gaia_levels) > 1:
        fig, axes = plt.subplots(1, len(gaia_levels), figsize=(5*len(gaia_levels), 5))
        if len(gaia_levels) == 1:
            axes = [axes]

        for idx, level in enumerate(sorted(gaia_levels)):
            ax = axes[idx]
            level_name = level.replace("gaia/", "").replace("_", " ").title()

            em_vals = []
            tf_vals = []
            c_counts = []
            t_counts = []
            valid_models = []

            for model in models:
                if level in all_data[model]:
                    data = all_data[model][level]
                    em = data["exact_match"]
                    ns = data["num_samples"]
                    em_vals.append(em)
                    tf_vals.append(data["token_f1"])
                    c_counts.append(int(round(em * ns)))
                    t_counts.append(ns)
                    valid_models.append(model)

            if valid_models:
                x = np.arange(len(valid_models))
                bars1 = ax.bar(x - width/2, em_vals, width, label='Exact Match', color='#2ecc71', alpha=0.8)
                bars2 = ax.bar(x + width/2, tf_vals, width, label='Token F1', color='#3498db', alpha=0.8)

                # Add labels
                for i, (bar, val) in enumerate(zip(bars1, em_vals)):
                   if val > 0 or t_counts[i] > 0:
                       label = f'{val:.2f}\n({c_counts[i]}/{t_counts[i]})'
                       ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                              label, ha='center', va='bottom', fontsize=7)

                ax.set_xlabel('Model')
                ax.set_ylabel('Score')
                ax.set_title(f'GAIA {level_name}')
                ax.set_xticks(x)
                # Use display name (replacing newline with space) for better readability
                ax.set_xticklabels([MODEL_INFO.get(m, {}).get("display", m).replace("\n", " ") for m in valid_models],
                                   rotation=45, ha='right', fontsize=9)
                ax.legend(fontsize=8)
                # Dynamic y-axis limit based on actual data
                max_val = max(max(em_vals) if em_vals else 0.1, max(tf_vals) if tf_vals else 0.1)
                ax.set_ylim(0, max(max_val * 1.3 + 0.05, 0.2))

        plt.tight_layout()

        for ext in ['png', 'pdf']:
            fig.savefig(output_dir / f'fig_final_answer_by_level.{ext}',
                       dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[SAVED] fig_final_answer_by_level.png/pdf")


# ============================================================================
# Figure: Combined GAIA 3-Level 3-Facet Evaluation
# ============================================================================

def plot_gaia_combined_3facet(summaries, output_dir, datasets):
    """
    Generate a combined 3-facet radar chart for ALL GAIA levels together.
    This aggregates level1, level2, level3 into a single view.

    Metrics shown:
    - Total Score (average across levels)
    - Planning Fidelity (Facet A)
    - Tool Usage (Facet B)
    """
    models = get_sorted_models(summaries)

    # Find all GAIA level datasets
    gaia_levels = [ds for ds in datasets if ds.startswith("gaia/level")]

    if not gaia_levels:
        print("[INFO] No GAIA level datasets found for combined 3-facet chart")
        return

    # Define 3 facets
    metrics = ["total", "facet_a", "facet_b"]
    metric_labels = ["Total Score", "Planning Fidelity", "Tool Usage"]
    n_metrics = len(metrics)

    # Collect AGGREGATED data for each model (average across all levels)
    model_data = {}

    for model in models:
        total_vals = []
        facet_a_vals = []
        facet_b_vals = []

        for level in gaia_levels:
            scores = compute_facet_scores(summaries.get(model, {}), level)
            if scores is not None:
                total_vals.append(scores.get("total", 0) or 0)
                facet_a_vals.append(scores.get("facet_a", 0) or 0)
                facet_b_vals.append(scores.get("facet_b", 0) or 0)

        if total_vals:  # Only include models that have data
            model_data[model] = [
                np.mean(total_vals),
                np.mean(facet_a_vals),
                np.mean(facet_b_vals),
            ]

    if not model_data:
        print("[INFO] No model data for combined GAIA 3-facet chart")
        return

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Compute angles for each metric
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]  # Complete the loop

    # Plot each model
    for model, values in model_data.items():
        values_closed = values + values[:1]  # Complete the loop
        color = MODEL_COLORS.get(model, FAMILY_COLORS.get(
            MODEL_INFO.get(model, {}).get("family", "Unknown"), "#999999"))
        info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
        fam = info.get('family', 'Unknown')
        sh = info.get('short', model)
        label = f"{fam}-{sh}" if fam != sh else fam

        ax.plot(angles, values_closed, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values_closed, alpha=0.15, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=12, fontweight='bold')

    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Title and legend
    level_str = ", ".join([l.replace("gaia/", "") for l in sorted(gaia_levels)])
    ax.set_title(f'GAIA Combined ({level_str}) - 3-Facet Evaluation',
                 fontweight='bold', fontsize=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()

    filename = "fig_gaia_combined_3facet"
    plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: {filename}.png/pdf (levels: {level_str})")


def plot_gaia_level_comparison(summaries, output_dir, datasets):
    """
    Generate a bar chart comparing model performance across GAIA levels.
    Shows Total Score, Planning, and Tool metrics side by side for each level.
    """
    models = get_sorted_models(summaries)

    # Find all GAIA level datasets
    gaia_levels = sorted([ds for ds in datasets if ds.startswith("gaia/level")])

    if not gaia_levels:
        print("[INFO] No GAIA level datasets found for level comparison")
        return

    n_levels = len(gaia_levels)
    n_models = len(models)

    # Collect data: model -> level -> score
    level_data = {level: {} for level in gaia_levels}

    for model in models:
        for level in gaia_levels:
            scores = compute_facet_scores(summaries.get(model, {}), level)
            if scores is not None:
                level_data[level][model] = scores.get("total", 0) or 0

    # Create figure with subplots for each level
    fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 6), sharey=True)
    if n_levels == 1:
        axes = [axes]

    # Prepare data for plotting
    for idx, level in enumerate(gaia_levels):
        ax = axes[idx]
        level_name = level.replace("gaia/", "").replace("_", " ").title()

        model_names = []
        scores = []
        colors = []

        for model in models:
            if model in level_data[level]:
                info = MODEL_INFO.get(model, {"short": model, "family": "Unknown"})
                model_names.append(info.get("short", model))
                scores.append(level_data[level][model])
                colors.append(MODEL_COLORS.get(model, "#999999"))

        if not scores:
            ax.set_title(f'{level_name}\n(No data)', fontweight='bold')
            continue

        x = np.arange(len(model_names))
        bars = ax.bar(x, scores, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

        # Add value labels
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'GAIA {level_name}', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3)

    axes[0].set_ylabel('Total Score', fontweight='bold', fontsize=11)

    plt.suptitle('GAIA Performance by Level', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    filename = "fig_gaia_level_comparison"
    plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: {filename}.png/pdf")


# ============================================================================
# Figure: GAIA 5-Axis Radar (Level 1/2/3)
# ============================================================================

def plot_gaia_5axis_radar(summaries, output_dir, datasets):
    """
    Generate a 5-axis radar chart for each GAIA level (1, 2, 3) side-by-side.
    Axes: Plan Node F1, Plan Edge F1, Tool Name F1, Param Name F1, Exact Match
    """
    models = get_sorted_models(summaries)
    
    # Identify GAIA levels
    gaia_levels = sorted([ds for ds in datasets if ds.startswith("gaia/level")])
    if not gaia_levels:
        print("[INFO] No GAIA level datasets found for 5-axis radar")
        return

    # Metrics configuration
    metrics = ["node_f1", "edge_f1", "tool_f1", "param_f1", "exact_match"]
    metric_labels = ["Plan Node F1", "Plan Edge F1", "Tool Name F1", "Param Name F1", "Exact Match"]
    n_metrics = len(metrics)

    # Setup figure
    fig, axes = plt.subplots(1, len(gaia_levels), figsize=(6 * len(gaia_levels), 6), subplot_kw=dict(polar=True))
    if len(gaia_levels) == 1:
        axes = [axes]

    # Angles
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]

    for idx, level in enumerate(gaia_levels):
        ax = axes[idx]
        level_name = level.replace("gaia/", "").replace("_", " ").title()

        # Collect data
        model_data = {}
        for model in models:
            summary = summaries.get(model, {})
            scores = compute_facet_scores(summary, level)
            
            if scores is None:
                continue

            # Need to fetch raw exact_match from summary -> answer
            answer_data = summary.get(level, {}).get("answer", {})
            exact_match = answer_data.get("exact_match", 0) or 0

            values = [
                scores.get("node_f1", 0) or 0,
                scores.get("edge_f1", 0) or 0,
                scores.get("tool_f1", 0) or 0,
                scores.get("param_f1", 0) or 0, # Fallback to 0 if None
                exact_match
            ]
            model_data[model] = values

        if not model_data:
            ax.set_visible(False)
            continue
            
        # Plot
        for model, values in model_data.items():
            values_closed = values + values[:1]
            color = MODEL_COLORS.get(model, FAMILY_COLORS.get(
                MODEL_INFO.get(model, {}).get("family", "Unknown"), "#999999"))
            info = MODEL_INFO.get(model, {"short": model})
            label = info.get("short", model)

            ax.plot(angles, values_closed, linewidth=2, label=label, color=color)
            ax.fill(angles, values_closed, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=9, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=7)
        ax.set_title(level_name, fontweight='bold', fontsize=14, y=1.1)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
               ncol=min(5, len(models)), fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for legend
    
    filename = "fig_gaia_5axis_radar"
    plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: {filename}.png/pdf")


# ============================================================================
# Figure: GAIA 3-Facet Radar (Level 1/2/3)
# ============================================================================

def plot_gaia_3facet_radar(summaries, output_dir, datasets):
    """
    Generate a 3-axis radar chart for each GAIA level (1, 2, 3) side-by-side.
    Axes: Planning Score, Tool Usage Score, Answer Score
    """
    models = get_sorted_models(summaries)
    
    gaia_levels = sorted([ds for ds in datasets if ds.startswith("gaia/level")])
    if not gaia_levels:
        print("[INFO] No GAIA level datasets found for 3-facet radar")
        return

    metrics = ["facet_a", "facet_b", "facet_c"]
    metric_labels = ["Planning Score", "Tool Usage Score", "Answer Score"]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, len(gaia_levels), figsize=(6 * len(gaia_levels), 6), subplot_kw=dict(polar=True))
    if len(gaia_levels) == 1:
        axes = [axes]

    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]

    for idx, level in enumerate(gaia_levels):
        ax = axes[idx]
        level_name = level.replace("gaia/", "").replace("_", " ").title()

        model_data = {}
        for model in models:
            summary = summaries.get(model, {})
            scores = compute_facet_scores(summary, level)
            if scores is None:
                continue
            
            values = [
                scores.get("facet_a", 0) or 0,
                scores.get("facet_b", 0) or 0,
                scores.get("facet_c", 0) or 0,
            ]
            model_data[model] = values

        if not model_data:
            ax.set_visible(False)
            continue
            
        for model, values in model_data.items():
            values_closed = values + values[:1]
            color = MODEL_COLORS.get(model, FAMILY_COLORS.get(
                MODEL_INFO.get(model, {}).get("family", "Unknown"), "#999999"))
            info = MODEL_INFO.get(model, {"display": model})
            label = info.get("display", model).replace("\n", " ")

            ax.plot(angles, values_closed, linewidth=2, label=label, color=color)
            ax.fill(angles, values_closed, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=7)
        ax.set_title(level_name, fontweight='bold', fontsize=14, y=1.1)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Shared legend (reuse if possible, or create new)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
               ncol=min(5, len(models)), fontsize=10, frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    filename = "fig_gaia_3facet_radar"
    plt.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"[OK] Saved: {filename}.png/pdf")


# ============================================================================
# Multi-Run Aggregated Bar Chart (with error bars)
# ============================================================================

def plot_aggregated_bar_chart(aggregated_path, output_dir):
    """
    Generate bar chart with error bars from aggregated multi-run results.

    Args:
        aggregated_path: Path to aggregated_all_models.json
        output_dir: Output directory for figures
    """
    aggregated_path = Path(aggregated_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated_path.exists():
        print(f"[WARN] Aggregated file not found: {aggregated_path}")
        return

    with open(aggregated_path) as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    num_runs = metadata.get("num_runs", 1)
    models_data = data.get("models", {})

    if not models_data:
        print("[WARN] No model data in aggregated file")
        return

    # Collect metrics for each model
    models = list(models_data.keys())

    # Prepare data structure for plotting
    metrics_to_plot = {
        "node_f1": {"label": "Node F1", "means": [], "stds": []},
        "edge_f1": {"label": "Edge F1", "means": [], "stds": []},
        "ssi": {"label": "SSI", "means": [], "stds": []},
        "tool_name_f1": {"label": "Tool F1", "means": [], "stds": []},
        "exact_match": {"label": "Exact Match", "means": [], "stds": []},
    }

    for model in models:
        model_data = models_data.get(model, {})

        # Get first dataset key (usually just one for GAIA per-category runs)
        dataset_keys = [k for k in model_data.keys() if k not in ["num_runs", "_overall"]]
        if not dataset_keys:
            continue

        # Aggregate across all dataset keys for this model
        for metric_key, metric_info in metrics_to_plot.items():
            means_collected = []
            stds_collected = []

            for dk in dataset_keys:
                dk_data = model_data.get(dk, {})

                # Check in plan, tool, and answer sections
                for section in ["plan", "tool", "answer"]:
                    section_data = dk_data.get(section, {})
                    if section_data and metric_key in section_data:
                        metric_agg = section_data.get(metric_key, {})
                        if isinstance(metric_agg, dict):
                            m = metric_agg.get("mean")
                            s = metric_agg.get("std")
                            if m is not None:
                                means_collected.append(m)
                            if s is not None:
                                stds_collected.append(s)

            # Average across datasets if multiple
            if means_collected:
                metric_info["means"].append(sum(means_collected) / len(means_collected))
                if stds_collected:
                    # Use pooled std if available
                    metric_info["stds"].append(sum(stds_collected) / len(stds_collected))
                else:
                    metric_info["stds"].append(0)
            else:
                metric_info["means"].append(0)
                metric_info["stds"].append(0)

    # Filter out metrics with no data
    metrics_with_data = {
        k: v for k, v in metrics_to_plot.items()
        if any(m > 0 for m in v["means"])
    }

    if not metrics_with_data:
        print("[WARN] No metrics with valid data for aggregated bar chart")
        return

    # Create figure
    n_metrics = len(metrics_with_data)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for idx, (metric_key, metric_info) in enumerate(metrics_with_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        x = np.arange(len(models))
        means = metric_info["means"]
        stds = metric_info["stds"]

        # Plot bars with error bars
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(f"{metric_info['label']} (N={num_runs} runs)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("-")[0][:12] for m in models], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:
                label = f"{mean:.2f}"
                if std > 0:
                    label += f"±{std:.2f}"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                       label, ha='center', va='bottom', fontsize=7, rotation=45)

    # Hide empty subplots
    for idx in range(len(metrics_with_data), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.suptitle(f"Aggregated Results ({num_runs} Runs)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "aggregated_bar_chart_with_std.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate dual-facet scoring figures (Fixed Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For Delta results
  python generate_facet_scores.py --results_dir ./results_delta/new

  # For UltraTool + TaskBench results  
  python generate_facet_scores.py --results_dir ./results
        """
    )
    parser.add_argument("--results_dir", type=Path, default=Path("./results"))
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (default: results_dir/figures)")
    # SPECIAL REQUEST: Add dynamic model support
    parser.add_argument("--add_models", nargs="+", default=[],
                        help="List of new model names to add (auto-register metadata and colors)")
    # Multi-run aggregated visualization
    parser.add_argument("--aggregated", type=Path, default=None,
                        help="Path to aggregated_all_models.json for multi-run bar charts with std")

    args = parser.parse_args()
    
    # -------------------------------------------------------------
    # Register new models dynamically if requested
    # -------------------------------------------------------------
    if args.add_models:
        print("\n" + "=" * 60)
        print(f"Registering {len(args.add_models)} new models...")
        print("=" * 60)
        
        # Consistent color palette for new models (using a distinct palette)
        # Avoid blues (Qwen), reds (Mistral), etc. if possible, or cycle through distinct ones
        new_model_colors = [
            "#00BCD4", # Cyan
            "#CDDC39", # Lime
            "#9C27B0", # Purple
            "#FF5722", # Deep Orange
            "#795548", # Brown
            "#607D8B", # Blue Grey
            "#E91E63", # Pink
            "#3F51B5", # Indigo
        ]
        
        for idx, model_name in enumerate(args.add_models):
            # 1. Register in MODEL_INFO
            if model_name not in MODEL_INFO:
                MODEL_INFO[model_name] = {
                    "family": "New", 
                    "size_b": 0, 
                    "display": model_name, 
                    "short": model_name, 
                    "order": 900 + idx
                }
                print(f"[INFO] Registered metadata for: {model_name}")
            else:
                print(f"[INFO] Model already in metadata: {model_name}")

            # 2. Register in MODEL_COLORS
            if model_name not in MODEL_COLORS:
                # Pick a color based on hash or index
                color = new_model_colors[idx % len(new_model_colors)]
                MODEL_COLORS[model_name] = color
                print(f"[INFO] Assigned color {color} to: {model_name}")

            # 3. Check for unified file existence (Sanity Check)
            # Try same dir as results, or parent dir (common structure)
            possible_paths = [
                args.results_dir / f"unified.{model_name}.jsonl",
                args.results_dir.parent / f"unified.{model_name}.jsonl",
                Path(f"taskbench150/unified.{model_name}.jsonl") # specific fallback
            ]
            
            found_unified = False
            for p in possible_paths:
                if p.exists():
                    print(f"[OK] Found unified file: {p}")
                    found_unified = True
                    break
            
            if not found_unified:
                print(f"[WARN] Could not find 'unified.{model_name}.jsonl' in common locations.")
                print(f"       Checked: {[str(p) for p in possible_paths]}")
                print(f"       Ensure the file exists for valid evaluation linkage.")
            else:
                # 4. Check for format and line count validation (SPECIAL REQUEST)
                # Find a reference file (e.g., Toolchestra or any existing model) to compare against
                reference_candidates = ["Toolchestra", "qwen2.5-14b", "qwen2.5-7b", "qwen2.5-3b"]
                ref_file = None
                for ref in reference_candidates:
                    # Skip self
                    if ref == model_name:
                        continue
                    
                    # Look for reference file
                    for p_ref in [args.results_dir / f"unified.{ref}.jsonl", args.results_dir.parent / f"unified.{ref}.jsonl", Path(f"taskbench150/unified.{ref}.jsonl")]:
                        if p_ref.exists():
                            ref_file = p_ref
                            break
                    if ref_file:
                        break
                
                # Perform validation
                target_file = None
                for p in possible_paths:
                    if p.exists():
                        target_file = p
                        break
                
                if target_file:
                    try:
                        # Count lines
                        with open(target_file) as f:
                            target_lines = sum(1 for _ in f)
                        
                        # Validate format (first line)
                        with open(target_file) as f:
                            first_line = f.readline()
                            try:
                                data = json.loads(first_line)
                                required_keys = ["gold", "meta"] # pred is optional or varied
                                missing_keys = [k for k in required_keys if k not in data]
                                if missing_keys:
                                    print(f"  [WARN] {model_name}: Invalid format! Missing keys in first line: {missing_keys}")
                                else:
                                    print(f"  [OK] {model_name}: Format format check passed.")
                            except json.JSONDecodeError:
                                print(f"  [WARN] {model_name}: Invalid format! First line is not valid JSON.")

                        # Compare with reference
                        if ref_file:
                            with open(ref_file) as f:
                                ref_lines = sum(1 for _ in f)
                            
                            if target_lines != ref_lines:
                                print(f"  [WARN] {model_name}: Line count mismatch! Found {target_lines}, expected {ref_lines} (from {ref_file.name}).")
                            else:
                                print(f"  [OK] {model_name}: Line count matches ({target_lines}).")
                                
                    except Exception as e:
                         print(f"  [WARN] Validation failed for {target_file}: {e}")
    
    # Default output directory
    if args.output_dir is None:
        args.output_dir = args.results_dir / "figures"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Loading results (Fixed Scoring Version)...")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    
    summaries = load_summaries(args.results_dir)
    print(f"\nFound {len(summaries)} models: {list(summaries.keys())}")
    
    if not summaries:
        print("[ERROR] No summary files found!")
        return
    
    # Auto-detect datasets from summaries
    available_datasets = get_available_datasets(summaries)
    print(f"Available datasets: {available_datasets}")
    
    datasets = available_datasets
    
    # Print summary table
    print_facet_summary(summaries, datasets)
    rows = build_metrics_table_rows(summaries, datasets)
    export_metrics_tables(rows, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    # Handle aggregated multi-run visualization if provided
    if args.aggregated:
        print("\n[INFO] Generating aggregated bar chart with std...")
        plot_aggregated_bar_chart(args.aggregated, args.output_dir)

    # Generate figures
    plot_per_dataset_facet_scores(summaries, args.output_dir, datasets)
    plot_detailed_metrics_bar(summaries, args.output_dir, datasets)
    plot_overall_bar_chart(summaries, args.output_dir)

    # Radar charts (with SSI and without SSI variants)
    plot_radar_with_ssi(summaries, args.output_dir, datasets)
    plot_radar_without_ssi(summaries, args.output_dir, datasets)
    plot_radar_combined(summaries, args.output_dir, datasets)
    plot_radar_3facet(summaries, args.output_dir, datasets)

    # Additional analysis plots
    plot_model_scaling(summaries, args.output_dir, datasets)
    plot_parse_error_impact(summaries, args.output_dir, datasets)

    # Final answer visualization (Facet C - GAIA)
    plot_final_answer_metrics(summaries, args.output_dir, datasets)

    # Combined GAIA level visualizations
    plot_gaia_combined_3facet(summaries, args.output_dir, datasets)
    plot_gaia_level_comparison(summaries, args.output_dir, datasets)
    
    # New Radar Charts (Specific Request)
    plot_gaia_5axis_radar(summaries, args.output_dir, datasets)
    plot_gaia_3facet_radar(summaries, args.output_dir, datasets)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
