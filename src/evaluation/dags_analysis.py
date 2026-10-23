#!/usr/bin/env python3
"""
Async DAG analysis utilities for GAIA structural-augmentation experiments.

This script stays evaluator-adjacent so we can study ordering sensitivity and
efficiency without perturbing the main evaluation pipeline.

Outputs when --output_dir is provided:
1. Dataset-only CSV/figures summarizing async structural augmentation.

Additional outputs when --prediction_path is provided:
1. A CSV with one row per sample comparing canonical vs parallel reference.
2. A CSV with one row per (sample, possible_ordering) and planning-aware scores.
3. Planning-aware distribution figure.
4. Ordering-sensitivity figure.
5. Critical-path efficiency figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import ASTEvaluationSystem
from .utils import load_jsonl, format_metric


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array or JSONL file."""
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    return list(load_jsonl(path))


def critical_path_length(plan_dag: Dict[str, Any]) -> int:
    """
    Compute the critical-path length of a DAG with unit node weights.

    Returns the number of nodes on the longest path. If the graph is cyclic or
    malformed, falls back to the number of nodes to avoid optimistic scores.
    """
    nodes = plan_dag.get("nodes") or []
    edges = plan_dag.get("edges") or []
    if not nodes:
        return 0

    node_ids = [str(node.get("node_id", f"n{i}")) for i, node in enumerate(nodes)]
    adjacency = {node_id: set() for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}

    def as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if value is None:
            return []
        return [str(value)]

    for edge in edges:
        sources = as_list(edge.get("source"))
        targets = as_list(edge.get("target"))
        for src in sources:
            if src not in adjacency:
                continue
            for tgt in targets:
                if tgt not in adjacency or tgt in adjacency[src]:
                    continue
                adjacency[src].add(tgt)
                indegree[tgt] += 1

    queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
    topo_order: List[str] = []
    best = {node_id: 1 for node_id in node_ids}

    while queue:
        node_id = queue.pop(0)
        topo_order.append(node_id)
        for nxt in adjacency[node_id]:
            best[nxt] = max(best[nxt], best[node_id] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(topo_order) != len(node_ids):
        return len(node_ids)
    return max(best.values(), default=0)


def build_reference_dag_from_ordering(async_rec: Dict[str, Any], ordering: List[int]) -> Dict[str, Any]:
    """
    Expand an executable-step ordering into a chain reference DAG over original nodes.

    `possible_orderings` is defined over executable steps. We map each original
    chain node back to its executable-step group via `step_mapping`, keep the
    original within-group order, then serialize groups in the requested order.
    """
    gold_dag = (async_rec.get("gold") or {}).get("plan_dag") or {}
    gold_nodes = sorted(gold_dag.get("nodes") or [], key=lambda n: n.get("step_index", 0))
    step_mapping = {int(k): int(v) for k, v in (async_rec.get("step_mapping") or {}).items()}

    grouped_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in gold_nodes:
        orig_idx = int(node.get("step_index", 0))
        exec_idx = step_mapping.get(orig_idx, orig_idx)
        grouped_nodes[exec_idx].append(node)

    linearized_nodes: List[Dict[str, Any]] = []
    seen_exec_ids = set()
    for exec_idx in ordering:
        exec_idx = int(exec_idx)
        seen_exec_ids.add(exec_idx)
        for node in grouped_nodes.get(exec_idx, []):
            linearized_nodes.append({
                "node_id": str(node.get("node_id", f"n{node.get('step_index', len(linearized_nodes))}")),
                "step_index": int(node.get("step_index", len(linearized_nodes))),
                "label": node.get("label", ""),
                "step_type": node.get("step_type", "thought"),
                "tool_id": node.get("tool_id"),
            })

    for exec_idx in sorted(grouped_nodes):
        if exec_idx in seen_exec_ids:
            continue
        for node in grouped_nodes[exec_idx]:
            linearized_nodes.append({
                "node_id": str(node.get("node_id", f"n{node.get('step_index', len(linearized_nodes))}")),
                "step_index": int(node.get("step_index", len(linearized_nodes))),
                "label": node.get("label", ""),
                "step_type": node.get("step_type", "thought"),
                "tool_id": node.get("tool_id"),
            })

    linear_edges = [
        {"source": linearized_nodes[i]["node_id"], "target": linearized_nodes[i + 1]["node_id"]}
        for i in range(max(0, len(linearized_nodes) - 1))
    ]
    return {"nodes": linearized_nodes, "edges": linear_edges}


def build_parallel_reference_dag(async_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a true parallel reference DAG over original gold nodes.

    Strategy:
    - Keep the original gold nodes and their step_index values unchanged.
    - Use step_mapping to group original chain nodes into executable-step groups.
    - Preserve within-group order as a local chain.
    - Use dependency_analysis.edges to connect executable groups.
      Each executable dependency maps to an edge from the last original node in the
      source group to the first original node in the target group.
    """
    gold_dag = (async_rec.get("gold") or {}).get("plan_dag") or {}
    gold_nodes = sorted(gold_dag.get("nodes") or [], key=lambda n: n.get("step_index", 0))
    if not gold_nodes:
        return {"nodes": [], "edges": []}

    step_mapping = {int(k): int(v) for k, v in (async_rec.get("step_mapping") or {}).items()}
    grouped_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    normalized_nodes: List[Dict[str, Any]] = []

    for node in gold_nodes:
        step_index = int(node.get("step_index", len(normalized_nodes)))
        exec_idx = step_mapping.get(step_index, step_index)
        normalized = {
            "node_id": str(node.get("node_id", f"n{step_index}")),
            "step_index": step_index,
            "label": node.get("label", ""),
            "step_type": node.get("step_type", "thought"),
            "tool_id": node.get("tool_id"),
        }
        normalized_nodes.append(normalized)
        grouped_nodes[exec_idx].append(normalized)

    edge_pairs = set()

    # Preserve within-group sequential order when multiple original chain steps
    # are merged into the same executable step.
    for group_nodes in grouped_nodes.values():
        group_nodes.sort(key=lambda n: n["step_index"])
        for prev, curr in zip(group_nodes, group_nodes[1:]):
            edge_pairs.add((prev["node_id"], curr["node_id"]))

    # Add inter-group dependencies from the async partial-order graph.
    dep_edges = (async_rec.get("dependency_analysis") or {}).get("edges") or []
    for edge in dep_edges:
        src_exec = int(edge.get("from"))
        tgt_exec = int(edge.get("to"))
        src_group = grouped_nodes.get(src_exec) or []
        tgt_group = grouped_nodes.get(tgt_exec) or []
        if not src_group or not tgt_group:
            continue
        src_group.sort(key=lambda n: n["step_index"])
        tgt_group.sort(key=lambda n: n["step_index"])
        edge_pairs.add((src_group[-1]["node_id"], tgt_group[0]["node_id"]))

    parallel_edges = [
        {"source": src, "target": tgt}
        for src, tgt in sorted(edge_pairs, key=lambda pair: (pair[0], pair[1]))
    ]
    return {"nodes": normalized_nodes, "edges": parallel_edges}


def summarize_async_records(records: List[Dict[str, Any]], top_k: int) -> None:
    """Print dataset-level statistics for the async-plan file."""
    total_orderings = [int(rec.get("total_orderings", 1) or 1) for rec in records]
    parallel_group_counts = [len(rec.get("parallel_groups") or []) for rec in records]
    max_parallelism = [
        max((len(group) for group in (rec.get("parallel_groups") or [])), default=0)
        for rec in records
    ]
    chain_lengths = [len((rec.get("gold") or {}).get("plan_dag", {}).get("nodes") or []) for rec in records]
    chain_over_async = [
        (chain_len / async_cp) for chain_len, async_cp in zip(chain_lengths, parallel_group_counts) if async_cp > 0
    ]

    print("=" * 72)
    print("ASYNC PLAN DATASET SUMMARY")
    print("=" * 72)
    print(f"Samples: {len(records)}")
    print(f"Mean total_orderings: {statistics.mean(total_orderings):.2f}")
    print(f"Median total_orderings: {statistics.median(total_orderings):.2f}")
    print(f"Mean optimal critical path: {statistics.mean(parallel_group_counts):.2f}")
    print(f"Median optimal critical path: {statistics.median(parallel_group_counts):.2f}")
    print(f"Mean max_parallelism: {statistics.mean(max_parallelism):.2f}")
    print(f"Mean chain/async critical-path ratio: {statistics.mean(chain_over_async):.2f}")
    print(f"Max chain/async critical-path ratio: {max(chain_over_async):.2f}")

    print("\nTotal ordering histogram:")
    for ordering_count, freq in sorted(Counter(total_orderings).items()):
        print(f"  {ordering_count:>3}: {freq}")

    rows = []
    for rec, ratio in zip(records, chain_over_async):
        metadata = rec.get("metadata", {}) or {}
        rows.append({
            "id": rec.get("meta", {}).get("id", ""),
            "ratio": ratio,
            "total_orderings": int(rec.get("total_orderings", 1) or 1),
            "optimal_cp": len(rec.get("parallel_groups") or []),
            "chain_cp": len((rec.get("gold") or {}).get("plan_dag", {}).get("nodes") or []),
            "max_parallelism": int(metadata.get("max_parallelism", 0) or 0),
            "exec_steps": int(metadata.get("executable_step_count", len(rec.get("executable_steps") or [])) or 0),
        })
    rows.sort(key=lambda row: (-row["ratio"], -row["total_orderings"], row["id"]))

    print(f"\nTop {top_k} efficiency-gap samples (chain_cp / optimal_cp):")
    for row in rows[:top_k]:
        print(
            f"  {row['id']}: ratio={row['ratio']:.2f}, "
            f"chain_cp={row['chain_cp']}, optimal_cp={row['optimal_cp']}, "
            f"orderings={row['total_orderings']}, max_parallelism={row['max_parallelism']}, "
            f"exec_steps={row['exec_steps']}"
        )


def build_async_dataset_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build per-sample structural summary rows for async-plan visualization."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        metadata = rec.get("metadata", {}) or {}
        parallel_groups = rec.get("parallel_groups") or []
        optimal_cp = len(parallel_groups)
        chain_cp = len((rec.get("gold") or {}).get("plan_dag", {}).get("nodes") or [])
        ratio = (chain_cp / optimal_cp) if optimal_cp > 0 else 0.0
        rows.append(
            {
                "id": rec.get("meta", {}).get("id", ""),
                "total_orderings": int(rec.get("total_orderings", 1) or 1),
                "parallel_group_count": optimal_cp,
                "optimal_cp": optimal_cp,
                "chain_cp": chain_cp,
                "chain_async_ratio": ratio,
                "max_parallelism": int(metadata.get("max_parallelism", 0) or 0),
                "exec_steps": int(
                    metadata.get("executable_step_count", len(rec.get("executable_steps") or [])) or 0
                ),
                "original_steps": int(
                    metadata.get("original_step_count", len(rec.get("original_steps") or [])) or 0
                ),
                "redundant_steps": int(metadata.get("redundant_step_count", 0) or 0),
            }
        )
    return rows


def plot_async_dataset_distribution(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Visualize how many legal orderings and how much async slack each sample has."""
    if not rows:
        return

    total_orderings = [row["total_orderings"] for row in rows]
    optimal_cp = [row["optimal_cp"] for row in rows]
    max_parallelism = [row["max_parallelism"] for row in rows]
    ratios = [row["chain_async_ratio"] for row in rows]
    exec_steps = [row["exec_steps"] for row in rows]
    ordering_hist = Counter(total_orderings)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    hist_x = sorted(ordering_hist)
    hist_y = [ordering_hist[x] for x in hist_x]
    axes[0, 0].bar(hist_x, hist_y, color="#4E79A7", alpha=0.85)
    axes[0, 0].set_title("Legal Ordering Count per Sample")
    axes[0, 0].set_xlabel("total_orderings")
    axes[0, 0].set_ylabel("samples")
    axes[0, 0].set_xscale("symlog", linthresh=1)
    axes[0, 0].grid(alpha=0.25, axis="y")

    axes[0, 1].hist(optimal_cp, bins=12, color="#59A14F", alpha=0.8, edgecolor="white")
    axes[0, 1].set_title("Optimal Critical Path Distribution")
    axes[0, 1].set_xlabel("optimal_cp (parallel group count)")
    axes[0, 1].set_ylabel("samples")
    axes[0, 1].grid(alpha=0.25, axis="y")

    axes[1, 0].hist(max_parallelism, bins=range(1, max(max_parallelism) + 2), color="#F28E2B", alpha=0.8, edgecolor="white", align="left")
    axes[1, 0].set_title("Maximum Parallelism Distribution")
    axes[1, 0].set_xlabel("max_parallelism")
    axes[1, 0].set_ylabel("samples")
    axes[1, 0].grid(alpha=0.25, axis="y")

    scatter = axes[1, 1].scatter(
        total_orderings,
        ratios,
        c=max_parallelism,
        s=[30 + 4 * step for step in exec_steps],
        cmap="viridis",
        alpha=0.8,
        edgecolors="none",
    )
    axes[1, 1].set_title("Ordering Count vs Chain/Async Gap")
    axes[1, 1].set_xlabel("total_orderings")
    axes[1, 1].set_ylabel("chain_cp / optimal_cp")
    axes[1, 1].set_xscale("symlog", linthresh=1)
    axes[1, 1].grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=axes[1, 1])
    colorbar.set_label("max_parallelism")

    fig.suptitle("GAIA Cat A Async Dataset: Ordering and Parallelism Structure", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_async_gap_ranking(rows: List[Dict[str, Any]], output_path: Path, top_k: int) -> None:
    """Plot the samples with the largest chain-vs-async efficiency gap."""
    if not rows:
        return

    top_rows = sorted(
        rows,
        key=lambda row: (-row["chain_async_ratio"], -row["total_orderings"], row["id"]),
    )[:top_k]

    labels = [row["id"][:8] for row in top_rows]
    ratios = [row["chain_async_ratio"] for row in top_rows]
    orderings = [row["total_orderings"] for row in top_rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, ratios, color="#E15759", alpha=0.85)
    ax.set_title(f"Top {top_k} Samples with Largest Chain/Async Gap")
    ax.set_xlabel("sample id (prefix)")
    ax.set_ylabel("chain_cp / optimal_cp")
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(axis="x", rotation=35)

    for bar, ordering in zip(bars, orderings):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"ord={ordering}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write a list of dict rows to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_planning_distribution(
    sample_rows: List[Dict[str, Any]],
    native_sample_rows: List[Dict[str, Any]],
    high_fidelity_sample_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Compare canonical chain GT vs true parallel reference over planning-aware metrics."""
    metric_pairs = [
        ("canonical_node_f1", "parallel_node_f1", "Node F1"),
        ("canonical_edge_f1", "parallel_edge_f1", "Edge F1"),
        ("canonical_node_label_similarity", "parallel_node_label_similarity", "Node Label Sim"),
        ("canonical_ssi", "parallel_ssi", "SSI"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
    panels = [
        (axes[0], sample_rows, f"All Samples (n={len(sample_rows)})"),
        (
            axes[1],
            native_sample_rows,
            (
                "Native Refined DAG Only\n"
                f"(used_stage3_fallback=false; n={len(native_sample_rows)})"
            ),
        ),
        (
            axes[2],
            high_fidelity_sample_rows,
            (
                "High-Fidelity Async Candidates\n"
                "(EM=1, pred_cp>0, total_orderings>1,\n"
                f"used_stage3_fallback=false, node_f1>=0.4; n={len(high_fidelity_sample_rows)})"
            ),
        ),
    ]

    for ax, rows, title in panels:
        positions = []
        data = []
        tick_positions = []
        tick_labels = []
        for i, (canon_key, parallel_key, label) in enumerate(metric_pairs):
            base = i * 3 + 1
            positions.extend([base, base + 0.9])
            tick_positions.append(base + 0.45)
            tick_labels.append(label)
            data.extend([
                [row[canon_key] for row in rows] if rows else [],
                [row[parallel_key] for row in rows] if rows else [],
            ])

        bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#4E79A7" if idx % 2 == 0 else "#E15759")
            patch.set_alpha(0.75)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")

    axes[0].set_ylabel("Score")
    axes[0].plot([], [], color="#4E79A7", linewidth=8, alpha=0.75, label="Canonical chain GT")
    axes[0].plot([], [], color="#E15759", linewidth=8, alpha=0.75, label="Parallel-DAG reference")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Canonical Chain GT vs True Parallel-DAG Reference", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ordering_sensitivity(
    sample_rows: List[Dict[str, Any]],
    native_sample_rows: List[Dict[str, Any]],
    high_fidelity_sample_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Scatter plots for parallel-reference gain versus ordering-space size."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    axes[0].scatter(
        [row["total_orderings"] for row in sample_rows],
        [row["parallel_edge_gain"] for row in sample_rows],
        alpha=0.45,
        color="#9EA3A8",
        edgecolors="black",
        linewidths=0.4,
        label="All samples",
    )
    if native_sample_rows:
        axes[0].scatter(
            [row["total_orderings"] for row in native_sample_rows],
            [row["parallel_edge_gain"] for row in native_sample_rows],
            alpha=0.6,
            color="#4E79A7",
            edgecolors="black",
            linewidths=0.5,
            label="Native refined DAGs",
        )
    if high_fidelity_sample_rows:
        axes[0].scatter(
            [row["total_orderings"] for row in high_fidelity_sample_rows],
            [row["parallel_edge_gain"] for row in high_fidelity_sample_rows],
            alpha=0.95,
            color="#E15759",
            edgecolors="black",
            linewidths=0.7,
            label="High-fidelity async candidates",
        )
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_title("Parallel Reference Gain: Edge F1")
    axes[0].set_xlabel("total_orderings")
    axes[0].set_ylabel("parallel - canonical")

    axes[1].scatter(
        [row["total_orderings"] for row in sample_rows],
        [row["parallel_ssi_gain"] for row in sample_rows],
        alpha=0.45,
        color="#9EA3A8",
        edgecolors="black",
        linewidths=0.4,
        label="All samples",
    )
    if native_sample_rows:
        axes[1].scatter(
            [row["total_orderings"] for row in native_sample_rows],
            [row["parallel_ssi_gain"] for row in native_sample_rows],
            alpha=0.6,
            color="#4E79A7",
            edgecolors="black",
            linewidths=0.5,
            label="Native refined DAGs",
        )
    if high_fidelity_sample_rows:
        axes[1].scatter(
            [row["total_orderings"] for row in high_fidelity_sample_rows],
            [row["parallel_ssi_gain"] for row in high_fidelity_sample_rows],
            alpha=0.95,
            color="#E15759",
            edgecolors="black",
            linewidths=0.7,
            label="High-fidelity async candidates",
        )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_title("Parallel Reference Gain: SSI")
    axes[1].set_xlabel("total_orderings")
    axes[1].set_ylabel("parallel - canonical")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Parallel-DAG Reference Gain with Native/High-Fidelity Views",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_efficiency(
    sample_rows: List[Dict[str, Any]],
    native_sample_rows: List[Dict[str, Any]],
    high_fidelity_sample_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Two-panel view of critical-path efficiency."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(
        [row["total_orderings"] for row in sample_rows],
        [row["critical_path_efficiency"] for row in sample_rows],
        alpha=0.45,
        color="#9EA3A8",
        edgecolors="black",
        linewidths=0.4,
        label="All samples",
    )
    if native_sample_rows:
        axes[0].scatter(
            [row["total_orderings"] for row in native_sample_rows],
            [row["critical_path_efficiency"] for row in native_sample_rows],
            alpha=0.6,
            color="#4E79A7",
            edgecolors="black",
            linewidths=0.5,
            label="Native refined DAGs",
        )
    if high_fidelity_sample_rows:
        axes[0].scatter(
            [row["total_orderings"] for row in high_fidelity_sample_rows],
            [row["critical_path_efficiency"] for row in high_fidelity_sample_rows],
            alpha=0.95,
            color="#E15759",
            edgecolors="black",
            linewidths=0.7,
            label="High-fidelity async candidates",
        )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("total_orderings")
    axes[0].set_ylabel("Critical-Path Efficiency")
    axes[0].set_title("Efficiency vs Ordering Space")
    axes[0].grid(alpha=0.3)

    max_axis = max(
        max((row["pred_cp"] for row in sample_rows), default=1),
        max((row["optimal_cp"] for row in sample_rows), default=1),
    )
    axes[1].scatter(
        [row["optimal_cp"] for row in sample_rows],
        [row["pred_cp"] for row in sample_rows],
        alpha=0.45,
        color="#9EA3A8",
        edgecolors="black",
        linewidths=0.4,
        label="All samples",
    )
    if native_sample_rows:
        axes[1].scatter(
            [row["optimal_cp"] for row in native_sample_rows],
            [row["pred_cp"] for row in native_sample_rows],
            alpha=0.6,
            color="#4E79A7",
            edgecolors="black",
            linewidths=0.5,
            label="Native refined DAGs",
        )
    if high_fidelity_sample_rows:
        axes[1].scatter(
            [row["optimal_cp"] for row in high_fidelity_sample_rows],
            [row["pred_cp"] for row in high_fidelity_sample_rows],
            alpha=0.95,
            color="#E15759",
            edgecolors="black",
            linewidths=0.7,
            label="High-fidelity async candidates",
        )
    axes[1].plot([0, max_axis], [0, max_axis], "--", color="gray", linewidth=1.2)
    axes[1].set_xlim(0, max_axis + 1)
    axes[1].set_ylim(0, max_axis + 1)
    axes[1].set_xlabel("Optimal Critical Path")
    axes[1].set_ylabel("Predicted Critical Path")
    axes[1].set_title("Predicted vs Optimal Critical Path")
    axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Critical-Path Efficiency with Native/High-Fidelity Views",
        fontsize=13,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def analyze_predictions(
    async_records: List[Dict[str, Any]],
    prediction_path: Path,
    output_dir: Path,
    top_k: int,
) -> None:
    """
    Compare one prediction file against async-plan metadata.

    Design choice:
    - Run one planner prediction per question.
    - Compare that fixed predicted DAG against all legal possible_orderings.
    - Report both ordering sensitivity and structural efficiency.
    """
    async_by_id = {rec.get("meta", {}).get("id", ""): rec for rec in async_records}
    evaluator = ASTEvaluationSystem(use_embeddings=False)

    ordering_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for rec in load_jsonl(prediction_path):
        meta = dict(rec.get("meta", {}) or {})
        sample_id = str(meta.get("id", ""))
        async_rec = async_by_id.get(sample_id)
        if async_rec is None:
            continue

        query = rec.get("query", {}) or {}
        attachments = query.get("attachments", []) or []
        meta["has_image_attachment"] = any(
            str(att.get("file_name", "")).lower().endswith(IMAGE_EXTENSIONS)
            for att in attachments
        )
        if "supports_native_vision" not in meta:
            model_name_norm = str(meta.get("model_name", "")).lower()
            meta["supports_native_vision"] = bool(
                model_name_norm and (
                    "vision" in model_name_norm
                    or "multimodal" in model_name_norm
                    or "phi-4" in model_name_norm
                )
            )
        tool_env = rec.get("tool_environment", {})
        meta["available_tools"] = tool_env if isinstance(tool_env, list) else tool_env.get("tools", [])

        pred = rec.get("pred") or rec.get("prediction") or {}
        pred_trace = (pred.get("_trace") or {}) if isinstance(pred, dict) else {}
        pred_plan = (pred or {}).get("plan_dag") or {}
        used_stage3_fallback = pred_trace.get("used_stage3_fallback")
        base_scores = evaluator.evaluate_record(async_rec.get("gold") or {}, pred, meta)
        parallel_gold = {
            "plan_dag": build_parallel_reference_dag(async_rec),
            "tool_calls": (async_rec.get("gold") or {}).get("tool_calls") or [],
            "final_answer": (async_rec.get("gold") or {}).get("final_answer") or {},
        }
        parallel_scores = evaluator.evaluate_record(parallel_gold, pred, meta)

        pred_cp = critical_path_length(pred_plan)
        optimal_cp = len(async_rec.get("parallel_groups") or [])
        chain_cp = len((async_rec.get("gold") or {}).get("plan_dag", {}).get("nodes") or [])
        total_orderings = int(async_rec.get("total_orderings", 1) or 1)

        cpe_raw = (optimal_cp / pred_cp) if pred_cp > 0 else 0.0
        cpe = min(1.0, cpe_raw)
        cpe_interpretable = (used_stage3_fallback is False) and pred_cp > 0 and base_scores["plan"]["node_f1"] >= 0.4

        legal_orderings = async_rec.get("possible_orderings") or []
        if not legal_orderings:
            legal_orderings = [list(range(len(async_rec.get("executable_steps") or [])))]

        metric_traces = {
            "node_f1": [],
            "node_label_similarity": [],
            "edge_f1": [],
            "ssi": [],
        }

        for ordering_index, ordering in enumerate(legal_orderings):
            ordering_gold = {
                "plan_dag": build_reference_dag_from_ordering(async_rec, ordering),
                "tool_calls": (async_rec.get("gold") or {}).get("tool_calls") or [],
                "final_answer": (async_rec.get("gold") or {}).get("final_answer") or {},
            }
            ordering_scores = evaluator.evaluate_record(ordering_gold, pred, meta)
            plan_scores = ordering_scores["plan"]

            for metric_name in metric_traces:
                metric_traces[metric_name].append(plan_scores[metric_name])

            ordering_rows.append({
                "id": sample_id,
                "subset": meta.get("subset", "unknown"),
                "ordering_index": ordering_index,
                "total_orderings": total_orderings,
                "ordering_sequence": json.dumps(ordering, ensure_ascii=False),
                "node_f1": plan_scores["node_f1"],
                "node_label_similarity": plan_scores["node_label_similarity"],
                "edge_f1": plan_scores["edge_f1"],
                "ssi": plan_scores["ssi"],
                "critical_path_efficiency": cpe,
                "critical_path_efficiency_raw": cpe_raw,
                "optimal_cp": optimal_cp,
                "pred_cp": pred_cp,
                "chain_cp": chain_cp,
                "used_stage3_fallback": used_stage3_fallback,
                "cpe_interpretable": cpe_interpretable,
                "canonical_node_f1": base_scores["plan"]["node_f1"],
                "canonical_edge_f1": base_scores["plan"]["edge_f1"],
                "canonical_ssi": base_scores["plan"]["ssi"],
                "exact_match": base_scores["answer"]["exact_match"],
            })

        sample_rows.append({
            "id": sample_id,
            "subset": meta.get("subset", "unknown"),
            "total_orderings": total_orderings,
            "critical_path_efficiency": cpe,
            "critical_path_efficiency_raw": cpe_raw,
            "optimal_cp": optimal_cp,
            "pred_cp": pred_cp,
            "chain_cp": chain_cp,
            "used_stage3_fallback": used_stage3_fallback,
            "cpe_interpretable": cpe_interpretable,
            "canonical_edge_f1": base_scores["plan"]["edge_f1"],
            "canonical_ssi": base_scores["plan"]["ssi"],
            "canonical_node_f1": base_scores["plan"]["node_f1"],
            "canonical_node_label_similarity": base_scores["plan"]["node_label_similarity"],
            "parallel_edge_f1": parallel_scores["plan"]["edge_f1"],
            "parallel_ssi": parallel_scores["plan"]["ssi"],
            "parallel_node_f1": parallel_scores["plan"]["node_f1"],
            "parallel_node_label_similarity": parallel_scores["plan"]["node_label_similarity"],
            "parallel_edge_gain": parallel_scores["plan"]["edge_f1"] - base_scores["plan"]["edge_f1"],
            "parallel_ssi_gain": parallel_scores["plan"]["ssi"] - base_scores["plan"]["ssi"],
            "edge_f1_mean": statistics.mean(metric_traces["edge_f1"]),
            "edge_f1_std": statistics.pstdev(metric_traces["edge_f1"]) if len(metric_traces["edge_f1"]) > 1 else 0.0,
            "edge_f1_min": min(metric_traces["edge_f1"]),
            "edge_f1_max": max(metric_traces["edge_f1"]),
            "ssi_mean": statistics.mean(metric_traces["ssi"]),
            "ssi_std": statistics.pstdev(metric_traces["ssi"]) if len(metric_traces["ssi"]) > 1 else 0.0,
            "ssi_min": min(metric_traces["ssi"]),
            "ssi_max": max(metric_traces["ssi"]),
            "node_f1_mean": statistics.mean(metric_traces["node_f1"]),
            "node_label_similarity_mean": statistics.mean(metric_traces["node_label_similarity"]),
            "exact_match": base_scores["answer"]["exact_match"],
        })

    if not ordering_rows:
        print(f"\nNo overlapping prediction records found in {prediction_path}")
        return

    native_ordering_rows = [
        row for row in ordering_rows
        if row["used_stage3_fallback"] is False
    ]
    native_sample_rows = [
        row for row in sample_rows
        if row["used_stage3_fallback"] is False
    ]

    high_fidelity_ordering_rows = [
        row for row in ordering_rows
        if (
            row["exact_match"] == 1.0
            and row["pred_cp"] > 0
            and row["total_orderings"] > 1
            and row["used_stage3_fallback"] is False
            and row["canonical_node_f1"] >= 0.4
        )
    ]
    high_fidelity_sample_rows = [
        row for row in sample_rows
        if (
            row["exact_match"] == 1.0
            and row["pred_cp"] > 0
            and row["total_orderings"] > 1
            and row["used_stage3_fallback"] is False
            and row["canonical_node_f1"] >= 0.4
        )
    ]

    csv_path = output_dir / "orderings_planning_distribution.csv"
    parallel_csv_path = output_dir / "parallel_reference_comparison.csv"
    write_csv(ordering_rows, csv_path)
    write_csv(sample_rows, parallel_csv_path)
    plot_planning_distribution(
        sample_rows,
        native_sample_rows,
        high_fidelity_sample_rows,
        output_dir / "planning_aware_distribution.png",
    )
    plot_ordering_sensitivity(
        sample_rows,
        native_sample_rows,
        high_fidelity_sample_rows,
        output_dir / "ordering_sensitivity.png",
    )
    plot_efficiency(
        sample_rows,
        native_sample_rows,
        high_fidelity_sample_rows,
        output_dir / "critical_path_efficiency.png",
    )

    print("\n" + "=" * 72)
    print("PREDICTION-LEVEL ASYNC ANALYSIS")
    print("=" * 72)
    print(f"Prediction file: {prediction_path}")
    print(f"Matched samples: {len(sample_rows)}")
    print(f"Ordering CSV: {csv_path}")
    print(f"Parallel comparison CSV: {parallel_csv_path}")
    print(f"Mean CPE: {statistics.mean(row['critical_path_efficiency'] for row in sample_rows):.4f}")
    print(f"Median CPE: {statistics.median(row['critical_path_efficiency'] for row in sample_rows):.4f}")
    print(f"Mean canonical Edge F1: {statistics.mean(row['canonical_edge_f1'] for row in sample_rows):.4f}")
    print(f"Mean parallel Edge F1: {statistics.mean(row['parallel_edge_f1'] for row in sample_rows):.4f}")
    print(f"Mean parallel Edge gain: {statistics.mean(row['parallel_edge_gain'] for row in sample_rows):.4f}")
    print(f"Mean canonical SSI: {statistics.mean(row['canonical_ssi'] for row in sample_rows):.4f}")
    print(f"Mean parallel SSI: {statistics.mean(row['parallel_ssi'] for row in sample_rows):.4f}")
    print(f"Mean parallel SSI gain: {statistics.mean(row['parallel_ssi_gain'] for row in sample_rows):.4f}")
    print(f"Mean EM: {statistics.mean((row['exact_match'] or 0.0) for row in sample_rows):.4f}")
    print(f"CPE-interpretable samples: {sum(1 for row in sample_rows if row['cpe_interpretable'])}")
    print(f"Native refined DAG ordering rows: {len(native_ordering_rows)}")
    print(f"High-fidelity ordering rows: {len(high_fidelity_ordering_rows)}")
    print(f"Native refined DAG samples: {len(native_sample_rows)}")
    print(f"High-fidelity async candidates: {len(high_fidelity_sample_rows)}")
    if native_sample_rows:
        print(f"Native-only mean CPE: {statistics.mean(row['critical_path_efficiency'] for row in native_sample_rows):.4f}")
        print(f"Native-only mean parallel Edge gain: {statistics.mean(row['parallel_edge_gain'] for row in native_sample_rows):.4f}")
    if high_fidelity_sample_rows:
        print(f"High-fidelity mean CPE: {statistics.mean(row['critical_path_efficiency'] for row in high_fidelity_sample_rows):.4f}")
        print(f"High-fidelity mean parallel Edge gain: {statistics.mean(row['parallel_edge_gain'] for row in high_fidelity_sample_rows):.4f}")

    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        buckets[row["total_orderings"]].append(row)

    print("\nOrdering sensitivity by total_orderings:")
    for ordering_count in sorted(buckets):
        bucket = buckets[ordering_count]
        print(
            f"  {ordering_count:>3}: n={len(bucket):>2}, "
            f"mean_edge_std={statistics.mean(r['edge_f1_std'] for r in bucket):.4f}, "
            f"mean_ssi_std={statistics.mean(r['ssi_std'] for r in bucket):.4f}, "
            f"mean_cpe={statistics.mean(r['critical_path_efficiency'] for r in bucket):.4f}"
        )

    by_cpe = sorted(
        sample_rows,
        key=lambda row: (row["critical_path_efficiency"], row["canonical_edge_f1"], row["id"]),
    )
    print(f"\nBottom {top_k} CPE samples:")
    for row in by_cpe[:top_k]:
        print(
            f"  {row['id']}: cpe={format_metric(row['critical_path_efficiency'])}, "
            f"pred_cp={row['pred_cp']}, optimal_cp={row['optimal_cp']}, "
            f"orderings={row['total_orderings']}, "
            f"edge_std={format_metric(row['edge_f1_std'])}, ssi_std={format_metric(row['ssi_std'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze async DAG augmentation, ordering sensitivity, and efficiency.")
    parser.add_argument(
        "--async_plan_path",
        type=Path,
        required=True,
        help="Path to async-augmented GAIA JSONL file.",
    )
    parser.add_argument(
        "--prediction_path",
        type=Path,
        default=None,
        help="Optional unified prediction JSONL to compare against async-plan metadata.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for CSV and figures. Defaults to prediction_path.parent/orderings_analysis.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top/bottom samples to print.",
    )
    args = parser.parse_args()

    async_records = load_json_or_jsonl(args.async_plan_path)
    summarize_async_records(async_records, top_k=args.top_k)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dataset_rows = build_async_dataset_rows(async_records)
        write_csv(dataset_rows, args.output_dir / "async_dataset_summary.csv")
        plot_async_dataset_distribution(
            dataset_rows,
            args.output_dir / "async_dataset_distribution.png",
        )
        plot_async_gap_ranking(
            dataset_rows,
            args.output_dir / "async_top_efficiency_gap.png",
            top_k=args.top_k,
        )
        print(f"\nSaved dataset-level async visualizations to: {args.output_dir}")

    if args.prediction_path is not None:
        output_dir = args.output_dir or (args.prediction_path.parent / "orderings_analysis")
        analyze_predictions(
            async_records,
            prediction_path=args.prediction_path,
            output_dir=output_dir,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
