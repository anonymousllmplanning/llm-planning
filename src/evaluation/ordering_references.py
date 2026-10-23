"""Validate recorded schedules and build task-specific chain references."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_ordering_schedule(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the frozen schedule format without overwriting a task's candidates."""
    with Path(path).open(encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Ordering schedule must be a JSON array or JSONL records")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("task_id"):
            raise ValueError(f"Schedule row {index} must contain task_id")
        kind, rank, order = row.get("kind"), row.get("candidate_rank"), row.get("ordering")
        if kind not in {"native", "candidate"}:
            raise ValueError(f"Schedule row {index} has invalid kind")
        if type(rank) is not int or rank < 0 or (kind == "native") != (rank == 0):
            raise ValueError(f"Schedule row {index} has invalid candidate_rank")
        if not isinstance(order, list) or any(type(step) is not int for step in order):
            raise ValueError(f"Schedule row {index} ordering must be a list of integer step indices")
        grouped[str(row["task_id"])].append(row)
    for task_id, refs in grouped.items():
        if sum(ref["kind"] == "native" for ref in refs) != 1:
            raise ValueError(f"Task {task_id} must have exactly one native ordering")
        if len({ref["candidate_rank"] for ref in refs}) != len(refs):
            raise ValueError(f"Task {task_id} has duplicate candidate ranks")
        if len({tuple(ref["ordering"]) for ref in refs}) != len(refs):
            raise ValueError(f"Task {task_id} has duplicate orderings")
        refs.sort(key=lambda ref: (ref["kind"] != "native", ref["candidate_rank"]))
    return dict(grouped)


def ordering_references(native_record: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse native nodes, replacing only adjacent edges for each recorded order.

    The compiled AD/SD DAG is a generation constraint, never a scoring target.
    Native comes first, followed by candidate rank, to make exact ties stable.
    """
    native = (native_record.get("gold") or {}).get("plan_dag") or {}
    nodes = native.get("nodes") or []
    task_id = str((native_record.get("meta") or {}).get("id") or "unknown")
    steps = {node.get("step_index"): index for index, node in enumerate(nodes)}
    if len(steps) != len(nodes) or any(type(step) is not int for step in steps) or sorted(steps) != list(range(len(nodes))):
        raise ValueError(f"Task {task_id}: native step indices must be a unique contiguous zero-based range")
    node_ids = {node.get("node_id") for node in nodes}
    if len(node_ids) != len(nodes) or None in node_ids or "" in node_ids:
        raise ValueError(f"Task {task_id}: native node IDs must be present and unique")
    if not refs or sum(ref.get("kind") == "native" for ref in refs) != 1:
        raise ValueError(f"Task {task_id}: schedule must include exactly one native ordering")
    native_pairs = set()
    for edge in native.get("edges") or []:
        sources = edge.get("source") if isinstance(edge.get("source"), list) else [edge.get("source")]
        targets = edge.get("target") if isinstance(edge.get("target"), list) else [edge.get("target")]
        for source in sources:
            for target in targets:
                if source not in node_ids or target not in node_ids:
                    raise ValueError(f"Task {task_id}: native edge has an unknown endpoint")
                native_pairs.add((source, target))
    seen_orders, seen_ranks, result = set(), set(), []
    for ref in sorted(refs, key=lambda row: (row.get("kind") != "native", row.get("candidate_rank", -1))):
        order, rank, kind = ref.get("ordering"), ref.get("candidate_rank"), ref.get("kind")
        if kind not in {"native", "candidate"} or type(rank) is not int or rank < 0 or (kind == "native") != (rank == 0):
            raise ValueError(f"Task {task_id}: invalid reference kind or rank")
        if not isinstance(order, list) or any(type(step) is not int for step in order) or sorted(order) != sorted(steps):
            raise ValueError(f"Task {task_id}: ordering must contain every native step exactly once")
        if tuple(order) in seen_orders or rank in seen_ranks:
            raise ValueError(f"Task {task_id}: duplicate ordering or candidate rank")
        seen_orders.add(tuple(order))
        seen_ranks.add(rank)
        pairs = {(steps[a], steps[b]) for a, b in zip(order, order[1:])}
        edges = [{"source": nodes[a]["node_id"], "target": nodes[b]["node_id"], "edge_type": "control_dep"}
                 for a, b in sorted(pairs)]
        if kind == "native" and (order != list(range(len(nodes))) or {(e["source"], e["target"]) for e in edges} != native_pairs):
            raise ValueError(f"Task {task_id}: recorded native ordering disagrees with the native graph")
        positions = {step: index for index, step in enumerate(order)}
        for edge in (native_record.get("ad_sd_v6") or {}).get("dag_edges") or []:
            source, target = edge.get("from"), edge.get("to")
            if source not in positions or target not in positions or positions[source] >= positions[target]:
                raise ValueError(f"Task {task_id}: ordering violates an AD/SD generation constraint")
        result.append({"name": "chain" if kind == "native" else f"candidate_{rank}",
                       "dag": dict(native, nodes=nodes, edges=edges)})
    return result
