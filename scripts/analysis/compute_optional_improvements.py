#!/usr/bin/env python3
"""Compute optional appendix diagnostics from paper result artifacts.

The outputs intentionally avoid raw GAIA questions, answers, or attachments.
They use task ids, model names, retained-ordering counts, and metric columns
already used by the paper tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "version2" / "paper_results" / "optional_improvements"

FINAL_GT_DIR = ROOT / "data" / "Augmented" / "Augmented_GT_NativePlusGemma4NonNative"
DAG_DIR = FINAL_GT_DIR if FINAL_GT_DIR.exists() else ROOT / "data" / "Augmented" / "DAGs"
OPEN_WEIGHT_PER_SAMPLE = (
    ROOT
    / "version2"
    / "paper_results"
    / "official_modified_gt_20260504"
    / "gaia_per_sample_official_modified_gt.csv"
)
OPENAI_PER_SAMPLE = (
    ROOT
    / "version2"
    / "paper_results"
    / "openai_analysis"
    / "openai_combined_per_sample_stage3fb.csv"
)
GEMINI_PER_SAMPLE = (
    ROOT
    / "version2"
    / "paper_results"
    / "gemini_analysis"
    / "gemini_combined_per_sample.csv"
)

DOMAIN = {"A": "Text", "B": "Document", "C": "Vision", "D": "Audio"}
OPEN_MODEL_ORDER = [
    "Mistral Large 3",
    "Llama-3.1",
    "Llama-3.3",
    "Gemma 4",
    "Mistral Small 3.2",
    "Llama-4 Maverick",
    "Gemma 3",
]
OPENAI_MODEL_ORDER = ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]
GEMINI_MODEL_ORDER = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
MODEL_DISPLAY = {
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4-Mini",
    "gpt-5.4-nano": "GPT-5.4-Nano",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite",
}


@dataclass
class DagProfile:
    sample_id: str
    cat: str
    domain: str
    level: int | None
    native_chain_length: int
    critical_path_length: int
    parallelism_ratio: float
    candidate_orderings: int
    retained_orderings: int
    filtered_orderings: int
    bucket: str
    max_parallelism: int | None
    retained_ordering_distance_mean: float
    retained_ordering_distance_median: float
    retained_ordering_distance_max: float


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_level(subset: Any) -> int | None:
    text = str(subset or "")
    parts = text.split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def retained_bucket(n: int) -> str:
    if n <= 0:
        return "native-only"
    if n == 1:
        return "exactly-one"
    return "multi-order-rich"


def critical_path_length(n: int, dependencies: list[dict[str, Any]]) -> int:
    parents: dict[int, list[int]] = {i: [] for i in range(n)}
    for dep in dependencies:
        try:
            sid = int(dep.get("step_id"))
        except Exception:
            continue
        if sid < 0 or sid >= n:
            continue
        ps = []
        for parent in dep.get("parents") or []:
            try:
                pid = int(parent.get("parent_id"))
            except Exception:
                continue
            if 0 <= pid < n:
                ps.append(pid)
        parents[sid] = ps

    memo: dict[int, int] = {}
    visiting: set[int] = set()

    def depth(node: int) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            return n
        visiting.add(node)
        if not parents.get(node):
            ans = 1
        else:
            ans = 1 + max(depth(parent) for parent in parents[node])
        visiting.remove(node)
        memo[node] = ans
        return ans

    return max((depth(i) for i in range(n)), default=0)


def normalized_kendall_distance(ordering: list[Any], n: int) -> float | None:
    try:
        order = [int(x) for x in ordering]
    except Exception:
        return None
    if len(order) != n or sorted(order) != list(range(n)):
        return None
    denom = n * (n - 1) / 2
    if denom <= 0:
        return 0.0
    inv = 0
    for i in range(n):
        ai = order[i]
        for j in range(i + 1, n):
            if ai > order[j]:
                inv += 1
    return inv / denom


def is_native_ordering(ordering: list[Any], n: int) -> bool:
    try:
        order = [int(x) for x in ordering]
    except Exception:
        return False
    return order == list(range(n))


def valid_ordering(ordering: list[Any], n: int) -> bool:
    try:
        order = [int(x) for x in ordering]
    except Exception:
        return False
    return len(order) == n and sorted(order) == list(range(n))


def retained_non_native_orderings(obj: dict[str, Any], n: int) -> list[list[Any]]:
    return [
        ordering
        for ordering in (obj.get("sampled_orderings") or [])
        if valid_ordering(ordering, n) and not is_native_ordering(ordering, n)
    ]


def reference_set_critical_path_length(orderings: list[list[Any]], n: int) -> int:
    """Longest chain in the partial order induced by final reference orderings.

    A pair i -> j is kept as an ordering constraint only when i precedes j in
    every final reference ordering for the task. This is a conservative
    replay-validated critical-path diagnostic: native-only tasks collapse to the
    native chain, while retained non-native references can remove only the
    ordering constraints they actually reorder.
    """
    if n <= 0:
        return 0
    valid_orderings = [list(map(int, o)) for o in orderings if valid_ordering(o, n)]
    if not valid_orderings:
        valid_orderings = [list(range(n))]
    positions = [{step: idx for idx, step in enumerate(ordering)} for ordering in valid_orderings]
    parents: dict[int, list[int]] = {i: [] for i in range(n)}
    for before in range(n):
        for after in range(n):
            if before == after:
                continue
            if all(pos[before] < pos[after] for pos in positions):
                parents[after].append(before)

    memo: dict[int, int] = {}

    def depth(node: int) -> int:
        if node in memo:
            return memo[node]
        if not parents.get(node):
            ans = 1
        else:
            ans = 1 + max(depth(parent) for parent in parents[node])
        memo[node] = ans
        return ans

    return max((depth(i) for i in range(n)), default=0)


def load_task_profiles() -> pd.DataFrame:
    rows: list[DagProfile] = []
    for cat in ["A", "B", "C", "D"]:
        path = DAG_DIR / f"gaia_cat_{cat}_async_plan.jsonl"
        for obj in iter_jsonl(path):
            meta = obj.get("meta") or {}
            md = obj.get("metadata") or {}
            sample_id = str(meta.get("id"))
            n = int(md.get("original_step_count") or len(obj.get("original_steps") or []))
            retained_orderings = retained_non_native_orderings(obj, n)
            retained = len(retained_orderings)
            final_orderings = [list(range(n)), *retained_orderings]
            cpl = reference_set_critical_path_length(final_orderings, n)
            candidates = int(
                md.get("num_candidate_orderings_before_gemma4_filter")
                or md.get("num_sampled_orderings")
                or 0
            )
            filtered = int(md.get("num_gemma4_filtered_orderings") or 0)
            distances = [
                d
                for ordering in retained_orderings
                for d in [normalized_kendall_distance(ordering, n)]
                if d is not None
            ]
            rows.append(
                DagProfile(
                    sample_id=sample_id,
                    cat=cat,
                    domain=DOMAIN[cat],
                    level=parse_level(meta.get("subset")),
                    native_chain_length=n,
                    critical_path_length=cpl,
                    parallelism_ratio=(1.0 - cpl / n) if n else np.nan,
                    candidate_orderings=candidates,
                    retained_orderings=retained,
                    filtered_orderings=filtered,
                    bucket=retained_bucket(retained),
                    max_parallelism=md.get("max_parallelism"),
                    retained_ordering_distance_mean=float(np.mean(distances)) if distances else np.nan,
                    retained_ordering_distance_median=float(np.median(distances)) if distances else np.nan,
                    retained_ordering_distance_max=float(np.max(distances)) if distances else np.nan,
                )
            )
    return pd.DataFrame([r.__dict__ for r in rows])


def load_reference_rows() -> pd.DataFrame:
    """Expand the final Augmented GAIA GT into reference-ordering rows.

    The final GT contains one native row per task plus Gemma 4-retained
    non-native rows. We only treat task-level parallelism as validated when at
    least one non-native row survived replay filtering; native-only tasks keep
    their native reference but do not contribute unvalidated DAG parallelism.
    """
    rows: list[dict[str, Any]] = []
    for cat in ["A", "B", "C", "D"]:
        path = DAG_DIR / f"gaia_cat_{cat}_async_plan.jsonl"
        for obj in iter_jsonl(path):
            meta = obj.get("meta") or {}
            sample_id = str(meta.get("id"))
            n = int((obj.get("metadata") or {}).get("original_step_count") or len(obj.get("original_steps") or []))
            orderings = [o for o in (obj.get("sampled_orderings") or []) if valid_ordering(o, n)]
            non_native_count = sum(1 for o in orderings if not is_native_ordering(o, n))
            bucket = retained_bucket(non_native_count)
            validated_cpl = reference_set_critical_path_length(orderings, n)
            validated_parallelism = (1.0 - validated_cpl / n) if n else np.nan
            native_seen = False
            row_index = 0
            for ordering in orderings:
                native = is_native_ordering(ordering, n)
                if native:
                    if native_seen:
                        continue
                    native_seen = True
                    reference_type = "native"
                else:
                    reference_type = "retained-non-native"
                rows.append(
                    {
                        "task_id": sample_id,
                        "cat": cat,
                        "domain": DOMAIN[cat],
                        "level": parse_level(meta.get("subset")),
                        "coverage_bucket": bucket,
                        "row_index": row_index,
                        "reference_type": reference_type,
                        "native_chain_length": n,
                        "critical_path_length": validated_cpl,
                        "parallelism_ratio": validated_parallelism,
                        "kendall_distance_from_native": normalized_kendall_distance(ordering, n),
                        "retained_non_native_orderings_for_task": non_native_count,
                    }
                )
                row_index += 1
    return pd.DataFrame(rows)


def summarize_reference_thresholds(reference_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [1, 2, 5, 10, 20, 50]:
        df = reference_rows[
            (reference_rows["reference_type"] == "retained-non-native")
            & (reference_rows["retained_non_native_orderings_for_task"] >= threshold)
        ]
        rows.append(
            {
                "threshold": f">={threshold}",
                "is_multi_order_rich_threshold": threshold == 2,
                "tasks": int(df["task_id"].nunique()),
                "retained_rows": int(len(df)),
                "parallelism_mean": df["parallelism_ratio"].mean(),
                "parallelism_median": df["parallelism_ratio"].median(),
                "kendall_mean": df["kendall_distance_from_native"].mean(),
                "kendall_median": df["kendall_distance_from_native"].median(),
                "kendall_max": df["kendall_distance_from_native"].max(),
            }
        )
    return pd.DataFrame(rows)


def summarize_reference_slice(
    section: str, slice_name: str, df: pd.DataFrame
) -> dict[str, Any]:
    return {
        "section": section,
        "slice": slice_name,
        "rows": int(len(df)),
        "native_rows": int((df["reference_type"] == "native").sum()),
        "non_native_rows": int((df["reference_type"] == "retained-non-native").sum()),
        "native_len_mean": df["native_chain_length"].mean(),
        "native_len_median": df["native_chain_length"].median(),
        "critical_path_mean": df["critical_path_length"].mean(),
        "critical_path_median": df["critical_path_length"].median(),
        "parallelism_mean": df["parallelism_ratio"].mean(),
        "parallelism_median": df["parallelism_ratio"].median(),
        "kendall_mean": df["kendall_distance_from_native"].mean(),
        "kendall_median": df["kendall_distance_from_native"].median(),
        "kendall_max": df["kendall_distance_from_native"].max(),
    }


def summarize_reference_slices(reference_rows: pd.DataFrame) -> pd.DataFrame:
    rows = [
        summarize_reference_slice("Reference type", "All final references", reference_rows),
        summarize_reference_slice(
            "Reference type",
            "Retained non-native references",
            reference_rows[reference_rows["reference_type"] == "retained-non-native"],
        ),
    ]
    for bucket in ["native-only", "exactly-one", "multi-order-rich"]:
        rows.append(
            summarize_reference_slice(
                "Coverage bucket",
                bucket,
                reference_rows[reference_rows["coverage_bucket"] == bucket],
            )
        )
    for level in sorted(reference_rows["level"].dropna().unique()):
        rows.append(
            summarize_reference_slice(
                "GAIA level",
                f"Level {int(level)}",
                reference_rows[reference_rows["level"] == level],
            )
        )
    for domain in ["Text", "Document", "Vision", "Audio"]:
        rows.append(
            summarize_reference_slice(
                "Domain",
                domain,
                reference_rows[reference_rows["domain"] == domain],
            )
        )
    return pd.DataFrame(rows)


def summarize_parallelism(tasks: pd.DataFrame) -> pd.DataFrame:
    order = ["all", "native-only", "exactly-one", "multi-order-rich"]
    rows = []
    for bucket in order:
        df = tasks if bucket == "all" else tasks[tasks["bucket"] == bucket]
        if df.empty:
            continue
        rows.append(
            {
                "bucket": bucket,
                "n_tasks": len(df),
                "native_chain_mean": df["native_chain_length"].mean(),
                "native_chain_median": df["native_chain_length"].median(),
                "native_chain_min": df["native_chain_length"].min(),
                "native_chain_max": df["native_chain_length"].max(),
                "critical_path_mean": df["critical_path_length"].mean(),
                "critical_path_median": df["critical_path_length"].median(),
                "critical_path_min": df["critical_path_length"].min(),
                "critical_path_max": df["critical_path_length"].max(),
                "parallelism_ratio_mean": df["parallelism_ratio"].mean(),
                "parallelism_ratio_median": df["parallelism_ratio"].median(),
                "parallelism_ratio_min": df["parallelism_ratio"].min(),
                "parallelism_ratio_max": df["parallelism_ratio"].max(),
                "retained_orderings_mean": df["retained_orderings"].mean(),
                "candidate_orderings_mean": df["candidate_orderings"].mean(),
                "retained_ordering_distance_mean": safe_series_mean(df["retained_ordering_distance_mean"]),
                "retained_ordering_distance_median": safe_series_median(df["retained_ordering_distance_median"]),
                "retained_ordering_distance_max": safe_series_max(df["retained_ordering_distance_max"]),
            }
        )
    return pd.DataFrame(rows)


def safe_series_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_series_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def safe_series_max(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.max()) if len(s) else np.nan


def normalize_open_weight(open_df: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    df = open_df.copy()
    df = df.rename(columns={"task_id": "sample_id", "Model": "model"})
    df["pool"] = "open-weight"
    df["family"] = "open-weight"
    df["display_model"] = df["model"]
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(OPEN_MODEL_ORDER)}).fillna(999)
    df = df.merge(
        tasks[["sample_id", "level", "retained_orderings", "bucket"]],
        on="sample_id",
        how="left",
        suffixes=("", "_task"),
    )
    if "level_task" in df.columns:
        df["level"] = df["level_task"]
        df = df.drop(columns=["level_task"])
    df["native_edge_proxy"] = 2.0 * df["chain_only_planning_score"] - np.nan
    df["edge_delta"] = 2.0 * (
        df["augmented_best_planning_score"] - df["chain_only_planning_score"]
    )
    return df


def normalize_closed(path: Path, family: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pool"] = "closed-model"
    df["family"] = family
    df["display_model"] = df["model"].map(MODEL_DISPLAY).fillna(df["model"])
    if family == "openai":
        df["model_order"] = df["model"].map({m: i for i, m in enumerate(OPENAI_MODEL_ORDER)})
    else:
        df["model_order"] = df["model"].map({m: i for i, m in enumerate(GEMINI_MODEL_ORDER)})
    df["edge_delta"] = (
        pd.to_numeric(df["augmented_best_semantic_edge_f1"], errors="coerce")
        - pd.to_numeric(df["chain_only_semantic_edge_f1"], errors="coerce")
    )
    return df


def best_match_breakdown(pairs: pd.DataFrame, eps: float = 1e-12) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pairs.copy()
    df["augmented_helped"] = df["edge_delta"] > eps
    df["native_or_tie"] = ~df["augmented_helped"]
    rich = df[df["bucket"].eq("multi-order-rich")].copy()

    rows = []
    for pool, g in rich.groupby("pool", dropna=False):
        rows.append(make_breakdown_row(pool, "All", g))
        level2 = g[g["level"].astype(float).eq(2)]
        if not level2.empty:
            rows.append(make_breakdown_row(pool, "GAIA Level 2", level2))

    # Add combined pool over all 15 models.
    for subset in ["All", "GAIA Level 2"]:
        h = rich if subset == "All" else rich[rich["level"].astype(float).eq(2)]
        rows.append(make_breakdown_row("all-models", subset, h))

    summary = pd.DataFrame(rows)
    summary["subset_order"] = summary["subset"].map({"All": 0, "GAIA Level 2": 1})
    summary["pool_order"] = summary["pool"].map(
        {"open-weight": 0, "closed-model": 1, "all-models": 2}
    )
    summary = summary.sort_values(["subset_order", "pool_order"]).drop(
        columns=["subset_order", "pool_order"]
    )

    by_model = []
    for (pool, family, display_model, model_order), g in rich.groupby(
        ["pool", "family", "display_model", "model_order"], dropna=False
    ):
        by_model.append(make_breakdown_row(pool, "All", g) | {
            "family": family,
            "model": display_model,
            "model_order": model_order,
        })
    by_model_df = pd.DataFrame(by_model).sort_values(["pool", "family", "model_order"])
    return summary, by_model_df


def make_breakdown_row(pool: str, subset: str, g: pd.DataFrame) -> dict[str, Any]:
    total = len(g)
    helped = int(g["augmented_helped"].sum())
    native = total - helped
    tasks = int(g["sample_id"].nunique())
    models = int(g["display_model"].nunique())
    return {
        "pool": pool,
        "subset": subset,
        "n_tasks": tasks,
        "n_models": models,
        "n_pairs": total,
        "native_or_tie_pairs": native,
        "augmented_helped_pairs": helped,
        "augmented_helped_rate": helped / total if total else np.nan,
        "mean_edge_delta": g["edge_delta"].mean(),
        "median_edge_delta": g["edge_delta"].median(),
        "positive_models": int(
            g.groupby("display_model")["augmented_helped"].any().sum()
        ),
    }


def write_report(
    best_summary: pd.DataFrame,
    by_model: pd.DataFrame,
    parallelism: pd.DataFrame,
    tasks: pd.DataFrame,
) -> None:
    all_open = best_summary[
        (best_summary["pool"] == "open-weight") & (best_summary["subset"] == "All")
    ].iloc[0]
    all_closed = best_summary[
        (best_summary["pool"] == "closed-model") & (best_summary["subset"] == "All")
    ].iloc[0]
    all_models = best_summary[
        (best_summary["pool"] == "all-models") & (best_summary["subset"] == "All")
    ].iloc[0]
    multi = parallelism[parallelism["bucket"] == "multi-order-rich"].iloc[0]
    allp = parallelism[parallelism["bucket"] == "all"].iloc[0]

    report = f"""# Optional Improvement Diagnostics

Generated from the paper result artifacts and replay-filtered DAG files.
No raw GAIA questions, answers, or attachments are used in the outputs.

## 1. Best-Match Reference Breakdown

Definition: a model-task pair is counted as `augmented_helped` when the
augmented best score has a strictly higher EdgeF1 than the native-chain
reference. For open-weight rows, the public paper summary only stores
PlanningScore, so EdgeF1 lift is recovered as
`2 * (augmented_best_planning_score - chain_only_planning_score)` because
NodeF1 is invariant between native and augmented references by construction.

Main open-weight pool: {int(all_open.augmented_helped_pairs)}/{int(all_open.n_pairs)}
multi-order-rich pairs ({all_open.augmented_helped_rate:.1%}) improve under an
augmented ordering rather than tying the native chain.

Closed-model extension: {int(all_closed.augmented_helped_pairs)}/{int(all_closed.n_pairs)}
multi-order-rich pairs ({all_closed.augmented_helped_rate:.1%}) improve under
an augmented ordering.

Across all 15 models: {int(all_models.augmented_helped_pairs)}/{int(all_models.n_pairs)}
pairs ({all_models.augmented_helped_rate:.1%}) improve under an augmented
ordering.

Suggested main-text sentence:

> On the multi-order-rich subset, {all_open.augmented_helped_rate:.1%} of
> open-weight task-model pairs are best matched by an augmented ordering rather
> than the native chain, so the lift is concentrated in genuinely parallel
> tasks rather than spread uniformly across pairs.

## 2. Critical Path vs. Native Chain Length

Definition: native chain length is `metadata.original_step_count`; critical
path length is the longest path in the conservative partial order induced by
the final Augmented GAIA reference orderings. The parallelism ratio is
`1 - critical_path_length / native_chain_length`.

All 165 tasks: mean native chain length {allp.native_chain_mean:.2f}, mean
critical path length {allp.critical_path_mean:.2f}, mean parallelism ratio
{allp.parallelism_ratio_mean:.3f}.

Multi-order-rich tasks: mean native chain length {multi.native_chain_mean:.2f},
mean critical path length {multi.critical_path_mean:.2f}, mean parallelism
ratio {multi.parallelism_ratio_mean:.3f}.

Suggested appendix sentence:

> Across the 165 GAIA tasks, reference-validated orderings reduce the mean
> path from a native chain length of {allp.native_chain_mean:.2f} steps to a
> critical path of {allp.critical_path_mean:.2f} steps. The effect is strongest
> on the {int(multi.n_tasks)} multi-order-rich tasks, where the mean critical path is
> {multi.critical_path_mean:.2f} versus a mean native chain length of
> {multi.native_chain_mean:.2f}, corresponding to a mean parallelism ratio of
> {multi.parallelism_ratio_mean:.1%}.

## Output Files

- `best_match_reference_breakdown_by_pool.csv`
- `best_match_reference_breakdown_by_model.csv`
- `best_match_reference_pairs.csv`
- `task_parallelism_profile.csv`
- `parallelism_summary_by_bucket.csv`
- `optional_improvements_summary.json`
- `optional_improvements_snippets.tex`
"""
    (OUT / "optional_improvements_report.md").write_text(report, encoding="utf-8")

    tex = f"""% Optional appendix snippets generated by scripts/analysis/compute_optional_improvements.py

% Best-match sentence for Section 5.3:
On the multi-order-rich subset, {all_open.augmented_helped_rate:.1%} of
open-weight task--model pairs are best matched by an augmented ordering rather
than the native chain, so the lift is concentrated in genuinely parallel tasks
rather than spread uniformly across pairs.

% Critical-path sentence for Appendix C:
Across the 165 GAIA tasks, reference-validated orderings reduce the mean
source-to-sink path
from a native chain length of {allp.native_chain_mean:.2f} steps to a critical
path of {allp.critical_path_mean:.2f} steps. The effect is strongest on the {int(multi.n_tasks)}
multi-order-rich tasks, where the mean critical path is
{multi.critical_path_mean:.2f} versus a mean native chain length of
{multi.native_chain_mean:.2f}, corresponding to a mean parallelism ratio of
{multi.parallelism_ratio_mean:.1%}.
"""
    (OUT / "optional_improvements_snippets.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    tasks = load_task_profiles()
    tasks.to_csv(OUT / "task_parallelism_profile.csv", index=False)
    parallelism = summarize_parallelism(tasks)
    parallelism.to_csv(OUT / "parallelism_summary_by_bucket.csv", index=False)
    reference_rows = load_reference_rows()
    reference_rows.to_csv(OUT / "final_reference_row_parallelism_profile.csv", index=False)
    summarize_reference_slices(reference_rows).to_csv(
        OUT / "final_reference_row_parallelism_summary_by_slice.csv", index=False
    )
    reference_thresholds = summarize_reference_thresholds(reference_rows)
    reference_thresholds.to_csv(
        OUT / "final_reference_threshold_parallelism_summary.csv", index=False
    )

    open_df = normalize_open_weight(pd.read_csv(OPEN_WEIGHT_PER_SAMPLE), tasks)
    open_df = open_df[open_df["model"].isin(OPEN_MODEL_ORDER)].copy()
    open_df["display_model"] = pd.Categorical(
        open_df["display_model"], categories=OPEN_MODEL_ORDER, ordered=True
    ).astype(str)

    openai_df = normalize_closed(OPENAI_PER_SAMPLE, "openai")
    gemini_df = normalize_closed(GEMINI_PER_SAMPLE, "gemini")
    pairs = pd.concat([open_df, openai_df, gemini_df], ignore_index=True, sort=False)
    pairs["level"] = pd.to_numeric(pairs["level"], errors="coerce")
    pairs["retained_orderings"] = pd.to_numeric(pairs["retained_orderings"], errors="coerce")
    missing_bucket = pairs["bucket"].isna() if "bucket" in pairs.columns else pd.Series(True, index=pairs.index)
    pairs.loc[missing_bucket, "bucket"] = pairs.loc[missing_bucket, "retained_orderings"].fillna(0).astype(int).map(retained_bucket)
    pairs = pairs[pairs["bucket"].notna()].copy()

    breakdown, by_model = best_match_breakdown(pairs)
    breakdown.to_csv(OUT / "best_match_reference_breakdown_by_pool.csv", index=False)
    by_model.to_csv(OUT / "best_match_reference_breakdown_by_model.csv", index=False)

    pair_cols = [
        "pool",
        "family",
        "display_model",
        "sample_id",
        "cat",
        "domain",
        "level",
        "retained_orderings",
        "bucket",
        "chain_only_planning_score",
        "augmented_best_planning_score",
        "chain_only_semantic_edge_f1",
        "augmented_best_semantic_edge_f1",
        "edge_delta",
        "augmented_helped",
    ]
    keep_cols = [c for c in pair_cols if c in pairs.columns]
    pairs[pairs["bucket"].eq("multi-order-rich")][keep_cols].to_csv(
        OUT / "best_match_reference_pairs.csv", index=False
    )

    summary = {
        "best_match": json.loads(breakdown.to_json(orient="records")),
        "parallelism": json.loads(parallelism.to_json(orient="records")),
        "task_counts": {
            "all": int(len(tasks)),
            "native_only": int((tasks["bucket"] == "native-only").sum()),
            "exactly_one": int((tasks["bucket"] == "exactly-one").sum()),
            "multi_order_rich": int((tasks["bucket"] == "multi-order-rich").sum()),
            "multi_order_rich_level2": int(
                ((tasks["bucket"] == "multi-order-rich") & (tasks["level"] == 2)).sum()
            ),
        },
    }
    (OUT / "optional_improvements_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_report(breakdown, by_model, parallelism, tasks)

    print(f"Wrote optional diagnostics to {OUT}")
    print(breakdown.to_string(index=False))
    print()
    print(parallelism.to_string(index=False))
    print()
    print(reference_thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
