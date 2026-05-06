#!/usr/bin/env python3
"""Recompute GAIA tool metrics after optional-control argument normalization.

This script leaves model generations unchanged. It reruns only the tool-call
metric layer from cached unified JSONL files and writes paper-facing aggregate
CSVs used to update the manuscript tables. If the main open-weight A/C/D run
cache is stored outside the repository, set MAIN8_ACD_RUN_ROOT to the directory
that contains cat_A/, cat_C/, and cat_D/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation.metrics import ASTEvaluationSystem  # noqa: E402
from src.inference.prompts import normalize_tool_environment  # noqa: E402


OUT = REPO / "version2" / "paper_results" / "optional_arg_normalization"
OUT.mkdir(parents=True, exist_ok=True)

CATS = {
    "A": REPO / "data" / "Augmented" / "cat_A_text" / "gaia.cat_A.json",
    "B": REPO / "data" / "Augmented" / "cat_B_document" / "gaia.cat_B.json",
    "C": REPO / "data" / "Augmented" / "cat_C_vision" / "gaia.cat_C.json",
    "D": REPO / "data" / "Augmented" / "cat_D_audio" / "gaia.cat_D.json",
}

OPEN_WEIGHT_MODELS: List[Tuple[str, str, str]] = [
    ("Mistral Large 3", "675B", "Mistral-Large-3-675B-Instruct-2512"),
    ("Llama-3.1", "405B", "Llama-3.1-405B-Instruct-FP8"),
    ("Llama-3.3", "70B", "Llama-3.3-70B-Instruct"),
    ("Gemma 4", "31B", "gemma-4-31B-it"),
    ("Mistral Small 3.2", "24B", "Mistral-Small-3.2-24B-Instruct-2506"),
    ("Llama-4 Maverick", "17B", "Llama-4-Maverick-17B-128E-Instruct-FP8"),
    ("Gemma 3", "12B", "gemma-3-12b-it"),
]

OPENAI_MODELS: List[Tuple[str, str, str]] = [
    ("GPT-5.5", "--", "gpt-5.5"),
    ("GPT-5.4", "--", "gpt-5.4"),
    ("GPT-5.4-Mini", "--", "gpt-5.4-mini"),
    ("GPT-5.4-Nano", "--", "gpt-5.4-nano"),
]

OPENAI_DIRS = {
    "gpt-5.5": "gpt5.5",
    "gpt-5.4": "gpt5.4",
    "gpt-5.4-mini": "gpt5.4mini",
    "gpt-5.4-nano": "gpt5.4nano",
}

GEMINI_MODELS: List[Tuple[str, str, str]] = [
    ("Gemini 3 Flash", "--", "gemini-3-flash-preview"),
    ("Gemini 3.1 Flash-Lite", "--", "gemini-3.1-flash-lite-preview"),
    ("Gemini 2.5 Flash", "--", "gemini-2.5-flash"),
    ("Gemini 2.5 Flash-Lite", "--", "gemini-2.5-flash-lite"),
]

OPEN_WEIGHT_ACD_ROOT = Path(os.getenv("MAIN8_ACD_RUN_ROOT", REPO / "organized_results" / "gaia"))

OPEN_WEIGHT_RUNS = {
    "A": OPEN_WEIGHT_ACD_ROOT / "cat_A" / "main8_answer_acd_tmp_20260504_1817.gaia_cat_A.answer",
    "B": REPO / "organized_results" / "gaia" / "cat_B" / "main8_answer_bacd_finalroot_20260504_1541.gaia_cat_B.answer",
    "C": OPEN_WEIGHT_ACD_ROOT / "cat_C" / "main8_answer_acd_tmp_20260504_1817.gaia_cat_C.answer",
    "D": OPEN_WEIGHT_ACD_ROOT / "cat_D" / "main8_answer_acd_tmp_20260504_1817.gaia_cat_D.answer",
}


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gold() -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for cat, path in CATS.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        out[cat] = {str(row["meta"]["id"]): row for row in data}
    return out


def run_root_for(family: str, cat: str, slug: str) -> Path:
    if family == "open-weight":
        return OPEN_WEIGHT_RUNS[cat]
    if family == "openai":
        return REPO / "organized_results" / "gaia" / f"cat_{cat}" / OPENAI_DIRS[slug]
    if family == "gemini":
        matches = sorted((REPO / "gemini_results" / "gaia" / f"cat_{cat}").glob(f"*.answer/{slug}"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Gemini run for cat {cat} {slug}, found {len(matches)}: {matches}")
        return matches[0].parent
    raise ValueError(f"Unknown family: {family}")


def unified_path_for(family: str, cat: str, slug: str) -> Path:
    root = run_root_for(family, cat, slug)
    return root / slug / f"unified.{slug}.jsonl"


def per_sample_path_for(family: str, cat: str, slug: str) -> Optional[Path]:
    if family == "open-weight":
        return OPEN_WEIGHT_RUNS[cat] / f"per_sample.{slug}.csv"
    if family == "openai":
        return REPO / "organized_results" / "gaia" / f"cat_{cat}" / OPENAI_DIRS[slug] / f"per_sample.{slug}.csv"
    if family == "gemini":
        root = run_root_for(family, cat, slug)
        return root / f"per_sample.{slug}.csv"
    return None


def recompute_family(
    family: str,
    models: List[Tuple[str, str, str]],
    gold_by_cat: Dict[str, Dict[str, Dict[str, Any]]],
    evaluator: ASTEvaluationSystem,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for display, params, slug in models:
        for cat in CATS:
            unified = unified_path_for(family, cat, slug)
            old_by_id: Dict[str, Dict[str, Any]] = {}
            per_path = per_sample_path_for(family, cat, slug)
            if per_path and per_path.exists():
                old_df = pd.read_csv(per_path)
                old_by_id = {str(r["sample_id"]): r for _, r in old_df.iterrows()}
            for rec in load_jsonl(unified):
                sid = str((rec.get("meta") or {}).get("id"))
                gold = gold_by_cat[cat][sid].get("gold") or {}
                pred = rec.get("pred") or {}
                available_tools = normalize_tool_environment(rec.get("tool_environment") or [])
                tool_scores = evaluator.evaluate_tool_calls(
                    gold.get("tool_calls") or [],
                    pred.get("tool_calls") or [],
                    has_arguments=True,
                    available_tools=available_tools,
                    pred_tool_outputs=pred.get("tool_outputs") or [],
                    dataset="gaia",
                    subset=str((rec.get("meta") or {}).get("subset") or ""),
                )
                old = old_by_id.get(sid, {})
                rows.append(
                    {
                        "family": family,
                        "model": display,
                        "params": params,
                        "model_id": slug,
                        "cat": cat,
                        "sample_id": sid,
                        "old_param_f1": pd.to_numeric(old.get("param_name_f1"), errors="coerce"),
                        "new_param_f1": tool_scores.param_name_f1,
                        "old_param_value_f1": pd.to_numeric(
                            old.get("normalized_type_aware_value_f1", old.get("type_aware_value_f1")),
                            errors="coerce",
                        ),
                        "new_param_value_f1": tool_scores.normalized_type_aware_value_f1
                        if tool_scores.normalized_type_aware_value_f1 is not None
                        else tool_scores.type_aware_value_f1,
                        "old_tool_usage_score": pd.to_numeric(old.get("tool_usage_score"), errors="coerce"),
                        "new_tool_usage_score": (
                            sum(
                                v
                                for v in [
                                    tool_scores.tool_name_f1,
                                    tool_scores.param_name_f1,
                                    tool_scores.type_aware_value_f1,
                                ]
                                if v is not None
                            )
                            / len(
                                [
                                    v
                                    for v in [
                                        tool_scores.tool_name_f1,
                                        tool_scores.param_name_f1,
                                        tool_scores.type_aware_value_f1,
                                    ]
                                    if v is not None
                                ]
                            )
                        ),
                        "new_tool_f1": tool_scores.tool_name_f1,
                    }
                )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (family, model, params), d in df.groupby(["family", "model", "params"], sort=False):
        rows.append(
            {
                "family": family,
                "model": model,
                "params": params,
                "N": len(d),
                "old_param_f1": pd.to_numeric(d["old_param_f1"], errors="coerce").mean(),
                "new_param_f1": pd.to_numeric(d["new_param_f1"], errors="coerce").mean(),
                "delta_param_f1": pd.to_numeric(d["new_param_f1"], errors="coerce").mean()
                - pd.to_numeric(d["old_param_f1"], errors="coerce").mean(),
                "old_param_value_f1": pd.to_numeric(d["old_param_value_f1"], errors="coerce").mean(),
                "new_param_value_f1": pd.to_numeric(d["new_param_value_f1"], errors="coerce").mean(),
                "old_tool_usage_score": pd.to_numeric(d["old_tool_usage_score"], errors="coerce").mean(),
                "new_tool_usage_score": pd.to_numeric(d["new_tool_usage_score"], errors="coerce").mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    evaluator = ASTEvaluationSystem(use_embeddings=False, verifier_model=None)
    gold_by_cat = load_gold()
    sample_parts = [
        recompute_family("open-weight", OPEN_WEIGHT_MODELS, gold_by_cat, evaluator),
        recompute_family("openai", OPENAI_MODELS, gold_by_cat, evaluator),
        recompute_family("gemini", GEMINI_MODELS, gold_by_cat, evaluator),
    ]
    sample_df = pd.concat(sample_parts, ignore_index=True)
    summary_df = summarize(sample_df)
    sample_df.to_csv(OUT / "gaia_optional_arg_normalized_tool_metrics_by_sample.csv", index=False)
    summary_df.to_csv(OUT / "gaia_optional_arg_normalized_tool_metrics_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
