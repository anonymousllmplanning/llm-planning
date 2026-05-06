#!/usr/bin/env python3
"""Smoke-test ParamF1 optional execution-control argument normalization.

This is intentionally dataset-free: it verifies the framework metric behavior
used by all future open-weight and closed-model runs through
ASTEvaluationSystem.evaluate_tool_calls().
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import ASTEvaluationSystem  # noqa: E402


def _call(tool_id: str, **arguments: str) -> dict:
    return {
        "tool_id": tool_id,
        "arguments": [{"name": name, "value": value} for name, value in arguments.items()],
    }


def _assert_close(name: str, value: float, expected: float) -> None:
    if not math.isclose(float(value), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{name}: expected {expected}, got {value}")


def main() -> None:
    evaluator = ASTEvaluationSystem(use_embeddings=False, verifier_model=None)

    gold = [_call("web_search", query="weather taipei")]

    with_control_defaults = [
        _call(
            "web_search",
            query="weather taipei",
            engine="google",
            max_results="5",
            page="1",
            language="en",
            action="search",
        )
    ]
    scores = evaluator.evaluate_tool_calls(gold, with_control_defaults, dataset="gaia")
    _assert_close("control defaults ignored", scores.param_name_f1, 1.0)
    if scores.pred_param_count != 1:
        raise AssertionError(f"expected one counted predicted param, got {scores.pred_param_count}")

    with_real_extra = [_call("web_search", query="weather taipei", timezone="Asia/Taipei")]
    scores = evaluator.evaluate_tool_calls(gold, with_real_extra, dataset="gaia")
    _assert_close("non-control extra still penalized", scores.param_name_f1, 2.0 / 3.0)

    gold_requires_language = [_call("translation", text="hello", language="fr")]
    pred_has_language = [_call("translation", text="hello", language="fr", engine="default")]
    scores = evaluator.evaluate_tool_calls(gold_requires_language, pred_has_language, dataset="gaia")
    _assert_close("gold-required control arg retained", scores.param_name_f1, 1.0)
    if scores.pred_param_count != 2:
        raise AssertionError(f"expected gold-required language to be counted, got {scores.pred_param_count}")

    print("optional argument normalization smoke test passed")


if __name__ == "__main__":
    main()
