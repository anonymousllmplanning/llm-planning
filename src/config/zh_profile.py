"""
Centralized configuration for Chinese cross-lingual GAIA evaluation.

English remains the default architecture. This module only adds a
dataset-driven profile for *_zh datasets so the rest of the pipeline can stay
shared with the English path.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional, Sequence

ZH_DATASETS = (
    "gaia_cat_A_zh",
    "gaia_cat_B_zh",
    "gaia_cat_C_zh",
    "gaia_cat_D_zh",
)

ZH_MODEL_SET = (
    "Google-Gemma-3-27B",
    "Mistral-Small-3.2-24B-Instruct-2506",
    "Llama-3.1-405B-Instruct-FP8",
    "Gemma-3-TAIDE-12b-Chat",
)

_COMMON_GUIDANCE = [
    "This is a Traditional Chinese cross-lingual task. Understand the user request in Traditional Chinese.",
    "Use Traditional Chinese for free-form reasoning or step descriptions when helpful.",
    "Keep tool_id values, argument names, file names, file paths, URLs, domain names, JSON keys, and schema literals exactly in English.",
    "Preserve English proper nouns, quoted strings, titles, and required numeric or date formats when the task depends on them.",
]

_STAGE_SPECIFIC_GUIDANCE = {
    "abstract": [
        "Focus on solving the user's request; do not translate or rename schema fields.",
    ],
    "creation": [
        "Evaluate tool sufficiency normally, and do not invent translated tool names.",
    ],
    "refinement": [
        "When constructing tool calls, keep tool IDs and argument names exactly in English.",
        "Only natural-language labels or free-text argument values may reflect the user's language.",
    ],
}

_SYSTEM_PROMPT_GUIDANCE = [
    "Chinese instructions may appear in the user prompt, but JSON keys and schema literals must remain in English.",
    "Return English proper nouns or original numeric formats unchanged when the task expects them.",
]


def is_zh_dataset(dataset: Optional[str]) -> bool:
    """Return True when the dataset uses the Chinese cross-lingual profile."""
    return bool(dataset) and dataset in ZH_DATASETS


def get_stage_prompt_suffix(dataset: Optional[str], stage: str) -> str:
    """Return an additive prompt block for Chinese cross-lingual datasets."""
    if not is_zh_dataset(dataset):
        return ""

    lines = list(_COMMON_GUIDANCE)
    lines.extend(_STAGE_SPECIFIC_GUIDANCE.get(stage, ()))
    bullet_block = "\n".join(f"- {line}" for line in lines)
    return "## Cross-Lingual Guidance\n" + bullet_block


def get_system_prompt_suffix(dataset: Optional[str]) -> str:
    """Return an additive system prompt block for Chinese cross-lingual datasets."""
    if not is_zh_dataset(dataset):
        return ""

    lines = [_COMMON_GUIDANCE[0], *_SYSTEM_PROMPT_GUIDANCE]
    bullet_block = "\n".join(f"- {line}" for line in lines)
    return "CHINESE CROSS-LINGUAL DATASET NOTE:\n" + bullet_block


def _emit_lines(values: Sequence[str]) -> None:
    for value in values:
        print(value)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Small CLI for shell scripts that need the centralized ZH profile."""
    parser = argparse.ArgumentParser(description="Chinese GAIA cross-lingual profile")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", action="store_true", help="Print the zh model set, one per line.")
    group.add_argument("--datasets", action="store_true", help="Print the zh dataset set, one per line.")
    group.add_argument("--dataset", type=str, help="Print one validated zh dataset name.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.models:
        _emit_lines(ZH_MODEL_SET)
        return 0
    if args.datasets:
        _emit_lines(ZH_DATASETS)
        return 0
    if args.dataset:
        if not is_zh_dataset(args.dataset):
            valid = ", ".join(ZH_DATASETS)
            parser.error(f"Unknown zh dataset: {args.dataset}. Valid values: {valid}")
        print(args.dataset)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
