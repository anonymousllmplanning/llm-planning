#!/usr/bin/env python3
"""Generate Croissant metadata for the NeurIPS evaluation-dataset artifact."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def file_object(file_id: str, name: str, description: str, content_url: str, encoding: str) -> dict[str, Any]:
    return {
        "@type": "cr:FileObject",
        "@id": file_id,
        "name": name,
        "description": description,
        "contentUrl": content_url,
        "encodingFormat": encoding,
    }


def field(field_id: str, name: str, description: str, data_type: str = "sc:Text") -> dict[str, Any]:
    return {
        "@type": "cr:Field",
        "@id": field_id,
        "name": name,
        "description": description,
        "dataType": data_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-url", default="https://huggingface.co/datasets/anonymousllmplanning/augmented-gaia-planning")
    parser.add_argument("--repo-url", default="https://github.com/anonymousllmplanning/llm-planning")
    parser.add_argument("--output", type=Path, default=Path("release/neurips2026_ed_dataset/croissant.json"))
    parser.add_argument("--name", default="Augmented GAIA Planning Evaluation")
    args = parser.parse_args()

    base = args.dataset_url.rstrip("/")
    metadata = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
        },
        "@type": "sc:Dataset",
        "name": args.name,
        "description": (
            "Evaluation artifact for a three-facet benchmark of agentic planning, "
            "tool invocation, and answer correctness. The artifact provides code, "
            "controlled Augmented GAIA annotations, aggregate result summaries, and "
            "scripts to rebuild the local evaluation data from official upstream "
            "sources. It does not redistribute GAIA validation/test questions, final "
            "answers, or attachments."
        ),
        "url": args.dataset_url,
        "codeRepository": args.repo_url,
        "dateCreated": str(date.today()),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": (
            "Derived annotations and code are released under the artifact license. "
            "GAIA raw data remains subject to the official gated GAIA terms; "
            "TaskBench is MIT; UltraTool is Apache-2.0."
        ),
        "keywords": [
            "agent evaluation",
            "planning",
            "tool use",
            "GAIA",
            "TaskBench",
            "UltraTool",
            "dependency DAG",
        ],
        "isBasedOn": [
            "https://huggingface.co/datasets/gaia-benchmark/GAIA",
            "https://huggingface.co/datasets/microsoft/Taskbench",
            "https://github.com/JoeYing1019/UltraTool",
        ],
        "distribution": [
            file_object(
                "gaia_annotations",
                "Controlled Augmented GAIA annotation bundle",
                "Annotation sidecar merged with an official GAIA snapshot by scripts/prepare_gaia_from_official.py.",
                f"{base}/resolve/main/annotations/gaia_annotations.zip",
                "application/zip",
            ),
            file_object(
                "aggregate_results",
                "Aggregate result summaries and figure data",
                "Sanitized aggregate tables, checksums, and figure source data used to audit the paper numbers.",
                f"{base}/resolve/main/results/results_summaries.zip",
                "application/zip",
            ),
            file_object(
                "scripts",
                "Preparation and evaluation scripts",
                "Scripts for fetching upstream sources, rebuilding local data, running experiments, and evaluating results.",
                f"{base}/resolve/main/scripts/scripts.zip",
                "application/zip",
            ),
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "augmented_gaia_annotations",
                "name": "Augmented GAIA annotation rows",
                "description": (
                    "Task-indexed planning/tool annotations, unfiltered candidate async orderings, "
                    "and Gemma-4 behavior-preserving retained async orderings. Top-level GAIA "
                    "questions, final answers, and attachments are omitted from the controlled bundle."
                ),
                "key": ["task_id", "domain"],
                "field": [
                    field("augmented_gaia_annotations/task_id", "task_id", "GAIA task identifier."),
                    field("augmented_gaia_annotations/domain", "domain", "GAIA input domain: text, document, vision, or audio."),
                    field("augmented_gaia_annotations/plan_dag", "plan_dag", "Gold planning DAG and dependency edges."),
                    field("augmented_gaia_annotations/tool_calls", "tool_calls", "Gold tool slots and argument annotations."),
                    field("augmented_gaia_annotations/sampled_orderings", "sampled_orderings", "Candidate or retained dependency-preserving async orderings."),
                    field("augmented_gaia_annotations/filter_metadata", "filter_metadata", "Gemma 4 replay-filter metadata for retained-ordering views."),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "local_augmented_gaia_view",
                "name": "Rebuilt local Augmented GAIA view",
                "description": (
                    "Generated local records created by merging official GAIA data with the annotation bundle. "
                    "This view is produced by scripts/prepare_gaia_from_official.py and is not included as a "
                    "crawlable public distribution."
                ),
                "key": ["task_id"],
                "field": [
                    field("local_augmented_gaia_view/query", "query", "Official GAIA question text and attachment references, present only after local rebuild."),
                    field("local_augmented_gaia_view/gold", "gold", "Merged plan, tool, and final-answer references used by the evaluator."),
                    field("local_augmented_gaia_view/dags", "DAGs", "Evaluator-facing filtered DAG reference folder with 1,468 retained orderings."),
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "crossbench_unified",
                "name": "TaskBench and UltraTool cross-benchmark records",
                "description": (
                    "Generated unified JSONL files for the paper's TaskBench and UltraTool transfer checks: "
                    "1,000 TaskBench rows and 1,000 UltraTool English rows."
                ),
                "key": ["dataset", "sample_id"],
                "field": [
                    field("crossbench_unified/dataset", "dataset", "TaskBench or UltraTool."),
                    field("crossbench_unified/query", "query", "User-facing task request."),
                    field("crossbench_unified/plan_dag", "plan_dag", "Planning graph reference."),
                    field("crossbench_unified/tool_calls", "tool_calls", "Tool-use reference slots."),
                ],
            },
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()
