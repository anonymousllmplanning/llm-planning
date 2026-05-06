#!/usr/bin/env python3
"""Generate Croissant metadata for the NeurIPS evaluation-dataset artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


def croissant_context() -> dict[str, Any]:
    return {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "equivalentProperty": "cr:equivalentProperty",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "samplingRate": "cr:samplingRate",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
        "wd": "https://www.wikidata.org/wiki/",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_object(
    file_id: str,
    name: str,
    description: str,
    content_url: str,
    encoding: str,
    local_path: Path | None = None,
) -> dict[str, Any]:
    obj = {
        "@type": "cr:FileObject",
        "@id": file_id,
        "name": name,
        "description": description,
        "contentUrl": content_url,
        "encodingFormat": encoding,
    }
    if local_path and local_path.exists():
        obj["sha256"] = sha256_file(local_path)
    return obj


def manifest_field(field_id: str, name: str, description: str, json_path: str, data_type: str = "sc:Text") -> dict[str, Any]:
    return {
        "@type": "cr:Field",
        "@id": field_id,
        "name": name,
        "description": description,
        "dataType": data_type,
        "source": {
            "fileObject": {"@id": "package_manifest"},
            "extract": {"jsonPath": json_path},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-url", default="https://huggingface.co/datasets/anonymousllmplanning/augmented-gaia-planning")
    parser.add_argument("--repo-url", default="https://github.com/anonymousllmplanning/llm-planning")
    parser.add_argument("--output", type=Path, default=Path("release/neurips2026_ed_dataset/croissant.json"))
    parser.add_argument("--name", default="Augmented GAIA Planning Evaluation")
    args = parser.parse_args()

    base = args.dataset_url.rstrip("/")
    release_root = args.output.parent
    metadata = {
        "@context": croissant_context(),
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
        "datePublished": str(date.today()),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "version": "1.0.0",
        "citeAs": "Anonymous submission artifact for Augmented GAIA Planning Evaluation.",
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
        "rai:dataCollection": (
            "The artifact is built from existing benchmark sources and derived "
            "planning/tool annotations. GAIA raw questions, final answers, and "
            "attachments are not redistributed; users rebuild them locally from "
            "the official gated GAIA source."
        ),
        "rai:dataCollectionType": "Derived annotations and evaluation-result summaries over existing benchmark tasks.",
        "rai:dataCollectionTypeOthers": "No web scraping or new human-subject data collection is performed.",
        "rai:dataCollectionMissing": (
            "GAIA raw task text, final answers, and attachments are intentionally "
            "omitted from the public artifact and must be obtained from the "
            "official gated dataset."
        ),
        "rai:dataCollectionRaw": (
            "Raw upstream data sources are GAIA, TaskBench, and UltraTool. The "
            "released artifact contains controlled annotations, scripts, checksums, "
            "and sanitized aggregate summaries."
        ),
        "rai:dataPreprocessingImputation": "No statistical imputation is applied.",
        "rai:dataPeprocessingProtocol": (
            "Scripts rebuild a local evaluation layout by merging official source "
            "records with derived annotations. The final Augmented GAIA scoring "
            "view contains 165 native chain references plus 1,357 Gemma 4-retained "
            "non-native async orderings."
        ),
        "rai:dataPreprocessingManipulation": (
            "GAIA questions, answers, and attachments are removed from released "
            "annotation bundles; local rebuild restores them from official sources."
        ),
        "rai:dataAnnotationProtocol": (
            "Dependency annotations are generated over planning steps, candidate "
            "async orderings are sampled from those dependencies, and Gemma 4 replay "
            "filtering retains behavior-preserving non-native orderings."
        ),
        "rai:dataAnnotationPlatform": "Local scripted annotation and replay-filtering pipeline.",
        "rai:dataAnnotationAnalysis": (
            "The paper reports construction statistics, replay-filter retention, "
            "human spot checks, metric definitions, and aggregate model results."
        ),
        "rai:dataAnnotationPerItem": (
            "Each GAIA task receives one native chain reference; tasks with retained "
            "non-native async orderings include those additional references."
        ),
        "rai:dataAnnotationDemographics": (
            "Not applicable: the artifact does not collect participant demographic "
            "data or conduct a human-subject study."
        ),
        "rai:dataAnnotationTools": "LLM-assisted dependency annotation, scripted sampling, and replay filtering.",
        "rai:dataBiases": [
            "Coverage follows the upstream benchmark task distributions, including small Vision and Audio slices.",
            "Replay filtering depends on one replay model and may exclude valid orderings that fail that model's behavior-preservation check.",
        ],
        "rai:dataUseCases": (
            "Research evaluation of agent planning, tool invocation, and final-answer "
            "correctness; audit and reproduction of the accompanying paper."
        ),
        "rai:dataLimitation": (
            "Not intended for training deployed models or for redistributing GAIA raw "
            "validation/test content. Full GAIA reproduction requires official gated access."
        ),
        "rai:dataSocialImpact": (
            "The artifact supports more transparent agent evaluation, but benchmark "
            "scores should not be interpreted as deployment readiness."
        ),
        "rai:dataSensitive": (
            "No intentionally released personal or sensitive data beyond upstream "
            "benchmark terms; GAIA raw files are excluded from the artifact."
        ),
        "rai:dataMaintenance": (
            "Versioned artifact with checksums and rebuild scripts; updates should "
            "regenerate the package manifest and Croissant metadata."
        ),
        "distribution": [
            file_object(
                "package_manifest",
                "Package manifest",
                "JSON manifest listing released files, byte sizes, and checksums.",
                f"{base}/resolve/main/package_manifest.json",
                "application/json",
                release_root / "package_manifest.json",
            ),
            file_object(
                "gaia_annotations",
                "Controlled Augmented GAIA annotation bundle",
                "Annotation sidecar merged with an official GAIA snapshot by scripts/prepare_gaia_from_official.py.",
                f"{base}/resolve/main/annotations/gaia_annotations.zip",
                "application/zip",
                release_root / "annotations/gaia_annotations.zip",
            ),
            file_object(
                "aggregate_results",
                "Aggregate result summaries and figure data",
                "Sanitized aggregate tables, checksums, and figure source data used to audit the paper numbers.",
                f"{base}/resolve/main/results/results_summaries.zip",
                "application/zip",
                release_root / "results/results_summaries.zip",
            ),
            file_object(
                "scripts",
                "Preparation and evaluation scripts",
                "Scripts for fetching upstream sources, rebuilding local data, running experiments, and evaluating results.",
                f"{base}/resolve/main/scripts/scripts.zip",
                "application/zip",
                release_root / "scripts/scripts.zip",
            ),
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "artifact_files",
                "name": "Artifact file manifest",
                "description": "Released artifact files with byte sizes and checksums.",
                "key": ["path"],
                "field": [
                    manifest_field("artifact_files/path", "path", "Path within the dataset artifact.", "$.files[*].path"),
                    manifest_field("artifact_files/sha256", "sha256", "SHA-256 checksum.", "$.files[*].sha256"),
                    manifest_field("artifact_files/bytes", "bytes", "File size in bytes.", "$.files[*].bytes", "sc:Integer"),
                ],
            },
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()
