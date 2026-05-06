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
        "prov": "http://www.w3.org/ns/prov#",
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
            "sources. The primary dataset contribution is the Augmented GAIA "
            "annotation and reference layer; TaskBench and UltraTool are auxiliary "
            "cross-benchmark validation sources. The artifact does not redistribute "
            "GAIA validation/test questions, final answers, or attachments."
        ),
        "url": args.dataset_url,
        "codeRepository": args.repo_url,
        "dateCreated": str(date.today()),
        "datePublished": str(date.today()),
        "conformsTo": [
            "http://mlcommons.org/croissant/1.1",
            "http://mlcommons.org/croissant/RAI/1.0",
        ],
        "version": "1.0.0",
        "citeAs": "Anonymous submission artifact for Augmented GAIA Planning Evaluation.",
        "license": "https://creativecommons.org/licenses/by/4.0/",
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
        "prov:wasDerivedFrom": [
            {
                "@id": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
                "prov:label": "GAIA",
                "description": (
                    "Official gated GAIA benchmark source used as the parent task "
                    "set for Augmented GAIA. This artifact uses derived annotations "
                    "over the GAIA validation split, whose final answers support "
                    "local answer-correctness scoring. Raw GAIA questions, final "
                    "answers, and attachments are not redistributed in this artifact, "
                    "and the held-out GAIA test split is not used for local answer "
                    "scoring because its answers are private."
                ),
                "sc:license": "Official gated Hugging Face dataset terms.",
            }
        ],
        "prov:wasGeneratedBy": [
            {
                "@type": "prov:Activity",
                "prov:label": "Collection from existing benchmark sources",
                "prov:type": "Collection",
                "description": (
                    "The primary raw source is the official gated GAIA benchmark. "
                    "TaskBench and UltraTool are referenced only as auxiliary "
                    "cross-benchmark validation sources and are materialized by "
                    "separate rebuild scripts."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:Agent",
                        "@id": "anonymous_research_team",
                        "prov:label": "Anonymous research team",
                        "description": "Research team identity withheld for double-blind review.",
                    },
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "scripts/fetch_official_sources.py",
                        "prov:label": "fetch_official_sources.py",
                        "description": "Helper script for obtaining official upstream sources locally.",
                    },
                ],
            },
            {
                "@type": "prov:Activity",
                "prov:label": "Local rebuild and sanitization",
                "prov:type": "Preprocessing",
                "description": (
                    "Scripts rebuild a local evaluation layout from official sources "
                    "and derived annotations. GAIA raw questions, final answers, "
                    "and attachments are removed from released archives; local "
                    "rebuild restores them from the official gated source."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "scripts/prepare_gaia_from_official.py",
                        "prov:label": "prepare_gaia_from_official.py",
                    },
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "scripts/export_gaia_annotations.py",
                        "prov:label": "export_gaia_annotations.py",
                    },
                ],
            },
            {
                "@type": "prov:Activity",
                "prov:label": "Dependency annotation and async-ordering generation",
                "prov:type": "Annotation",
                "description": (
                    "GPT-4o is used to annotate dependency DAGs over planning-intent "
                    "steps. Candidate non-native async orderings are then sampled "
                    "from those dependency constraints."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "GPT-4o",
                        "prov:label": "GPT-4o",
                        "description": "LLM used for dependency annotation.",
                    },
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "async_ordering_sampler",
                        "prov:label": "Dependency-aware async-ordering sampler",
                    },
                ],
            },
            {
                "@type": "prov:Activity",
                "prov:label": "Behavior-preserving replay filtering",
                "prov:type": "Filtering",
                "description": (
                    "Gemma 4 replay filtering keeps non-native async orderings that "
                    "preserve native-chain execution behavior under the shared tool "
                    "layer. The final Augmented GAIA scoring view contains 165 native "
                    "chain references plus 1,357 retained non-native references."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "Gemma_4",
                        "prov:label": "Gemma 4",
                        "description": "Replay model used for behavior-preserving filtering.",
                    },
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "scripts/utils/model_guided_async_ordering_replay.py",
                        "prov:label": "model_guided_async_ordering_replay.py",
                    },
                ],
            },
            {
                "@type": "prov:Activity",
                "prov:label": "Release packaging",
                "prov:type": "Packaging",
                "description": (
                    "Release scripts package controlled annotations, sanitized "
                    "aggregate summaries, checksums, code, and Croissant metadata."
                ),
                "prov:wasAttributedTo": [
                    {
                        "@type": "prov:SoftwareAgent",
                        "@id": "scripts/generate_croissant.py",
                        "prov:label": "generate_croissant.py",
                    }
                ],
            },
        ],
        "rai:dataCollection": (
            "The artifact is built from existing benchmark sources and derived "
            "planning/tool annotations. The primary GAIA-derived component uses "
            "the official validation split because its final answers support local "
            "answer-correctness scoring. GAIA raw questions, final answers, and "
            "attachments are not redistributed; users rebuild them locally from "
            "the official gated GAIA source."
        ),
        "rai:dataCollectionType": [
            "Secondary Data analysis",
            "Existing datasets",
            "Others: LLM-assisted derived annotation and scripted replay filtering",
        ],
        "rai:dataCollectionMissingData": (
            "GAIA raw task text, final answers, and attachments are intentionally "
            "omitted from the public artifact and must be obtained from the "
            "official gated dataset. The held-out GAIA test split is not included "
            "in local answer scoring because its final answers are private."
        ),
        "rai:dataCollectionRawData": (
            "The primary raw upstream source is the official gated GAIA dataset. "
            "The released artifact uses derived annotations over the GAIA validation "
            "split and does not redistribute GAIA raw questions, final answers, or "
            "attachments. The held-out GAIA test split is not used for local answer "
            "scoring because its final answers are private. TaskBench and UltraTool "
            "are used only as auxiliary cross-benchmark validation sources through "
            "rebuild scripts. The released artifact contains controlled annotations, "
            "scripts, checksums, and sanitized aggregate summaries."
        ),
        "rai:dataImputationProtocol": "No statistical imputation is applied.",
        "rai:dataPreprocessingProtocol": [
            (
                "Scripts rebuild a local evaluation layout by merging official source "
                "records with derived annotations."
            ),
            (
                "The final Augmented GAIA scoring view contains 165 native chain "
                "references plus 1,357 Gemma 4-retained non-native async orderings."
            ),
        ],
        "rai:dataManipulationProtocol": (
            "GAIA questions, answers, and attachments are removed from released "
            "annotation bundles; local rebuild restores them from official sources."
        ),
        "rai:dataAnnotationProtocol": [
            (
                "Dependency annotations are generated over planning steps using GPT-4o."
            ),
            (
                "Candidate non-native async orderings are sampled from the dependency "
                "constraints and filtered by Gemma 4 replay for behavior preservation."
            ),
        ],
        "rai:dataAnnotationPlatform": ["Local scripted annotation and replay-filtering pipeline."],
        "rai:dataAnnotationAnalysis": [
            (
                "The paper reports construction statistics, replay-filter retention, "
                "human spot checks, metric definitions, and aggregate model results."
            )
        ],
        "rai:annotationsPerItem": (
            "Each GAIA task receives one native chain reference; tasks with retained "
            "non-native async orderings include those additional references."
        ),
        "rai:annotatorDemographics": (
            "Not applicable: the artifact does not collect participant demographic "
            "data or conduct a human-subject study."
        ),
        "rai:machineAnnotationTools": [
            "GPT-4o dependency annotation",
            "Dependency-aware async-ordering sampler",
            "Gemma 4 behavior-preserving replay filter",
            "Local Python rebuild and evaluation scripts",
        ],
        "rai:dataBiases": (
            "Coverage follows the upstream GAIA task distribution, including small "
            "Vision and Audio slices. The replay-filtered reference set also depends "
            "on one replay model, so valid reorderings may be excluded when they fail "
            "that model's behavior-preservation check."
        ),
        "rai:dataUseCases": (
            "Validated use cases are research evaluation of agent planning, tool "
            "invocation, and final-answer correctness, plus audit and reproduction "
            "of the accompanying paper's aggregate results. The artifact is not "
            "validated for model training, fine-tuning, deployment readiness "
            "assessment, or replacing official GAIA access."
        ),
        "rai:dataLimitations": (
            "The artifact is evaluation-only and is not intended for training or "
            "fine-tuning deployed models. It uses derived annotations over the GAIA "
            "validation split and does not redistribute GAIA raw validation/test "
            "questions, final answers, or attachments, so full GAIA reproduction "
            "requires official gated access. The held-out GAIA test split is not "
            "used for local answer scoring because its final answers are private. "
            "The benchmark inherits GAIA's task distribution and has small Vision "
            "and Audio slices."
        ),
        "rai:dataSocialImpact": (
            "The artifact supports more transparent diagnosis of tool-using agents "
            "by separating planning, tool invocation, and final-answer correctness. "
            "Risks include over-interpreting benchmark scores as deployment readiness "
            "or using the annotations outside their validated evaluation setting. "
            "Mitigations include evaluation-only documentation, source-access guidance, "
            "sanitized release archives, and explicit exclusion of GAIA raw content."
        ),
        "rai:hasSyntheticData": True,
        "rai:personalSensitiveInformation": (
            "No intentionally released personal or sensitive data beyond upstream "
            "benchmark terms; GAIA raw files are excluded from the artifact."
        ),
        "rai:dataReleaseMaintenancePlan": (
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
