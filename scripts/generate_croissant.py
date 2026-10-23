#!/usr/bin/env python3
"""Describe an explicitly supplied controlled, unfiltered GAIA annotation view.

Example (after obtaining the controlled annotations separately)::

    python scripts/generate_croissant.py --annotations-root CONTROLLED_BUNDLE

The output defaults to ``croissant.json`` beside the selected JSONL files.
It describes those local files and does not claim that they are published.
Public URLs and publication dates must be supplied explicitly. This generator
neither creates nor redistributes annotations, and does not describe the
code-and-aggregate release as a published dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from urllib.parse import quote, urlparse
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


def annotation_directory(root: Path) -> Path:
    """Select the current candidate view without using a legacy filtered view."""
    for candidate in (root / "Asynchronous_output", root / "GPT4o_NonNative_Orderings", root):
        if candidate.is_dir() and any(candidate.glob("gaia_cat_*_async_plan.jsonl")):
            return candidate
    raise ValueError(
        "No annotation JSONL files found. Supply --annotations-root with a controlled "
        "unfiltered view, or a bundle containing Asynchronous_output/. "
        "The code-and-aggregate release does not include those annotations."
    )


def inspect_annotations(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Count native plus unique non-native references from the supplied records."""
    seen_tasks: set[str] = set()
    summaries = []
    totals = {"tasks": 0, "native_references": 0, "non_native_references": 0, "total_references": 0}
    for path in paths:
        counts = dict.fromkeys(totals, 0)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                location = f"{path.name}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid annotation JSON at {location}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an annotation object at {location}")
                metadata = row.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(f"Expected metadata object at {location}")
                view = str(metadata.get("ordering_view", "")).lower()
                filtered = (
                    (any(word in view for word in ("filtered", "retained")) and "unfiltered" not in view)
                    or bool(metadata.get("replay_filter_model"))
                    or any("gemma4" in key.lower() for key in metadata)
                )
                if filtered:
                    raise ValueError(
                        f"Legacy replay-filtered input at {location}; supply the current "
                        "unfiltered candidate view instead."
                    )
                task_id = row.get("meta", {}).get("id") if isinstance(row.get("meta"), dict) else None
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError(f"Missing meta.id at {location}")
                if task_id in seen_tasks:
                    raise ValueError(f"Repeated task ID at {location}; select only one annotation view")
                seen_tasks.add(task_id)
                steps = row.get("executable_steps")
                sampled = row.get("sampled_orderings")
                if not isinstance(steps, list) or not isinstance(sampled, list):
                    raise ValueError(f"Expected executable_steps and sampled_orderings lists at {location}")
                native = tuple(range(len(steps)))
                orderings: set[tuple[int, ...]] = set()
                for ordering in sampled:
                    if (
                        not isinstance(ordering, list)
                        or any(type(index) is not int for index in ordering)
                        or sorted(ordering) != list(native)
                    ):
                        raise ValueError(f"Invalid step-index permutation at {location}")
                    orderings.add(tuple(ordering))
                counts["tasks"] += 1
                counts["native_references"] += 1
                counts["non_native_references"] += len(orderings - {native})
        if not counts["tasks"]:
            raise ValueError(f"No annotation records in {path.name}")
        counts["total_references"] = counts["native_references"] + counts["non_native_references"]
        summaries.append({"name": path.name, **counts})
        for key, value in counts.items():
            totals[key] += value
    return summaries, totals


def http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise argparse.ArgumentTypeError("Expected an HTTP(S) URL without embedded credentials")
    return value.rstrip("/")


def iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a publication date in YYYY-MM-DD format") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True,
                        help="Controlled unfiltered JSONL view or bundle root, obtained separately")
    parser.add_argument("--output", type=Path,
                        help="Defaults beside the input JSONL files; other locations require --content-base-url")
    parser.add_argument("--dataset-url", type=http_url,
                        help="Optional actual dataset landing URL; no hosting URL is assumed")
    parser.add_argument("--content-base-url", type=http_url,
                        help="Optional actual URL directory containing the supplied JSONL files")
    parser.add_argument("--repo-url", type=http_url, default="https://github.com/anonymousllmplanning/llm-planning")
    parser.add_argument("--published-date", type=iso_date,
                        help="Actual publication date, supplied only for an explicitly hosted dataset")
    parser.add_argument("--license", help="Optional applicable annotation license URL or identifier")
    parser.add_argument("--name", default="Augmented GAIA planning references (unfiltered)")
    args = parser.parse_args()

    if args.published_date and not (args.dataset_url and args.content_base_url):
        parser.error("--published-date requires explicit --dataset-url and --content-base-url")
    try:
        annotation_dir = annotation_directory(args.annotations_root)
        paths = sorted(annotation_dir.glob("gaia_cat_*_async_plan.jsonl"))
        summaries, totals = inspect_annotations(paths)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = args.output or annotation_dir / "croissant.json"
    if output.parent.resolve() != annotation_dir.resolve() and not args.content_base_url:
        parser.error("Local metadata must be saved beside its JSONL files; otherwise supply --content-base-url")
    if output.resolve() in {path.resolve() for path in paths}:
        parser.error("--output must not overwrite an annotation file")

    distribution = []
    record_sets = []
    for path in paths:
        file_id = path.stem
        content_url = f"{args.content_base_url}/{quote(path.name)}" if args.content_base_url else path.name
        obj = file_object(file_id, path.name, "Controlled, unfiltered planning annotation records.",
                          content_url, "application/jsonlines", path)
        obj["contentSize"] = str(path.stat().st_size)
        distribution.append(obj)
        record_sets.append({
            "@type": "cr:RecordSet",
            "@id": f"{file_id}_tasks",
            "name": f"{file_id} tasks",
            "key": {"@id": f"{file_id}/task_id"},
            "field": [{
                "@type": "cr:Field",
                "@id": f"{file_id}/task_id",
                "name": "task_id",
                "description": "Upstream GAIA task identifier used to join controlled annotations locally.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": file_id},
                    "extract": {"column": "meta"},
                    "transform": {"jsonPath": "$.id"},
                },
            }],
        })
    metadata: dict[str, Any] = {
        "@context": croissant_context(),
        "@type": "sc:Dataset",
        "name": args.name,
        "description": (
            "Metadata for explicitly supplied controlled GAIA planning annotations. "
            "The current reference policy adds one native chain per task to unique "
            "non-native sampled orderings without replay filtering. Counts and checksums "
            "are computed from the supplied JSONL files; they do not certify equivalence "
            "to a separately frozen paper reference set or ordering schedule. "
            "Annotations may include task-specific solution steps and require controlled "
            "distribution. Official GAIA access is required to reconstruct local inputs. "
            "The code-and-aggregate release does not include these annotations."
        ),
        "codeRepository": args.repo_url,
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "isBasedOn": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "measurementTechnique": "Native chain plus unique unfiltered non-native step permutations per task.",
        "numberOfItems": totals["tasks"],
        "additionalProperty": [
            {"@type": "sc:PropertyValue", "name": key, "value": value}
            for key, value in totals.items()
        ],
        "distribution": distribution,
        "recordSet": record_sets,
        "rai:dataCollectionMissingData": (
            "Official GAIA questions, final answers, and attachments must be obtained "
            "through the upstream access process. This metadata does not grant access."
        ),
        "rai:dataLimitations": (
            "Counts describe the supplied unfiltered candidate view. Reproducing paper "
            "results additionally requires the matching frozen references, ordering "
            "schedule, model versions, and run configuration."
        ),
    }
    if args.dataset_url:
        metadata["url"] = args.dataset_url
    if args.published_date:
        metadata["datePublished"] = args.published_date
    if args.license:
        metadata["license"] = args.license
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] Wrote metadata for {totals['tasks']} tasks and {totals['total_references']} references")
    for summary in summaries:
        print(f"[OK] {summary['name']}: {summary['native_references']} native, "
              f"{summary['non_native_references']} non-native")


if __name__ == "__main__":
    main()

