#!/usr/bin/env python3
"""Export Augmented GAIA annotation bundles without GAIA raw task files.

The local ``data/Augmented`` folder is a reconstructed working directory: it
contains GAIA questions, final answers, and attachments and should not be
committed to a public repository. This exporter writes the annotation sidecar
needed by ``prepare_gaia_from_official.py`` while stripping the top-level GAIA
query and final-answer fields.

The exported planning/tool annotations may still contain task-specific solution
step text. Treat the resulting bundle as a controlled review artifact unless a
separate policy review decides it is safe for public redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.evaluation.ordering_references import load_ordering_schedule, ordering_references


ANNOTATION_DIRS = (
    ("GPT4o_NonNative_Orderings", "Asynchronous_output"),
    ("Augmented_GT_NativePlusGemma4NonNative", "Augmented_GT_NativePlusGemma4NonNative"),
)
SUMMARY_FILES = (
    ("GPT4o_NonNative_Orderings/README.md", "Asynchronous_output/README.md"),
    ("GPT4o_NonNative_Orderings/summary.json", "Asynchronous_output/summary.json"),
    ("GPT4o_NonNative_Orderings/category_summary.csv", "Asynchronous_output/category_summary.csv"),
    ("Augmented_GT_NativePlusGemma4NonNative/README.md", "Augmented_GT_NativePlusGemma4NonNative/README.md"),
    ("Augmented_GT_NativePlusGemma4NonNative/summary.json", "Augmented_GT_NativePlusGemma4NonNative/summary.json"),
    ("Augmented_GT_NativePlusGemma4NonNative/category_summary.csv", "Augmented_GT_NativePlusGemma4NonNative/category_summary.csv"),
    ("Augmented_GT_NativePlusGemma4NonNative/task_summary.csv", "Augmented_GT_NativePlusGemma4NonNative/task_summary.csv"),
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def strip_raw_gaia_fields(row: dict[str, Any], strip_final_answer: bool) -> dict[str, Any]:
    out = deepcopy(row)
    out.pop("query", None)
    gold = out.get("gold")
    if isinstance(gold, dict) and strip_final_answer:
        gold.pop("final_answer", None)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    with (root / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/Augmented"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--keep-final-answer",
        action="store_true",
        help="Keep gold.final_answer in exported rows. Not recommended for public artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-root if it already exists.",
    )
    parser.add_argument("--reference-view", choices=("current", "legacy-filtered"), default="current",
                        help="current exports source candidates; legacy-filtered includes an explicitly selected retained annotation view")
    parser.add_argument("--candidate-dir", type=Path, default=None,
                        help="Controlled unfiltered annotation directory; relative to --source-root unless absolute")
    parser.add_argument("--legacy-dir", type=Path, default=None,
                        help="Controlled retained annotation directory for --reference-view legacy-filtered")
    parser.add_argument("--ordering-schedule", type=Path, default=None,
                        help="Authoritative JSON/JSONL recorded schedule; copied separately and validated against native nodes")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Missing source Augmented GAIA root: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_dir = args.candidate_dir
    if candidate_dir is None:
        candidate_dir = next((Path(name) for name in ("Asynchronous_output", "GPT4o_NonNative_Orderings")
                              if (source_root / name).is_dir()), Path("Asynchronous_output"))
    annotation_dirs = [(str(candidate_dir), "Asynchronous_output")]
    if args.reference_view == "legacy-filtered":
        annotation_dirs.append((str(args.legacy_dir or "Augmented_GT_NativePlusGemma4NonNative"),
                                "Augmented_GT_LegacyFiltered"))
    if args.ordering_schedule and args.reference_view != "current":
        raise ValueError("--ordering-schedule requires --reference-view current")
    schedule_path = args.ordering_schedule
    if schedule_path is None and args.reference_view == "current":
        default_schedule = source_root / "ordering_schedule.json"
        if default_schedule.exists():
            schedule_path = default_schedule
    schedule = load_ordering_schedule(schedule_path) if schedule_path else None
    exported_tasks = set()

    manifest: dict[str, Any] = {
        "source_root": "data/Augmented",
        "strip_top_level_query": True,
        "strip_gold_final_answer": not args.keep_final_answer,
        "ordering_view": args.reference_view,
        "annotation_dirs": [dst for _, dst in annotation_dirs],
        "source_annotation_dirs": {dst: src for src, dst in annotation_dirs},
        "files": [],
    }

    for source_dirname, output_dirname in annotation_dirs:
        src_dir = source_root / source_dirname
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing annotation directory: {src_dir}")
        for src in sorted(src_dir.glob("gaia_cat_*_async_plan.jsonl")):
            rel = Path(output_dirname) / src.name
            dst = output_root / rel
            rows = []
            for row in read_jsonl(src):
                task_id = str((row.get("meta") or row.get("metadata") or {}).get("id") or row.get("task_id") or row.get("id") or "")
                if not task_id or task_id in exported_tasks:
                    raise ValueError("Annotation task IDs must be present and unique within a view")
                exported_tasks.add(task_id)
                clean = strip_raw_gaia_fields(row, strip_final_answer=not args.keep_final_answer)
                clean.setdefault("metadata", {})["ordering_view"] = args.reference_view
                if schedule is not None:
                    if task_id not in schedule:
                        raise ValueError(f"Recorded schedule is missing annotation task {task_id}")
                    ordering_references(row, schedule[task_id])
                rows.append(clean)
            write_jsonl(dst, rows)
            manifest["files"].append(rel.as_posix())

        if not list(src_dir.glob("gaia_cat_*_async_plan.jsonl")):
            raise FileNotFoundError(f"No annotation JSONL files in {src_dir}")
        if schedule is not None and set(schedule) != exported_tasks:
            raise ValueError("Recorded schedule and annotation task IDs must match exactly")
        exported_tasks.clear()

    if schedule is not None:
        schedule_path = output_root / "ordering_schedule.json"
        schedule_path.write_text(json.dumps([row for refs in schedule.values() for row in refs], indent=2) + "\n", encoding="utf-8")
        manifest["files"].append("ordering_schedule.json")
        manifest["recorded_schedule"] = {"path": "ordering_schedule.json", "tasks": len(schedule),
                                          "native": sum(ref["kind"] == "native" for refs in schedule.values() for ref in refs),
                                          "candidates": sum(ref["kind"] == "candidate" for refs in schedule.values() for ref in refs)}

    # Copy retained-view summaries only when explicitly requested.
    for source_rel_text, output_rel_text in SUMMARY_FILES if args.reference_view == "legacy-filtered" else ():
        src = source_root / source_rel_text
        if not src.exists():
            continue
        dst = output_root / output_rel_text
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest["files"].append(output_rel_text)

    readme = output_root / "README.md"
    readme.write_text(
        "# Augmented GAIA Annotation Bundle\n\n"
        "This bundle omits GAIA questions, final answers, and attachments. It is\n"
        "intended to be merged with an official GAIA snapshot by\n"
        "`scripts/prepare_gaia_from_official.py`.\n\n"
        f"Selected reference view: `{args.reference_view}`. Current evaluation uses\n"
        "the native chain plus each recorded candidate-order chain separately.\n"
        "Use the recorded schedule with `--gaia_ordering_schedule`; the compiled\n"
        "dependency graph constrains generation and is not a scoring reference.\n\n"
        "Caution: planning step labels and tool annotations are derived benchmark\n"
        "annotations and may reveal task-specific solution structure. Distribute\n"
        "this bundle through the controlled artifact channel chosen for release.\n",
        encoding="utf-8",
    )
    manifest["files"].append("README.md")

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_checksums(output_root)

    print(f"[OK] Wrote annotation bundle: {output_root}")
    print("[OK] Top-level query and gold.final_answer stripped by default")


if __name__ == "__main__":
    main()
