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
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


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

    manifest: dict[str, Any] = {
        "source_root": "data/Augmented",
        "strip_top_level_query": True,
        "strip_gold_final_answer": not args.keep_final_answer,
        "annotation_dirs": [dst for _, dst in ANNOTATION_DIRS],
        "source_annotation_dirs": {dst: src for src, dst in ANNOTATION_DIRS},
        "files": [],
    }

    for source_dirname, output_dirname in ANNOTATION_DIRS:
        src_dir = source_root / source_dirname
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing annotation directory: {src_dir}")
        for src in sorted(src_dir.glob("gaia_cat_*_async_plan.jsonl")):
            rel = Path(output_dirname) / src.name
            dst = output_root / rel
            rows = (
                strip_raw_gaia_fields(row, strip_final_answer=not args.keep_final_answer)
                for row in read_jsonl(src)
            )
            write_jsonl(dst, rows)
            manifest["files"].append(rel.as_posix())

    for source_rel_text, output_rel_text in SUMMARY_FILES:
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
        "The final scoring view is `Augmented_GT_NativePlusGemma4NonNative`: one\n"
        "native chain reference per GAIA task plus Gemma 4-retained non-native\n"
        "async orderings. Rebuilding maps this view to `data/Augmented/DAGs` for\n"
        "compatibility with the experiment scripts.\n\n"
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
