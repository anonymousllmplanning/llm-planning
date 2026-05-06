#!/usr/bin/env python3
"""Rebuild the local Augmented GAIA layout from official GAIA plus annotations.

This script intentionally does not ship GAIA raw data. It expects the user to
obtain GAIA through the official gated Hugging Face dataset, then merges that
local snapshot with an Augmented GAIA annotation bundle exported by
``scripts/export_gaia_annotations.py``.

Output layout is compatible with ``scripts/exp.sh``:

  data/Augmented/
    cat_A_text/gaia.cat_A.json
    cat_B_document/{gaia.cat_B.json,attachments/}
    cat_C_vision/{gaia.cat_C.json,attachments/}
    cat_D_audio/{gaia.cat_D.json,attachments/}
    Asynchronous_output/        # non-native candidate async orderings, 1,771 rows total
    DAGs/                       # final Augmented GT scoring view, 1,522 reference rows
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.inference.prompts import get_available_tool_list
except Exception:  # pragma: no cover - fallback for minimal artifact envs
    get_available_tool_list = None


CAT_DIRS = {
    "A": "cat_A_text",
    "B": "cat_B_document",
    "C": "cat_C_vision",
    "D": "cat_D_audio",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
FINAL_GT_DIR = "Augmented_GT_NativePlusGemma4NonNative"
ANNOTATION_DIRS = ("Asynchronous_output", FINAL_GT_DIR)


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            handle.write("\n")


def record_id(record: dict[str, Any]) -> str:
    meta = record.get("meta") or record.get("metadata") or {}
    return str(meta.get("id") or meta.get("sample_id") or record.get("task_id") or record.get("id") or "")


def category_for_file(file_name: str | None) -> str:
    if not file_name:
        return "A"
    ext = Path(str(file_name)).suffix.lower()
    if ext in IMAGE_EXTS:
        return "C"
    if ext in AUDIO_EXTS:
        return "D"
    return "B"


def default_tool_environment() -> list[dict[str, Any]]:
    if get_available_tool_list is None:
        return []
    return get_available_tool_list()


def path_for_record(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_official_row(row: dict[str, Any], source_root: Path) -> dict[str, Any]:
    task_id = str(row.get("task_id") or row.get("Task ID") or row.get("id") or row.get("sample_id") or "")
    question = row.get("Question") or row.get("question") or row.get("query") or ""
    level = row.get("Level") or row.get("level") or ""
    final_answer = row.get("Final answer") or row.get("final_answer") or row.get("answer") or ""
    file_path = row.get("file_path") or row.get("File Path") or ""
    file_name = row.get("file_name") or row.get("file") or row.get("File") or ""
    if not file_name and file_path:
        file_name = Path(str(file_path)).name

    attachment_source = None
    if file_path:
        candidate = Path(str(file_path))
        if candidate.is_absolute() and candidate.exists():
            attachment_source = candidate
        else:
            candidate = source_root / str(file_path)
            if candidate.exists():
                attachment_source = candidate
    if attachment_source is None and file_name:
        matches = list(source_root.rglob(str(file_name)))
        if matches:
            attachment_source = matches[0]

    cat = category_for_file(file_name)
    return {
        "task_id": task_id,
        "question": str(question),
        "level": str(level).replace("level_", "").replace("Level ", ""),
        "final_answer": final_answer,
        "file_name": str(file_name) if file_name else "",
        "attachment_source": attachment_source,
        "cat": cat,
    }


def load_from_existing_augmented(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cat, dirname in CAT_DIRS.items():
        path = source_root / dirname / f"gaia.cat_{cat}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data:
            meta = record.get("meta") or {}
            query = record.get("query") or {}
            gold = record.get("gold") or {}
            final = (gold.get("final_answer") or {}).get("answer", "")
            attachments = query.get("attachments") or []
            file_name = ""
            attachment_source = None
            if attachments:
                file_name = attachments[0].get("file_name") or Path(attachments[0].get("file_path", "")).name
                raw_path = attachments[0].get("file_path") or ""
                candidates = [Path(raw_path), REPO_ROOT / raw_path, source_root / dirname / "attachments" / file_name]
                attachment_source = next((p for p in candidates if p.exists()), None)
            rows.append(
                {
                    "task_id": str(meta.get("id", "")),
                    "question": str(query.get("user_query", "")),
                    "level": str(meta.get("subset", "")).split("_")[1] if "_" in str(meta.get("subset", "")) else "",
                    "final_answer": final,
                    "file_name": file_name,
                    "attachment_source": attachment_source,
                    "cat": cat,
                }
            )
    return rows


def load_metadata_files(source_root: Path) -> list[dict[str, Any]]:
    existing = load_from_existing_augmented(source_root)
    if existing:
        return existing

    metadata_files = sorted(source_root.rglob("metadata.jsonl"))
    parquet_files = sorted(source_root.rglob("metadata.parquet"))
    rows: list[dict[str, Any]] = []

    for path in metadata_files:
        rows.extend(normalize_official_row(row, source_root) for row in read_jsonl(path))

    if parquet_files:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Reading GAIA parquet metadata requires pandas and pyarrow") from exc
        for path in parquet_files:
            frame = pd.read_parquet(path)
            rows.extend(normalize_official_row(row, source_root) for row in frame.to_dict("records"))

    if not rows:
        raise FileNotFoundError(
            f"No GAIA metadata found under {source_root}. Expected metadata.jsonl, "
            "metadata.parquet, or an existing Augmented GAIA cat_* layout."
        )

    return [row for row in rows if row.get("task_id")]


def load_annotation_rows(annotation_root: Path, dirname: str, fallbacks: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    folder = next((annotation_root / name for name in (dirname, *fallbacks) if (annotation_root / name).exists()), None)
    if folder is None:
        names = ", ".join((dirname, *fallbacks))
        raise FileNotFoundError(f"Missing annotation folder under {annotation_root}; expected one of: {names}")
    by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("gaia_cat_*_async_plan.jsonl")):
        for row in read_jsonl(path):
            rid = record_id(row)
            if rid:
                by_id[rid] = row
    return by_id


def rewrite_paths(value: Any, old_to_new: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for old, new in old_to_new.items():
            out = out.replace(old, new)
        return out
    if isinstance(value, list):
        return [rewrite_paths(item, old_to_new) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_paths(val, old_to_new) for key, val in value.items()}
    return value


def build_query(row: dict[str, Any], output_root: Path, repo_root: Path, copy_attachments: bool) -> tuple[dict[str, Any], dict[str, str]]:
    attachments: list[dict[str, str]] = []
    replacements: dict[str, str] = {}
    file_name = row.get("file_name")
    src = row.get("attachment_source")
    if file_name and src:
        cat_dir = CAT_DIRS[row["cat"]]
        dst = output_root / cat_dir / "attachments" / str(file_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_attachments:
            if not dst.exists() or sha256(Path(src)) != sha256(dst):
                shutil.copy2(src, dst)
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(Path(src).resolve(), dst)
        new_path = path_for_record(dst, repo_root)
        attachments.append({"file_name": str(file_name), "file_path": new_path})
        replacements[str(src)] = new_path
        replacements[Path(str(src)).name] = Path(new_path).name
        replacements[f"data/GAIA/{cat_dir}/attachments/{file_name}"] = new_path
        replacements[f"data/Augmented/{cat_dir}/attachments/{file_name}"] = new_path
        replacements[f"data/GAIA_tool_candidates_20260504/{cat_dir}/attachments/{file_name}"] = new_path
    return {"user_query": row["question"], "extra_instruction": "", "attachments": attachments}, replacements


def final_answer_obj(answer: Any) -> dict[str, Any]:
    return {"answer_type": "string", "answer": "" if answer is None else str(answer), "aliases": []}


def build_unified_record(
    official: dict[str, Any],
    annotation: dict[str, Any] | None,
    query: dict[str, Any],
    replacements: dict[str, str],
) -> dict[str, Any]:
    cat = official["cat"]
    task_id = official["task_id"]
    level = official.get("level") or ""
    gold = deepcopy((annotation or {}).get("gold") or {})
    if gold:
        gold = rewrite_paths(gold, replacements)
    gold["final_answer"] = final_answer_obj(official.get("final_answer"))
    return {
        "meta": {
            "dataset": "gaia",
            "subset": f"level_{level}_{cat}" if level else f"cat_{cat}",
            "split": "validation",
            "id": task_id,
            "plan_type": "chain",
            "has_arguments": True,
            "has_answer": True,
        },
        "query": query,
        "tool_environment": default_tool_environment(),
        "gold": gold,
    }


def merge_annotation_record(
    annotation: dict[str, Any],
    official: dict[str, Any],
    query: dict[str, Any],
    replacements: dict[str, str],
) -> dict[str, Any]:
    out = rewrite_paths(deepcopy(annotation), replacements)
    out["query"] = query
    meta = dict(out.get("meta") or {})
    meta.update(
        {
            "dataset": "gaia",
            "split": "validation",
            "id": official["task_id"],
            "has_answer": True,
        }
    )
    out["meta"] = meta
    gold = dict(out.get("gold") or {})
    gold["final_answer"] = final_answer_obj(official.get("final_answer"))
    out["gold"] = gold
    return out


def copy_tree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_filtered_summary_files(annotation_root: Path, output_root: Path) -> None:
    src_dir = next(
        (
            annotation_root / name
            for name in (FINAL_GT_DIR, "DAGs", "Gemma4_Filtered_DAGs")
            if (annotation_root / name).exists()
        ),
        None,
    )
    if src_dir is None:
        return
    for name in ("README.md", "gemma4_filtered_dag_summary.json", "gemma4_filtered_dag_task_summary.csv"):
        src = src_dir / name
        if src.exists():
            dst = output_root / "DAGs" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_checksums(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "checksums.sha256")
    with (root / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")


def maybe_download_gaia(repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Pass --gaia-source or install huggingface_hub for --download") from exc
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaia-source", type=Path, help="Official GAIA snapshot root or existing local Augmented root")
    parser.add_argument("--download", action="store_true", help="Download gaia-benchmark/GAIA with huggingface_hub")
    parser.add_argument("--repo-id", default="gaia-benchmark/GAIA")
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/Augmented")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--symlink-attachments", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.download:
        gaia_source = maybe_download_gaia(args.repo_id)
    elif args.gaia_source:
        gaia_source = args.gaia_source
    else:
        raise SystemExit("Provide --gaia-source or --download")

    gaia_source = gaia_source.resolve()
    annotation_root = args.annotation_root.resolve()
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    official_rows = load_metadata_files(gaia_source)
    official_by_id = {row["task_id"]: row for row in official_rows}
    async_by_id = load_annotation_rows(annotation_root, "Asynchronous_output")
    final_gt_by_id = load_annotation_rows(
        annotation_root,
        FINAL_GT_DIR,
        fallbacks=("DAGs", "Gemma4_Filtered_DAGs"),
    )

    categorized: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CAT_DIRS}
    query_by_id: dict[str, dict[str, Any]] = {}
    replacements_by_id: dict[str, dict[str, str]] = {}

    for task_id, official in sorted(official_by_id.items()):
        query, replacements = build_query(
            official,
            output_root=output_root,
            repo_root=repo_root,
            copy_attachments=not args.symlink_attachments,
        )
        query_by_id[task_id] = query
        replacements_by_id[task_id] = replacements
        annotation = final_gt_by_id.get(task_id) or async_by_id.get(task_id)
        categorized[official["cat"]].append(build_unified_record(official, annotation, query, replacements))

    for cat, rows in categorized.items():
        out = output_root / CAT_DIRS[cat] / f"gaia.cat_{cat}.json"
        rows.sort(key=lambda rec: rec["meta"]["id"])
        write_json(out, rows)

    for dirname, by_id in (("Asynchronous_output", async_by_id), ("DAGs", final_gt_by_id)):
        grouped: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CAT_DIRS}
        for task_id, annotation in by_id.items():
            official = official_by_id.get(task_id)
            if not official:
                continue
            grouped[official["cat"]].append(
                merge_annotation_record(
                    annotation,
                    official,
                    query_by_id[task_id],
                    replacements_by_id.get(task_id, {}),
                )
            )
        for cat, rows in grouped.items():
            rows.sort(key=record_id)
            write_jsonl(output_root / dirname / f"gaia_cat_{cat}_async_plan.jsonl", rows)

    copy_filtered_summary_files(annotation_root, output_root)

    manifest = {
        "gaia_source": str(gaia_source),
        "annotation_root": str(annotation_root),
        "output_root": str(output_root),
        "scheme": (
            "DAGs is the final Augmented GT scoring view: one native chain "
            "reference per task plus Gemma 4-retained non-native async orderings"
        ),
        "counts": {
            "tasks": sum(len(v) for v in categorized.values()),
            "cat_A": len(categorized["A"]),
            "cat_B": len(categorized["B"]),
            "cat_C": len(categorized["C"]),
            "cat_D": len(categorized["D"]),
        },
    }
    write_json(output_root / "build_manifest.json", manifest)
    write_checksums(output_root)

    def count_orderings(folder: str) -> int:
        total = 0
        for path in (output_root / folder).glob("gaia_cat_*_async_plan.jsonl"):
            for row in read_jsonl(path):
                total += len(row.get("sampled_orderings") or [])
        return total

    print(f"[OK] Wrote {output_root}")
    print(f"[OK] Tasks: {manifest['counts']['tasks']}")
    print(f"[OK] Asynchronous_output orderings: {count_orderings('Asynchronous_output')}")
    print(f"[OK] DAGs final reference orderings: {count_orderings('DAGs')}")


if __name__ == "__main__":
    main()
