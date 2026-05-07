#!/usr/bin/env python3
"""Prepare local TaskBench and UltraTool files without tracking data in Git.

The public repository keeps scripts and checksums, not generated dataset files.
This helper materializes the two cross-benchmark files used by the paper:

* ``data/Taskbench/unified_taskbench_order_chain500_dag500.jsonl``
* ``data/Ultratool/unified_ultratool_en_1000.jsonl``

It can copy those files from a controlled local artifact, derive them from
larger already-converted unified files, or convert the official raw layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.conversion.converter import convert_taskbench, convert_ultratool  # noqa: E402


TASKBENCH_TARGET = "Taskbench/unified_taskbench_order_chain500_dag500.jsonl"
ULTRATOOL_TARGET = "Ultratool/unified_ultratool_en_1000.jsonl"
TASKBENCH_ALL = "Taskbench/unified_taskbench_all.jsonl"
ULTRATOOL_ALL_EN = "Ultratool/unified_ultratool_en.jsonl"


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


def build_taskbench_subset(unified_all: Path, output: Path, seed: int, chain_n: int, dag_n: int) -> int:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(unified_all):
        plan_type = str((row.get("meta") or {}).get("plan_type") or "").lower()
        if plan_type in {"chain", "dag"}:
            buckets[plan_type].append(row)
    rng = random.Random(seed)
    selected: list[dict] = []
    for plan_type, needed in (("chain", chain_n), ("dag", dag_n)):
        rows = list(buckets[plan_type])
        if len(rows) < needed:
            raise ValueError(f"TaskBench {plan_type} has {len(rows)} rows, fewer than requested {needed}")
        rng.shuffle(rows)
        selected.extend(rows[:needed])
    selected.sort(key=lambda row: str((row.get("meta") or {}).get("id", "")))
    return write_jsonl(output, selected)


def first_n_jsonl(src: Path, dst: Path, limit: int) -> int:
    def rows():
        for i, row in enumerate(read_jsonl(src)):
            if i >= limit:
                break
            yield row
    return write_jsonl(dst, rows())


def find_existing(root: Path, relative: str, alternatives: list[str]) -> Path | None:
    candidates = [root / relative] + [root / alt for alt in alternatives]
    return next((path for path in candidates if path.exists()), None)


def find_taskbench_raw_dirs(root: Path) -> list[Path]:
    """Find official TaskBench subset folders containing the expected raw files."""
    required = {"data.json", "graph_desc.json", "tool_desc.json"}
    dirs = []
    for data_file in sorted(root.rglob("data.json")):
        parent = data_file.parent
        names = {p.name for p in parent.iterdir() if p.is_file()}
        has_requests = "user_requests.json" in names or "user_requests.jsonl" in names
        if required.issubset(names) and has_requests:
            dirs.append(parent)
    return dirs


def taskbench_subset_name(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("data_"):
        name = name.removeprefix("data_")
    return {
        "dailylifeapis": "dailylifeapis",
        "huggingface": "huggingface",
        "multimedia": "multimedia",
    }.get(name, name)


def build_taskbench_unified_from_raw(raw_dirs: list[Path], output: Path, no_alignment_filter: bool) -> int:
    records = []
    for raw_dir in raw_dirs:
        subset = taskbench_subset_name(raw_dir)
        records.extend(
            convert_taskbench(
                raw_dir,
                subset_name=subset,
                limit=None,
                use_alignment_ids=not no_alignment_filter,
            )
        )
    if not records:
        raise ValueError(f"No TaskBench records converted from: {raw_dirs}")
    records.sort(key=lambda row: (str((row.get("meta") or {}).get("subset", "")), str((row.get("meta") or {}).get("id", ""))))
    return write_jsonl(output, records)


def find_ultratool_raw_dir(root: Path) -> Path | None:
    required = {"test.json", "tool_usage.json", "tool_usage_awareness.json", "tool_creation_awareness.json"}
    for test_file in sorted(root.rglob("test.json")):
        parent = test_file.parent
        if required.issubset({p.name for p in parent.iterdir() if p.is_file()}):
            return parent
    return None


def build_ultratool_unified_from_raw(raw_dir: Path, output: Path) -> int:
    records = convert_ultratool(raw_dir, limit=None)
    if not records:
        raise ValueError(f"No UltraTool records converted from: {raw_dir}")
    records.sort(key=lambda row: str((row.get("meta") or {}).get("id", "")))
    return write_jsonl(output, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--taskbench-chain", type=int, default=500)
    parser.add_argument("--taskbench-dag", type=int, default=500)
    parser.add_argument("--ultratool-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        choices=("both", "taskbench", "ultratool"),
        default="both",
        help="Materialize only one auxiliary benchmark when the source root contains just that upstream dataset.",
    )
    parser.add_argument(
        "--taskbench-raw-dirs",
        type=Path,
        nargs="*",
        help="Optional official TaskBench raw subset dirs. If omitted, the script searches source-root.",
    )
    parser.add_argument(
        "--taskbench-no-alignment-filter",
        action="store_true",
        help="Convert all TaskBench raw rows instead of applying alignment_ids.json when present.",
    )
    parser.add_argument(
        "--ultratool-raw-dir",
        type=Path,
        help="Optional UltraTool English test_set raw dir. If omitted, the script searches source-root.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest: dict[str, dict] = {}

    if args.only in {"both", "taskbench"}:
        taskbench_out = output_root / TASKBENCH_TARGET
        taskbench_existing = find_existing(
            source_root,
            TASKBENCH_TARGET,
            ["Taskbench/unified_taskbench_all.jsonl"],
        )
        if taskbench_existing is None:
            raw_dirs = [p.resolve() for p in args.taskbench_raw_dirs or []]
            if not raw_dirs:
                raw_dirs = find_taskbench_raw_dirs(source_root)
            if not raw_dirs:
                raise FileNotFoundError(
                    "Could not find TaskBench unified or raw source. Provide a source root containing "
                    f"{TASKBENCH_TARGET}, {TASKBENCH_ALL}, or official raw subset folders with "
                    "data.json / graph_desc.json / user_requests.jsonl / tool_desc.json."
                )
            taskbench_all = output_root / TASKBENCH_ALL
            build_taskbench_unified_from_raw(
                raw_dirs,
                taskbench_all,
                no_alignment_filter=args.taskbench_no_alignment_filter,
            )
            taskbench_existing = taskbench_all
        if taskbench_existing.name == "unified_taskbench_all.jsonl":
            rows = build_taskbench_subset(
                taskbench_existing,
                taskbench_out,
                seed=args.seed,
                chain_n=args.taskbench_chain,
                dag_n=args.taskbench_dag,
            )
        else:
            copy_file(taskbench_existing, taskbench_out)
            rows = sum(1 for _ in taskbench_out.open("r", encoding="utf-8"))
        manifest[TASKBENCH_TARGET] = {"rows": rows, "sha256": sha256(taskbench_out)}

    if args.only in {"both", "ultratool"}:
        ultratool_out = output_root / ULTRATOOL_TARGET
        ultratool_existing = find_existing(
            source_root,
            ULTRATOOL_TARGET,
            ["Ultratool/unified_ultratool_en.jsonl"],
        )
        if ultratool_existing is None:
            raw_dir = args.ultratool_raw_dir.resolve() if args.ultratool_raw_dir else find_ultratool_raw_dir(source_root)
            if raw_dir is None:
                raise FileNotFoundError(
                    "Could not find UltraTool unified or raw source. Provide a source root containing "
                    f"{ULTRATOOL_TARGET}, {ULTRATOOL_ALL_EN}, or an official English test_set folder "
                    "with test.json / tool_usage*.json files."
                )
            ultratool_all = output_root / ULTRATOOL_ALL_EN
            build_ultratool_unified_from_raw(raw_dir, ultratool_all)
            ultratool_existing = ultratool_all
        if ultratool_existing.name == "unified_ultratool_en.jsonl":
            rows = first_n_jsonl(ultratool_existing, ultratool_out, args.ultratool_limit)
        else:
            copy_file(ultratool_existing, ultratool_out)
            rows = sum(1 for _ in ultratool_out.open("r", encoding="utf-8"))
        manifest[ULTRATOOL_TARGET] = {"rows": rows, "sha256": sha256(ultratool_out)}

    manifest_path = output_root / "crossbench_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    for relative in manifest:
        print(f"[OK] Wrote {output_root / relative}")
    print(f"[OK] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
