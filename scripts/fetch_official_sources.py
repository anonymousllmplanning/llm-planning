#!/usr/bin/env python3
"""Fetch official benchmark sources into an untracked local directory.

This helper is intentionally separate from conversion. It downloads or clones
the upstream sources required by the local rebuild flow under the original
licenses and terms, then the prepare scripts materialize the local `data/`
layout used by `scripts/exp.sh`.

GAIA is gated and must be accessed through the official Hugging Face dataset.
Do not commit the downloaded snapshot or any regenerated GAIA task files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULTS = {
    "gaia": {
        "kind": "hf_dataset",
        "repo_id": "gaia-benchmark/GAIA",
        "license_note": "Official gated GAIA terms; do not reshare validation/test in crawlable form.",
    },
    "taskbench": {
        "kind": "hf_dataset",
        "repo_id": "microsoft/Taskbench",
        "license_note": "MIT as listed on the Hugging Face dataset card.",
    },
    "ultratool": {
        "kind": "git",
        "repo_url": "https://github.com/JoeYing1019/UltraTool.git",
        "license_note": "Apache-2.0 as listed in the upstream GitHub repository.",
    },
}


def snapshot_download(repo_id: str, dst: Path, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to fetch Hugging Face datasets") from exc

    dst.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            local_dir=dst,
            local_dir_use_symlinks=False,
        )
    )


def git_clone(repo_url: str, dst: Path, overwrite: bool) -> Path:
    if dst.exists():
        if not overwrite:
            return dst
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dst)], check=True)
    return dst


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sources": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["gaia", "taskbench", "ultratool", "all"],
        default=[],
        help="Dataset to fetch. Repeatable. Default: all.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("raw_sources"))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = args.dataset or ["all"]
    if "all" in selected:
        selected = ["gaia", "taskbench", "ultratool"]

    rows: list[dict[str, Any]] = []
    for name in selected:
        spec = DEFAULTS[name]
        dst = args.output_root / name
        row: dict[str, Any] = {
            "name": name,
            "kind": spec["kind"],
            "destination": str(dst),
            "license_note": spec["license_note"],
        }
        if spec["kind"] == "hf_dataset":
            row["repo_id"] = spec["repo_id"]
            if args.dry_run:
                row["status"] = "dry-run"
            else:
                if dst.exists() and not args.overwrite:
                    row["status"] = "exists"
                else:
                    actual = snapshot_download(spec["repo_id"], dst, token=args.hf_token)
                    row["status"] = "downloaded"
                    row["actual_path"] = str(actual)
        elif spec["kind"] == "git":
            row["repo_url"] = spec["repo_url"]
            if args.dry_run:
                row["status"] = "dry-run"
            else:
                actual = git_clone(spec["repo_url"], dst, overwrite=args.overwrite)
                row["status"] = "cloned" if actual == dst else "exists"
        rows.append(row)

    manifest = args.output_root / "source_manifest.json"
    if not args.dry_run:
        write_manifest(manifest, rows)
    print(json.dumps({"manifest": str(manifest), "sources": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
