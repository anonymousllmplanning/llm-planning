---
pretty_name: Augmented GAIA Planning Evaluation
viewer: false
license: cc-by-4.0
tags:
  - evaluation
  - agentic-planning
  - tool-use
  - gaia
  - taskbench
  - ultratool
  - croissant
---

# NeurIPS 2026 Evaluation Dataset Artifact

This folder mirrors the anonymous evaluation-dataset artifact. The released
derived annotations and packaging metadata are licensed under CC-BY-4.0. It
intentionally contains metadata, archives, checksums, and packaging
instructions, not GAIA raw validation/test questions, final answers, or
attachments.

## Contents

- `croissant.json`: Croissant metadata with Responsible AI fields.
- `PACKAGING.md`: rebuild and smoke-test procedure.
- `annotations/gaia_annotations.zip`: controlled Augmented GAIA annotation
  sidecar.
- `results/results_summaries.zip`: sanitized aggregate result summaries.
- `scripts/scripts.zip`: rebuild and evaluation scripts.
- `package_manifest.json`: byte sizes and SHA-256 checksums for the released
  files.

GAIA raw data must be obtained from the official gated Hugging Face dataset and
rebuilt locally.

The Hugging Face Dataset Viewer is intentionally disabled because this artifact
is not a single tabular dataset split. It is a controlled evaluation artifact
containing archives, scripts, checksums, and Croissant metadata.

## Entry Points

- Croissant metadata: `croissant.json`
- Rebuild instructions: `PACKAGING.md`
- Controlled annotation bundle: `annotations/gaia_annotations.zip`
- Sanitized aggregate results: `results/results_summaries.zip`
- Rebuild/evaluation scripts: `scripts/scripts.zip`
