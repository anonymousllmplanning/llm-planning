#!/bin/bash
# Compatibility wrapper.
#
# This repository no longer downloads or mirrors GAIA raw data. GAIA must be
# obtained from the official gated Hugging Face dataset and rebuilt locally with
# scripts/prepare_gaia_from_official.py.

set -euo pipefail

cat <<'EOF'
This script is intentionally disabled.

GAIA raw questions, answers, and attachments are not redistributed from this
repository. Prepare local data with:

  python scripts/export_gaia_annotations.py \
    --source-root data/Augmented \
    --output-root annotations/gaia_annotations \
    --overwrite

  python scripts/prepare_gaia_from_official.py \
    --gaia-source /path/to/official/GAIA \
    --annotation-root annotations/gaia_annotations \
    --output-root data/Augmented \
    --overwrite

Cross-benchmark files can be materialized with:

  python scripts/prepare_crossbench.py \
    --source-root /path/to/crossbench_source \
    --output-root data

See README.md for the full artifact workflow.
EOF

exit 1
