# Annotation Artifacts

The public GitHub repository intentionally does not track GAIA raw data or the
local `data/` build output.

For reproducibility, use a controlled annotation bundle exported with:

```bash
python scripts/export_gaia_annotations.py \
  --source-root data/Augmented \
  --output-root /path/to/controlled/gaia_annotations \
  --overwrite
```

Then rebuild the local GAIA working tree from the official gated GAIA snapshot:

```bash
python scripts/prepare_gaia_from_official.py \
  --gaia-source /path/to/official/GAIA \
  --annotation-root /path/to/controlled/gaia_annotations \
  --output-root data/Augmented \
  --overwrite
```

The annotation bundle omits top-level GAIA questions, final answers, and
attachments. Planning step labels and tool annotations are derived benchmark
annotations and may expose task-specific solution structure, so distribute them
through the controlled artifact channel chosen for release rather than a
crawlable public repository.
