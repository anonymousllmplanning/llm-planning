# Packaging Checklist

## Source Policy

- Do not upload GAIA validation/test questions, final answers, or attachments in
  a crawlable format.
- Upload only the controlled Augmented GAIA annotation bundle, sanitized result
  summaries, scripts, checksums, and Croissant metadata.
- Obtain GAIA from `gaia-benchmark/GAIA` after accepting the
  official access conditions.

## Rebuild Commands

```bash
python scripts/fetch_official_sources.py \
  --dataset gaia \
  --dataset taskbench \
  --dataset ultratool \
  --output-root raw_sources

python scripts/prepare_gaia_from_official.py \
  --gaia-source raw_sources/gaia \
  --annotation-root annotations/gaia_annotations \
  --output-root data/Augmented \
  --overwrite

python scripts/prepare_crossbench.py \
  --source-root raw_sources \
  --output-root data
```

`prepare_gaia_from_official.py` writes `data/Augmented/DAGs` as the final
Augmented GT scoring view: 165 native chain references plus 1,357
Gemma 4-retained non-native async orderings, for 1,522 total reference ordering
rows. This keeps the default `scripts/exp.sh` path unchanged.

## Smoke Tests

```bash
bash scripts/exp.sh \
  --dataset gaia_cat_A \
  --dataset taskbench \
  --dataset ultratool_en_1000 \
  --mode order \
  --backend api \
  --provider-profile openai \
  --model gpt-5.5 \
  --limit 1 \
  --dry-run

bash scripts/exp.sh \
  --dataset gaia_cat_A \
  --mode answer \
  --backend api \
  --provider-profile openai \
  --model gpt-5.5 \
  --limit 1 \
  --max-turns 1 \
  --dry-run
```

For live API smoke tests, set a fresh API key in the shell environment. Do not
commit keys or paste them into command logs.
