# Core aggregate results

[WWW 2027 tables](www2027/README.md) contain the displayed main and supporting
results from the manuscript: **10 tables and 87 aggregate rows**. They cover
eleven closed-model and nine open-weight configurations, reference sensitivity,
planning/answer associations, Web-source results, and TaskBench/UltraTool.

The shared scoring pool has **1,594 references**: 165 Native chains and 1,429
replay-retained alternatives from 1,786 sampled candidates. See the
[reference guide](../references/README.md) and
[count ledger](../references/reference_counts.json).

CSV cells preserve manuscript display precision. Confidence intervals and
percentages retain their table units. Differences were computed before display
rounding; do not recompute them from rounded endpoints. The manifest records
table labels, captions, source snapshot hashes, columns, row counts, and file
checksums without local source paths. Aggregate tables contain no task-level
questions, answers, annotator identities, or model traces.

To rebuild displayed table exports from separately supplied manuscript files:

```bash
python scripts/export_paper_results.py --main /path/to/main.tex \
  --appendix /path/to/appendix.tex --output /path/to/new-export
```

The [final overview](../docs/assets/overview_figure_final_1594.png),
[reference coverage](../docs/assets/reference_coverage_1786_1429.png), and
[dependency-synthesis pipeline](../docs/assets/pipeline_figure.png) accompany
the results. Published code and result archives exclude manuscript sources.
