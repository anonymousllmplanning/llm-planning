# Core aggregate results

[WWW 2027 tables](www2027/README.md) contain the main and supporting numeric
results used in the current manuscript. The export includes 26 tables and
183 aggregate rows, including eleven closed-model and nine open-weight
configurations, reference sensitivity, replay coverage, human dependency review,
and the TaskBench/UltraTool extension.

CSV cells preserve the values and precision displayed in the manuscript.
These are aggregate table exports, not full-precision per-task measurements.
Confidence intervals and percentages retain their table units. Differences were
computed before display rounding; do not recompute them from rounded endpoints.

The manifest records table labels, captions, source snapshot hashes, column
names, row counts, and file checksums. It contains no local source paths.
No questions, answers, task identifiers, annotator identities, or individual
model traces are included. Rebuild the export from supplied manuscript files:

```bash
python scripts/export_paper_results.py --main /path/to/main.tex \
  --appendix /path/to/appendix.tex --output /path/to/new-export
```

The current [framework diagram](../docs/assets/framework_overview.png) uses
1,786 candidate orderings and 1,951 total references. Its editable vector
versions and rendering code are included.
