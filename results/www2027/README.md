# Final manuscript-aligned aggregate results

These ten tables contain 87 aggregate rows and preserve the displayed values
from the manuscript. The main manuscript contains its appendix; the separate appendix
copy is checked for consistent duplicate tables during export.

| Directory | Contents |
|---|---|
| `main/` | Dataset profile, twenty-configuration headline, fixed-cohort gains, reference budget, planning/answer correlations, and Web-source results |
| `appendix/reference_robustness/` | Best-match outcomes and alignment-threshold sensitivity |
| `appendix/planning_answer/` | Tool-threshold sensitivity |
| `appendix/cross_benchmark/` | Five-model TaskBench and UltraTool planning diagnostics |

The final pool is 165 Native plus 1,429 retained alternative references (1,594
total). Every configuration uses the same pool. The original 63-task rich
comparison cohort is fixed; the retained-rich reference-budget cohort has 56
tasks. See [the reference guide](../../references/README.md) for the
successful-replay rule, count ledger, and source files.

`manifest.json` maps each CSV to its final manuscript table label, caption,
source snapshot hash, displayed-value fingerprints, and file checksum.
`SHA256SUMS` covers the table files and manifest. Export and validation do not
rerun inference, reconstruct unreported precision, or independently certify
the supplied semantic answer-matching decisions.
