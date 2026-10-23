# Manuscript-aligned WWW 2027 aggregates

| Directory | Results |
|---|---|
| `main/` | Dataset profile, twenty-configuration three-facet headline, reference-score lift, reference budget, Web-source slice |
| `appendix/reference_robustness/` | Best-match distribution and auxiliary S1 alignment thresholds |
| `appendix/planning_answer/` | Stage/cohort, tool-clean, tool-intensive, and threshold diagnostics |
| `appendix/data_profile/` | Tool demand, reference accounting, precedence depth, candidate strata, dependency synthesis |
| `appendix/replay/` | Frozen Llama-4 and Gemma 4 coverage and outcome transitions |
| `appendix/human_dependency_audit/` | Pair-level majorities, anonymous rater totals, and task-level checks |
| `appendix/cross_benchmark/` | TaskBench and UltraTool auxiliary S1 results, five models and 1,000 tasks per benchmark |
| `appendix/closed_models/` | Eleven-model S3 planning, stage comparison, and chain-zero-credit sensitivity |

S3 is the primary GAIA planning stage; S1 rows are explicitly auxiliary. The
main reference set contains 165 Native chains and all 1,786 recorded candidates.
Replay and human review are separate diagnostics; they do not filter the main
results. The evolving third replay run is not inserted into this fixed paper
snapshot.

Planning and answers use 165 tasks per configuration. Tool scores use the same
144 tasks with gold tool slots. Open-weight answers use synthesized S5; closed
answers use the recorded submission or, when absent, recorded final response.
Accepted human judgments remain attached to that same response field.

`manifest.json` maps every CSV to its manuscript table label and caption.
All values have **manuscript display precision**. Blank multirow subset cells
are expanded into explicit subset labels. `group` records the displayed model
group or benchmark heading; it is empty when no group heading is present.
Confidence intervals and paired-count strings remain text fields. Aggregate
means retain their table weighting and are not inferred by averaging rounded
CSV rows.

Only aggregate tables are exported. Qualitative source-task examples, task-ID
prefixes, prompt templates, bibliographic tables, manuscript source files,
execution traces, and private provenance files are excluded.
