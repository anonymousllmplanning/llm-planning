# Optional Improvement Diagnostics

Generated from the paper result artifacts and replay-filtered DAG files.
No raw GAIA questions, answers, or attachments are used in the outputs.

## 1. Best-Match Reference Breakdown

Definition: a model-task pair is counted as `augmented_helped` when the
augmented best score has a strictly higher EdgeF1 than the native-chain
reference. For open-weight rows, the public paper summary only stores
PlanningScore, so EdgeF1 lift is recovered as
`2 * (augmented_best_planning_score - chain_only_planning_score)` because
NodeF1 is invariant between native and augmented references by construction.

Main open-weight pool: 75/378
multi-order-rich pairs (19.8%) improve under an
augmented ordering rather than tying the native chain.

Closed-model extension: 76/456
multi-order-rich pairs (16.7%) improve under
an augmented ordering.

Across all 15 models: 151/834
pairs (18.1%) improve under an augmented
ordering.

Suggested main-text sentence:

> On the multi-order-rich subset, 19.8% of
> open-weight task-model pairs are best matched by an augmented ordering rather
> than the native chain, so the lift is concentrated in genuinely parallel
> tasks rather than spread uniformly across pairs.

## 2. Critical Path vs. Native Chain Length

Definition: native chain length is `metadata.original_step_count`; critical
path length is the longest path in the conservative partial order induced by
the final Augmented GAIA reference orderings. The parallelism ratio is
`1 - critical_path_length / native_chain_length`.

All 165 tasks: mean native chain length 7.78, mean
critical path length 6.45, mean parallelism ratio
0.116.

Multi-order-rich tasks: mean native chain length 10.80,
mean critical path length 6.81, mean parallelism
ratio 0.339.

Suggested appendix sentence:

> Across the 165 GAIA tasks, reference-validated orderings reduce the mean
> path from a native chain length of 7.78 steps to a
> critical path of 6.45 steps. The effect is strongest
> on the 54 multi-order-rich tasks, where the mean critical path is
> 6.81 versus a mean native chain length of
> 10.80, corresponding to a mean parallelism ratio of
> 33.9%.

## Output Files

- `best_match_reference_breakdown_by_pool.csv`
- `best_match_reference_breakdown_by_model.csv`
- `best_match_reference_pairs.csv`
- `task_parallelism_profile.csv`
- `parallelism_summary_by_bucket.csv`
- `optional_improvements_summary.json`
- `optional_improvements_snippets.tex`
