# Final planning references

The final reference set contains 165 Native chains and 1,429 retained alternatives
(1,594 references in total), selected from 1,786 sampled candidates.

- `reference_counts.json`: final count ledger, retention rate, cohort sizes,
  and hashes of the three reference files below.
- `replay_retained_orderings.json`: retained candidate IDs, permutations,
  supporting executors, and the final retention and answer-matching policy.
- `reference_set_1594.csv`: all Native and retained alternatives, with source
  node/graph hashes and recorded precedence-violation counts.
- `ordering_schedule.json`: the same CSV identities and permutations converted
  to the JSON schedule consumed by `src.evaluation.ordering_references`.

At least one of GPT-6 Luna, Gemma 4 31B, or Llama-4 Maverick must complete the
specified ordering without flagged ERROR, MOCK, or empty tool output and return
a gold-equivalent answer. An explicit submission takes precedence; otherwise,
the complete final synthesis response is used. Answer matching includes
task-aware normalization and LLM-assisted semantic review. All Native chains
are retained regardless of their replay outcomes.

There are 64 tasks with retained alternatives and 56 with at least two. The
original 63-task rich cohort is retained for the corresponding fixed-cohort
comparison. Successful replay is executor-dependent evidence, not certification
of every step's semantics or every dependency. Gemma 4 and Llama-4 Maverick also
appear in the evaluated model cohort, and executors share tool helpers.

These files record the reference-retention decisions. Offline validation
checks identities, permutations, hashes, and accounting; it does not rerun
models or independently adjudicate semantic answer matches. Source annotations
and the official gated GAIA inputs are required to rebuild runnable tasks.
