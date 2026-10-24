# Final reference selection and replay evidence

## Published reference set

The final set has 165 Native chains plus 1,429 retained alternatives: 1,594
references shared by all evaluated configurations. The sampled pool contained
1,786 alternatives on 72 tasks. After retention, 64 tasks have an alternative
and 56 have at least two. The original rich cohort of 63 tasks, including its
51-task Levels 2+3 subset, remains fixed for the corresponding score comparison.
The budget experiment instead uses the 56 retained-rich tasks.

Use `references/ordering_schedule.json` with the matching source annotations.
Each reference preserves the Native nodes and replaces adjacent chain edges
with the recorded permutation. Predictions and node alignment remain fixed
across references. Select maximum PlanningScore; exact ties prefer Native,
then lower candidate rank. The compiled dependency DAG constrains generation
and is not the adjacency-scoring target.

## Successful-replay retention

Retain a candidate if at least one of GPT-6 Luna, Gemma 4 31B, or Llama-4
Maverick completes its specified step order without flagged ERROR, MOCK, or
empty tool output and returns a gold-equivalent final answer. An explicit
submitted answer takes precedence; otherwise use the complete final synthesis
response. Matching includes task-aware number, unit, list, and source-alias
normalization with LLM-assisted semantic review. Keep every Native chain
regardless of its own replay outcome.

The supplied final candidate decisions and supporting executors are preserved
in `references/replay_retained_orderings.json`. Their full permutations and
source hashes appear in `references/reference_set_1594.csv`. The JSON evaluator
schedule is a lossless conversion of that CSV's task IDs, kinds, ranks, and
orders. Offline checks do not recreate live replay or semantic answer review.

Earlier Native/candidate outcome-preservation diagnostics answer a different
question: both-correct and same-wrong pairs can preserve outcome. A same-wrong
pair does not establish successful gold-equivalent replay under the final rule.
Do not substitute earlier outcome-majority counts for the final retained list.

## Scope of evidence

The final rule is an at-least-one-success union, not an executor intersection
or majority vote. The evaluated cohort includes Gemma 4 and Llama-4 Maverick;
executors also share helpers. Retention is therefore configuration-dependent
evidence, and the original rich cohort must not be redefined after filtering.

Source plans may include answer literals. Successful replay does not certify
step semantics, safe parallel execution, or every AD/SD relation. Human review
assesses dependency semantics on its stated panel and remains complementary
to replay, source-plan coherence checks, and final-answer adjudication.
