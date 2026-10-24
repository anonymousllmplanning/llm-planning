# Experiment protocol

The nine-model cohort uses a staged planner and executor. S1 generates abstract
planning intent, S2 assesses the tool catalog, S3 constructs a concrete plan,
S4 executes tools, and S5 synthesizes the answer. The default S1-to-S3 handoff
uses steps only. S4 receives the recorded S3 nodes, edges, arguments, output
variables, and planned calls as structured context, without treating planned
content as observed evidence.

## Model and helper configuration

The inference backend reads `LLM_API_BASE`, `LLM_API_KEY`, and
`LLM_API_PROVIDER`. Use the exact model identifier accepted by the endpoint.
Configure helper deployments separately through `VISION_API_BASE`,
`VISION_API_KEY`, `VISION_MODEL_NAME`, `VISION_FALLBACK_MODEL`,
`REASONING_API_BASE`, `REASONING_API_KEY`, and `REASONING_MODEL_NAME`.

The cohort's shared vision helper was `gemma-4-31B-it`, with
`VISION_USE_PLANNER_MODEL=false`; the reasoning helper was
`Llama-3.3-70B-Instruct`. The technical repair workers used local audio fallback
through `ASR_USE_LOCAL_ONLY=true`. Deployment addresses and credentials are
supplied by the operator. Record the exact helper configuration with each run;
changing helper models changes the experiment.

## Parsing and repair

The parser attempts valid JSON before stripping model wrappers. Normalization
copies its input and handles root graph envelopes. Saved raw S5 responses are
interpreted with a common parser; the native S4 submission and S5 answer remain
separate fields.

Technical selection covers malformed or missing graphs, fallback planning,
truncated S1/S3 generations, missing execution evidence, invalid S4 actions,
non-uniform recovery policies, and empty or ambiguous S5 answers. The earliest
affected stage determines where to resume. A case receives at most one new
repair attempt. The promotion gate checks graph validity, parseability, task
identity, and exact preservation of upstream stages.

The common repair caps are 8,192 output tokens for S1, 16,384 for S3, and 4,096
for S5. S4 has 15 turns and up to two action-generation requests per turn.
Selection and promotion never depend on reference answers or evaluation scores.
These caps apply to repair attempts. Record the actual first-pass budgets
and generation metadata of supplied checkpoints. Preserve stage lineage and
the associated model-specific selection manifests with each experiment.

## Evidence

Inference records include raw generations, parsed plans, S1 and S3 snapshots,
API generation metadata, S4 decision streams, tool arguments and outputs, S5
parse status, and answer handoff fields. Repair adds an immutable baseline copy,
normalization audit, selected cases, per-case command, raw attempt logs, repaired
predictions, and technical promotion decisions. Credentials are not intentionally
written into these records. Keep local run outputs untracked.

## Reference evaluation

The reference set contains 165 Native chains plus 1,429 retained
alternatives (1,594 references), selected from 1,786 sampled candidates.
Each candidate shares the Native node inventory and connects adjacent actions
according to its recorded permutation. The evaluator reuses the same node
alignment, computes each reference's PlanningScore, then selects the maximum.
Native wins exact ties, followed by lower candidate rank. S1 and S3 use the same
selection rule and are reported separately.

S3 is the primary pre-execution planning stage; S1 is an auxiliary comparison.
Open-weight final-answer correctness uses S5. Closed-model results use the
recorded submitted response, or the recorded final response when submission is
absent. Human judgments refer to that same group-specific returned response;
they are not transferred between response fields. Automatic exact matching is
a diagnostic; human judgments are a separate input. Tool metrics retain their eligibility
denominators and normalized-value diagnostics. TaskBench and UltraTool are
auxiliary benchmarks and do not share GAIA's final-answer target.

The planning maximum uses the published retained pool. Retention requires at
least one successful replay by GPT-6 Luna, Gemma 4 31B, or Llama-4 Maverick:
complete ordered execution, no flagged ERROR/MOCK/empty tool output, and a
gold-equivalent final answer. All Native chains are included. The final matcher
and submitted-answer fallback are recorded in `references/README.md` and
`references/replay_retained_orderings.json`.

Missing answers, transport failures, partial traversals, and tool errors do not
by themselves establish semantic invalidity of a DAG. Successful replay also
does not certify every step or AD/SD relation. Human dependency review remains
a separate semantic assessment.
