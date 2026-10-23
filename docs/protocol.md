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
Historical first-pass budgets differ; the framework preserves checkpoints and
lineage rather than claiming every cohort prediction is a fresh identical-budget
run. The portable repair wrapper accepts operator-supplied checkpoints; raw
historical checkpoints and their model-specific selection manifests are separate
experimental artifacts.

## Evidence

Inference records include raw generations, parsed plans, S1 and S3 snapshots,
API generation metadata, S4 decision streams, tool arguments and outputs, S5
parse status, and answer handoff fields. Repair adds an immutable baseline copy,
normalization audit, selected cases, per-case command, raw attempt logs, repaired
predictions, and technical promotion decisions. Credentials are not intentionally
written into these records. Keep local run outputs untracked.

## Reference evaluation

The reference version contains 165 Native chains plus 1,786 candidate orderings.
Each candidate shares the Native node inventory and connects adjacent actions
according to its recorded permutation. The evaluator reuses the same node
alignment, computes each reference's PlanningScore, then selects the maximum.
Native wins exact ties, followed by lower candidate rank. S1 and S3 use the same
selection rule and are reported separately.

Final-answer correctness uses S5. Automatic exact matching is a diagnostic;
human judgments are a separate input. Tool metrics retain their eligibility
denominators and normalized-value diagnostics. TaskBench and UltraTool are
auxiliary benchmarks and do not share GAIA's final-answer target.

Replay evidence is separate from the planning maximum. Missing answers,
transport failures, partial traversals, early submissions, mock outputs, and
tool errors require review; they do not establish semantic invalidity of a DAG.
