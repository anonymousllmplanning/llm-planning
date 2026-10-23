# Reference selection and replay evidence

## Planning references

Keep each prediction fixed. Build the Native reference and one reference for
every supplied candidate ordering, retaining the same node inventory. Score
each reference with a shared node alignment and select the highest
PlanningScore. Exact ties prefer Native, then candidate rank.

The ordering schedule is part of the experiment input. Record its checksum and
the number of candidate-bearing tasks. A sensitivity analysis that changes the
reference budget must preserve the prediction and evaluation cohort.

## Fixed-order replay

Replay provides evidence about behavior under a prescribed ordering. Execute
the scheduled steps in order, retain concrete intermediate results, and compare
the Native and candidate final answers. Pure reasoning steps do not need an
artificial tool call. Tools must retain their actual arguments and observations.

Report these outcomes separately:

- Both executions produce an answer matching the reference.
- Both are incorrect and produce equivalent answers.
- Both are incorrect and produce different answers.
- One execution matches and the other does not.
- Execution or scoring is insufficient to decide, including tool faults,
  incomplete traversals, mock outputs, and missing answers.

Equivalent incorrect answers are a behavior-preservation observation. They are
not evidence of task correctness. Unobserved candidates remain unknown.
Answer matching used by replay must be identified separately from the completed
human review of the main model answers.

## Model committees

An intersection requires every selected validator to support a candidate;
a union requires at least one; a voting rule requires its stated quorum.
Use the same observed candidate pool when comparing these policies. Record
unknown cases separately and keep Native as a scoring fallback.

Choose the committee before inspecting which policy improves a model's planning
score. When a validator also appears in the evaluated cohort, report that
overlap and include a sensitivity analysis. Incomplete third-model coverage
cannot support a full-pool three-model comparison.

Human AD/SD review checks whether the generated dependencies make semantic
sense. It is separate from replay outcomes and final-answer grading. Report
completed annotations and their sampling design; do not transfer a quality
percentage from an earlier annotation set.
