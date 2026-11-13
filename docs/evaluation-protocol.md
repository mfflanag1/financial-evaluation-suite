# Evaluation protocol

## Freeze the experiment

Before a run, record the repository commit, scorer version and binding; case IDs and related-case groups; split and information condition; exact provider messages and tool schemas; model version; sampling settings; retry/attempt limits; and runtime image identity. Hash candidate-visible packets and record sequential revisions.

Label the published static bank `public-reference`. Historical holdout assignments do not establish current secrecy. For generated tasks, record generator version, split configuration and a non-secret key identity. Do not reveal the key or choose seeds after inspecting performance.

## Control information and actions

Keep candidate inputs separate from answer keys, evaluator-only annotations, future states and hidden continuations. A solver given the entire public clone has a different information condition from one given a task packet; report which condition applies.

Use the coordinator for sequential cases. Only submitted actions advance state. A counterfactual replay under corrected scoring is not a fresh solver trial. Retrieval budgets, released documents and actual actions belong in the private run record.

Collect the saved program and reproduce it in a fresh public-data container before financial grading. State whether replay used original inputs only or alternate realizations too. Same-input reproduction does not by itself rule out hardcoded outputs.

## Treat failures explicitly

Separate candidate failures from invalid observations. Intentional empty completions, malformed calls and invalid schemas can be scored candidate failures. Pending tool execution, provider outages, dead containers and unresolved cancellation may be infrastructure failures or unknown. Preserve the raw runtime outcome alongside diagnostic classification.

Report planned attempts, completed attempts, candidate failures and unresolved/infrastructure outcomes. An all-attempt sensitivity analysis may count missing attempts as zero, but must be labelled and accompany the observed-outcome report. Successful-only averages must not stand alone.

Retries follow the frozen rule. Rerunning after seeing a bad answer is a new attempt. Do not choose the best response retrospectively unless best-of-k was specified.

## Report interpretable metrics

Report per-track denominators and distinguish:

- Strict case correctness and per-output financial accuracy.
- Evidence support and citation contract validity.
- Schema validity and action feasibility.
- Checkpoint completion and independent program reproduction.
- Economic reward, regret and constraint violations in their stated units.
- Attempt completion, infrastructure failures and unknown outcomes.

Do not add dollar regret, basis-point loss and abstract utility. Expected regret is not realized trading loss. Separate public-calculation rewards from withheld-path economics and report reference ceilings.

Resample independent cases or financial mechanisms for uncertainty, not correlated output fields. Group variants and repeated attempts with the parent case. Use paired comparisons when methods share cases. A small pilot supports debugging and preliminary difficulty estimates, not a stable leaderboard.

## Compare versions

Static responses can be graded by frozen and corrected scorers. Label regradings and retain both reports. Sequential corrections require fresh sessions because execution can change subsequent state. Do not combine scorer versions in one headline metric.

Reference replay, independent calculation, scripted-provider tests, model trials and practitioner review are distinct evidence. Name only the checks actually completed.
