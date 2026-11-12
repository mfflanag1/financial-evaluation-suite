# Prerequisite comparison protocol v1

This diagnostic compares two instruct models on four fixed synthetic tasks before
either advances to portfolio research. It is separate from the 224-question bank
and does not change any existing question, scoring version, or portfolio condition.
It tests whether the model can inspect files, save and execute Python, repair
supplied broken source, and produce complete structured results.

| Order | Task ID | Required behavior |
| --- | --- | --- |
| 1 | `prerequisite-schema-v1` | Inspect CSV and JSON, distinguish source and output schemas. |
| 2 | `prerequisite-calculation-v1` | Save and execute calculations from supplied raw inputs. |
| 3 | `prerequisite-repair-v1` | Repair the supplied broken starter and save working source. |
| 4 | `prerequisite-coverage-v1` | Include every requested ID with correct integer and null types. |

Each task requires `research.py` and `policy.json`. Correct saved output is
necessary but insufficient: the source must reproduce correct output on both the
original public inputs and alternate public inputs of the same schema in fresh
isolated containers. Reference answers stay in the trusted evaluator. The solver
uses only the Python standard library and receives no model credentials or
evaluator secrets.

The repair prompt also asks the model to run the broken starter before fixing it.
Inspect the trusted trace separately for that failed execution and subsequent
recovery. Reward checks the corrected output and source reproduction; it does not
enforce the starter execution or prove that an execution error occurred.

## Fixed comparison

Use `financial-prerequisites` 0.1.0 with the existing
`financial-portfolio-staged` 0.4.0 harness and Verifiers 0.4.1. Bind two available
instruct models before running either. Retain the same task order, prompt, public
inputs, scorer, harness, Docker image, temperature, budgets and retry settings for
both models. Each model receives one attempt per task: four attempts per model,
eight attempts total. Do not select the better of repeated runs.

Both models receive 16 turns, 32,768 output tokens, 131,072 total tokens,
8,192 tokens per response, temperature 0.2 and concurrency two. Episode, agent,
scoring and SDK retries are zero. The Docker runtime has no network allowlist
entries and forwards no host environment variables. Scoring is isolated. Malformed
tool arguments are candidate protocol failures; provider failures remain
infrastructure failures. Never repair a model response and then count the repaired
attempt as the original.

The checked-in configuration is dry-run-only with an invalid placeholder endpoint.
Create a separate evaluator-local bound copy and fresh output directory for each
model. Before live execution, record model and endpoint identifiers, available
deployment and capability metadata, dependency and container image versions, the
source revision, manifest digest, exact bound configuration, public input hashes,
prompt hashes, and four expected attempts. Store credentials only in the host
process environment. Live traces and run summaries stay outside the repository.

## Advancement gate

A model advances only after **4/4 task passes**. Every pass requires correct saved
output and successful original-input and alternate-input source reproduction.
Missing attempts, unknown outcomes, unresolved infrastructure failures or a failed
task prevent advancement. Report failure reasons, malformed calls, token usage,
missing usage and costs separately; a successful API request is not a task pass.
These four tasks diagnose workflow readiness and do not estimate financial skill.

Only a passing model runs the unchanged four portfolio episodes in
[`staged_v1`](../../financial_portfolio/staged_v1/README.md), training seeds 4–7,
one attempt each. Use the original 32-turn portfolio budgets, the same public
episodes and evaluator-key fingerprint, and the same frozen staged instructions
and scorer for every advancing model. The prerequisite tasks do not consume or
replace any portfolio episode. Record portfolio checkpoint completion, source
reproduction and financial correctness separately.

The 64-attempt calibration remains a later gate: first account for all diagnostic
attempts, resolve infrastructure failures and obtain at least one reproducible
portfolio submission. Passing prerequisites alone does not authorize that
expansion. Preserve unsuccessful runs and do not claim model superiority or
population accuracy from a four-case sample.

## Preparation and integrity

Install the new package alongside the existing pinned packages using Python 3.12
or later, then validate with the native runner:

```sh
uv pip install --python .venv-portfolio/bin/python ./environments/financial_research ./environments/financial_portfolio ./environments/financial_portfolio_staged ./environments/financial_prerequisites
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_prerequisite*.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_prerequisites/prerequisite_v1/comparison.toml --dry-run
```

`manifest.json` binds all new package and protocol files, these protocol tests,
the ordered task IDs, and the frozen staged predecessor manifest. It does not pin
the remote model deployment. The freeze helper verifies predecessor hashes before
writing; run it only after implementation, review and local validation are complete,
and before the first live call:

```sh
python3 configs/financial_prerequisites/prerequisite_v1/freeze_manifest.py --write
python3 configs/financial_prerequisites/prerequisite_v1/freeze_manifest.py --check
```

`--write` refuses to replace an existing manifest. If a pre-live draft needs
correction, remove that draft explicitly, revalidate the changes and freeze again.
After live execution, preserve this condition; any correction requires a new
versioned condition rather than overwriting its manifest or results.
