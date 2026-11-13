# Portfolio runtime

A separately versioned successor to the clarified portfolio environment, using
Verifiers 0.4.1 native `Taskset`, `Task` and `Harness` interfaces. Financial data,
checkpoint rules, reproduction rules and financial scoring are inherited intact.
This package does not implement the legacy `load_environment` interface.

The harness binds the actual initial outbound messages after native network
notices and Bash/Edit instructions are added. A host wrapper checks that binding
before contacting a model. Optional `guidance` is empty by default and remains
idempotent when the coordinator resolves, copies and launches a prompt. Resumed
exchanges are unsupported in runtime v1.

Host journals record provider requests, tool gates/results and terminal runtime
observations. They contain hashes and bounded metadata, never credentials or
exception bodies. An unanswered tool call is an unresolved execution outcome;
it does not earn an invented zero financial score. Intentional empty completions,
malformed tool arguments and completed incorrect submissions remain candidate
failures. A cancelled request does not establish provider fault.

## Install

Install all predecessor packages and this package in the same Python 3.12
environment, using the repository's pinned dependencies. For the new package:

```sh
uv pip install --python "$EVAL_PYTHON" --no-deps environments/financial_portfolio_runtime
```

## Fresh bootstrap

An installed Docker image and the frozen runtime manifest are required. No
historical run, saved private answer or existing evaluator credential is needed.
The command generates a new private evaluator key and runs four public-only
reference reproductions serially. Images are never pulled implicitly.

```sh
financial-portfolio-runtime bootstrap --repo "$REPO" \
  --protocol "$REPO/configs/financial_portfolio/runtime_v1/manifest.json" \
  --out "$PRIVATE_RUN_ROOT/bootstrap" --image python:3.12-slim
```

The fresh output directory must be outside the repository. Keep the generated
`evaluator.key`, reference answers and reports private. Bootstrap establishes
software/reference reproducibility; it does not establish model performance,
an independent financial audit, or unseen-distribution generalization.

## Evaluate

Copy the [smoke configuration](../../configs/financial_portfolio/runtime_v1/smoke.toml)
outside the repository. Bind an absolute output directory, model, endpoint and
installed Docker image. Supply the model credential through
`PORTFOLIO_MODEL_API_KEY` on the host. The evaluator key is loaded from bootstrap
and neither credential is forwarded to solver containers. Keep `dry_run = true`
until the local bindings have been checked, then use fresh output and journal
directories for a live invocation. Provider use may incur charges.

```sh
financial-portfolio-runtime run --repo "$REPO" --config "$BOUND_CONFIG" \
  --protocol "$REPO/configs/financial_portfolio/runtime_v1/manifest.json" \
  --bootstrap "$PRIVATE_RUN_ROOT/bootstrap/references.json" \
  --journal-dir "$PRIVATE_RUN_ROOT/journal"
financial-portfolio-runtime report --run "$NATIVE_RUN_DIR" --expected 4 \
  --journal-dir "$PRIVATE_RUN_ROOT/journal" \
  --out "$PRIVATE_RUN_ROOT/report.json"
```

Runtime v1 enforces one attempt per task, serial container execution, zero
transport/episode retries and isolated scoring. Failed attempts stay in the
denominator. An aggregate reward is available only when every scheduled attempt
has a resolved financial outcome and its task, input, prompt, model, image and
host-journal evidence match the prelaunch binding. Count-only reports remain
diagnostic and have no aggregate score. Provider usage coverage must be reported
separately from financial outcome coverage.

## Verification and limits

```sh
"$EVAL_PYTHON" -B -m unittest discover -s tests -p 'test_runtime_release_*.py' -v
RUN_RUNTIME_DOCKER_TESTS=1 "$EVAL_PYTHON" -B -m unittest discover \
  -s tests -p 'test_runtime_release_native.py' -v
```

Docker integration fixtures use a local scripted provider and public reference
source; they never call a paid model. A fixture success demonstrates the native
execution/grading path, not solver capability. Model calibration and an actual
RL optimizer step are separate validation requirements. This runtime makes no
claim that training has succeeded.
