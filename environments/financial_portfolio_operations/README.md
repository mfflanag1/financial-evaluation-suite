# Portfolio operational controls

An opt-in trusted-host layer around the frozen clarified portfolio taskset.
It adds persisted model-request events and numerical reference controls. It does
not change the solver prompt, public data, candidate grid, tools or financial
scorer. The package targets Verifiers 0.4.1 and is a separately labelled condition.

## Install

Install the existing local portfolio packages first, then this package in the
same Python 3.12 environment:

```sh
uv pip install --python "$EVAL_PYTHON" --no-deps -e environments/financial_portfolio_operations
```

The [operational protocol](../../configs/financial_portfolio/operational_v2/README.md)
specifies the fixed schedule, failure rules and binding requirements. Package
installation alone does not launch evaluation, training or publication.

## Request telemetry

The operations harness inherits the staged harness unchanged and requires an
active trusted-host capture context. The CLI opens that context before invoking
the native runner. A scoped hook around the pinned native interception client's
resolver is restored when the context exits; installed dependencies are not
modified. The hook fails closed for unsupported clients or missing registration.

Each request gets a persisted start before awaiting the provider. A terminal
record describes success, error or cancellation, elapsed time and available
provider usage. Request metadata binds to the native trace and task. Journal
files contain no prompt, response, HTTP headers, credential values or exception
bodies. They live only on the trusted host and are not forwarded to solver
containers. A new private journal directory is required for each invocation.

```sh
financial-portfolio-operations run --config "$BOUND_CONFIG" \
  --protocol "$REPO/configs/financial_portfolio/operational_v2/manifest.json" \
  --journal-dir "$JOURNAL_DIR" --reference-report "$REFERENCE_DIR/references.json"
```

The config controls `dry_run`; the checked-in template sets it to true. Every
invocation requires a fresh result directory, including after a dry run. A live
invocation requires completed numerical controls, matching task identities and
inputs, the original evaluator-key fingerprint and Docker image. Imported source
paths and dependency versions must match the bound checkout. Credentials are
supplied through host environment variables, never the config or command line.

An interrupted process can leave a start without a terminal event. That request
remains unresolved; usage and cost are unknown. Cancellation does not establish
provider fault. Report known usage and its coverage separately from total spend.
Never add reasoning tokens twice when they are included in completion tokens.

```sh
financial-portfolio-operations requests --journal "$JOURNAL_DIR" --out "$REQUEST_REPORT"
```

## Numerical reference controls

```sh
financial-portfolio-operations references \
  --out "$REFERENCE_DIR" --repo "$REPO" \
  --protocol "$REPO/configs/financial_portfolio/operational_v2/manifest.json" \
  --baseline-preflight "$BASELINE_PREFLIGHT" --key-file "$EVALUATOR_KEY_FILE"
```

This computes every development reference, seeds 128–159, using the baseline
evaluator key and pinned Docker image. Only public inputs and reference source
enter the isolated reproduction container. The trusted host grades the returned
policy and compares it with the private reference. It persists exact rewards,
components, public/source hashes and the key fingerprint; no evaluator key is
written. Output must be a fresh directory outside the repository.

These are model-free reference controls with public reproduction and financial
grading, not a native model evaluation. A publicly correct reference can score
below one because its required selection fixes the withheld economic component.
The scores describe attainable ceilings; they do not normalize or alter official
rewards. Keep this report and its artifacts outside solver context.

## Verification

```sh
"$EVAL_PYTHON" -B -m unittest discover -s tests -p 'test_portfolio_operations_*.py' -v
"$EVAL_PYTHON" -B configs/financial_portfolio/operational_v2/freeze_manifest.py --check
```

Synthetic provider tests check request completion, errors, cancellation and
incomplete journals without spending model credits. They are not model results.
