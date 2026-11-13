# Selection-adjusted research and funded allocation

Experimental evaluation environment for one financial mechanism: reconstruct a dated
research population, correct its selectively observed outcome archive, transfer
estimates to a disclosed population mix, and choose integer funded allocations.
Original synthetic data only. This package is an experimental release;
it does not change existing question banks, scores or environment protocols.

The public reference reads five files and produces a declarative policy. The
evaluator independently reconstructs and checks the same public calculation;
private complete-population values are separate economic diagnostics. An archive
estimate is not exact recovery of missing returns. The sampling standard error
is not market risk or a post-selection confidence bound.

See [CONTRACT.md](CONTRACT.md) for the precise input, identification, accounting
and output rules. Authored mathematical controls exercise the mechanism; generated
episodes and evaluator keys belong outside the repository.

The native Verifiers 0.4.1 Taskset collects `research.py` and `policy.json` from
the solver. It reruns that exact source in three fresh public-only containers:
one must reproduce the submitted policy, and two must solve distinct generated
packets. No task package, generation key or hidden population is placed in those
containers. The standalone calculation verifier does not execute candidate code.

## Reward and reporting

Each packet awards 0.20 audit, 0.35 statistical calculations, 0.15 cash ledger
and 0.30 canonical optimal allocation. Current-policy reproduction is mandatory;
afterward the native reward averages the current and two fresh packet scores.
Fresh candidate failures earn zero for their packet. Infrastructure failures
remain unresolved and are excluded from performance claims, rather than silently
treated as financial errors. `evaluation_admission(trace)` validates the native
outcome before a caller includes it in a performance denominator.

Report exact full-policy success alongside shaped reward, its components and
condition-specific baselines. A public-fixture control that never reads markouts
and always holds cash earned **0.67454 mean shaped reward and 0/100 fully correct
policies**. Partial credit alone is insufficient evidence of financial reasoning
or successful RL. Native metrics include current exact success, fresh exact
success fraction, all-packets-correct and reproduction status. They expose no
fresh answer values. Aggregate reporting must also account for every planned
attempt; the admission helper is not a complete experiment ledger.

## Local verification

Python 3.12 is required for the native adapter. From the repository root, install
the local replay dependency and this package together in a new environment:

```sh
python3.12 -m venv ../financial-selection-venv
../financial-selection-venv/bin/python -m pip install \
  --constraint requirements/runtime-constraints.txt \
  ./environments/financial_research './environments/financial_selection[runtime]'
../financial-selection-venv/bin/python -m pip check
source ../financial-selection-venv/bin/activate
```

This installs `financial-research` 0.2.0 and Verifiers 0.4.1. The core financial
modules use only the standard library. Reproduce the documented 100-fixture cash
baseline without Docker or a model endpoint:

```sh
../financial-selection-venv/bin/python -B -m financial_selection.baseline
```

The command prints aggregate scores to stdout using an explicitly public fixture
key. Expected mean shaped reward is `0.6745416666666667`, with zero fully correct
policies. It writes no generated policies or datasets.

For a model-free native lifecycle check, prepare a local `python:3.12-slim` Docker
image. If absent, obtain it with `docker pull python:3.12-slim`. Resolve its
immutable identity and create a fresh worker-only key for this validation session:

```sh
export FINANCIAL_SELECTION_DOCKER_IMAGE="$(docker image inspect python:3.12-slim --format '{{.Id}}')"
export FINANCIAL_SELECTION_KEY="$(../financial-selection-venv/bin/python -c 'import secrets; print(secrets.token_hex(32))')"
../financial-selection-venv/bin/vf-validate \
  @ environments/financial_selection/configs/native_validate.toml \
  --taskset.task.image "$FINANCIAL_SELECTION_DOCKER_IMAGE" \
  --runtime.image "$FINANCIAL_SELECTION_DOCKER_IMAGE" \
  --output-dir ../financial-selection-validation
unset FINANCIAL_SELECTION_KEY
```

The pinned `vf-validate` command runs a reference submission and an untouched
submission, serially, with no agent or model call. The reference must reproduce
its policy and solve two fresh packets; the untouched submission must score zero.
Require `outcomes.valid: 1`, all other outcome counts zero, and `owed: 0` in the
emitted summary, with both `gold` and `noop` checks valid. The untouched check's
reward in `results.jsonl` must be zero. A CLI exit status alone is insufficient.
Outputs go outside the checkout. Preserve the same key securely outside the
repository if an identical dataset is needed later; the session example discards it.

From this package directory, run the CPU and wire controls in the installed
environment:

```sh
cd environments/financial_selection
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

The default run explicitly skips five real-container tests. With Docker available,
prepare the local `python:3.12-slim` image used by the replay controls, set its
immutable `sha256:...` identity for the native gate, and run both gates serially:

```sh
export FINANCIAL_SELECTION_DOCKER_IMAGE='sha256:REPLACE_WITH_LOCAL_IMAGE_ID'
FINANCIAL_SELECTION_DOCKER_TESTS=1 PYTHONPATH=src \
  python -B -m unittest discover -s tests -p test_replay.py -v
FINANCIAL_SELECTION_NATIVE_DOCKER_TESTS=1 PYTHONPATH=src \
  python -B -m unittest discover -s tests -p test_native.py -v
```

These controls use explicitly public fixture keys, no model endpoint and no
inference spend. [VALIDATION.md](VALIDATION.md) separates completed checks from
unverified claims. The source distribution includes the native validation config;
wheel users can obtain it from the repository or pass the equivalent CLI flags.

## Native boundary

The exported plugin ID is `financial-selection`. Configure
`SelectionTasksetConfig` with an immutable `task.image`, a split (`train`, `dev`
or `final`), one of five conditions, and at most 64 current packet indices.
Set `FINANCIAL_SELECTION_KEY` on trusted workers only: 32–512 random bytes
encoded as hexadecimal. Never use a documented fixture key for a private run.
Worker identity recovery needs no process-local registration. All workers must
receive the same key; rotating it changes the dataset. Related generated draws
are not new independent financial mechanisms.

Task payloads contain only the prompt, opaque identity and five public files.
The bounded registry regenerates two distinct fresh packets per current index.
Solver, verifier and replay must share the exact immutable image; host mounts,
forwarded environment variables, egress, scoring runtime overrides and scoring
retries are rejected. Only regular bounded output files are captured. Submitted
Python executes under the existing privilege-dropping public replay controller,
never on the trusted host or in a private-data grader.

The caller must preserve attempt accounting and runtime isolation when integrating
an optimizer. No live model calibration, optimizer update, checkpoint reload,
training benefit or practitioner validation has been demonstrated. This package
remains unlicensed; publication grants no new reuse permission.
