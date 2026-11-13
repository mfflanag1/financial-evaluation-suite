# Quickstart

Run commands from the repository root. Static-bank commands require Python 3.11 or later. Native environment integration requires Python 3.12 and Verifiers 0.4.1. Reference checks need no model credentials.

## Static bank

```sh
make check
python3 -B -m scoring_v3.grade validate
python3 -B -m scoring_v3.grade export q076 > question.json
```

Give the candidate only `question.json` and authorized inputs. Follow the exported response schema; tracks do not all use the same format. Save the response and grade it on the trusted evaluator host:

```sh
python3 -B -m scoring_v3.grade grade q076 response.json
```

For the seven source cases with reviewed evidence, build a packet containing matching v3 contracts and permitted source files:

```sh
python3 -B -m distribution.corrected create --out ../source-contracts-v3
python3 -B -m distribution.corrected verify ../source-contracts-v3
```

Give the solver only the selected case directory. The packet includes no answers or grader. Consult the [source policy](source-policy.md) for included cases, omissions and distribution limits.

## Sequential cases

```sh
python3 -B -m scoring_v3.sessions start c207 --condition supplied --session sessions/evaluator --out sessions/candidate
python3 -B -m scoring_v3.sessions submit --session sessions/evaluator --out sessions/candidate --revision 0 --response response.json
python3 -B -m scoring_v3.sessions report --session sessions/evaluator
```

Read each next revision from the current public packet. Zero applies only to the initial submission. These paths illustrate the protocol; separate directories alone do not enforce access control. The solver must be unable to read evaluator state, future turns or answer files.

`--allow-holdout` is a historical protocol gate. It does not turn publicly committed answers into an unseen test set.

## Workflow programs

The six executable workflows use a separate controller. Build its image and follow the two-attempt protocol in the [scorer documentation](../scoring_v3/README.md). Static executed-answer grading does not run a submitted program.

## Verification tiers

The base command runs manifests, reference controls and discovered unit tests. Read its skipped-test summary. The release runner also checks case-level audit records and source inventories:

```sh
python3 -B scripts/check.py --tier core
python3.12 scripts/bootstrap.py --venv ../financial-evals-venv
../financial-evals-venv/bin/python -B scripts/check.py --tier native
```

The bootstrap requires a new destination outside the checkout and network access for dependencies. It defaults to a sibling directory. This keeps installed packages separate from source imports during validation. It installs every local package together using the recorded third-party runtime constraints. Offline installation is available with a complete wheelhouse; see `python3.12 scripts/bootstrap.py --help`.

The native tier fails if the pinned dependency is absent or a non-Docker test is skipped. The Docker tier enables all container tests and fails on any skip:

```sh
docker pull python:3.12-slim
docker build -f scoring_v2/Dockerfile.workflow -t financial-evals-workflow:v2 scoring_v2
../financial-evals-venv/bin/python -B scripts/check.py --tier docker
```

Use `--report ../verification.json` to write a new machine-readable report outside the checkout. `--pattern 'test_runtime_release*.py'` selects a narrower test family; a filtered run is not a full-suite pass. GitHub Actions runs core and native tiers automatically, with Docker available through manual dispatch.

Scripted local providers exercise native protocols without paid model calls. They test orchestration, not model performance. Run container tests serially so resource contention does not obscure outcomes.

## Selection-adjusted research

The bootstrap includes the separate [selection environment](../environments/financial_selection/README.md). It generates original synthetic observations, grades exact statistics and funded allocations, and reruns submitted source on the current packet plus two fresh packets. Follow its package instructions for a model-free native validation with an immutable local Docker image and a worker-only generation key.

To reproduce the documented outcome-blind cash baseline after installation:

```sh
../financial-evals-venv/bin/python -B -m financial_selection.baseline
```

This command uses an explicitly public fixture key, prints aggregate calculation scores, and makes no model requests. It does not use a private evaluation key or provide contamination-free test results. Its shaped reward and exact-success rate measure different things; report both.

The [selection calibration protocol](../configs/financial_selection/calibration_v1/README.md) prepares five instruct-model attempts with exact task/configuration bindings. Its supplied configurations perform dry runs only; missing actual attempts leave performance means unset. Use the separate [selection training installation](../training/selection/README.md) for trainer compatibility checks, and the [practitioner review procedure](selection-practitioner-review.md) for financial assessment.

## Results and secrets

Store responses, traces and evaluator keys outside the repository. Ignore rules are not a security boundary. Never mount the full clone, installed evaluator package or evaluator key inside the solver container. Preserve environment versions, exact prompts and private run artifacts for reproducibility.
