# Financial research pilot

A separate generated environment for point-in-time data repair, signal testing,
cost-aware validation and a constrained portfolio decision. It does not alter the
frozen question banks or their scores and is not counted as additional independent
questions in the main catalog.

Environment **0.2.0** targets **Verifiers 0.4.1** using its native v1 `Taskset` API.
The core generator and policy grader use only Python's standard library. Install
the `prime` extra for the container-backed integration.
Completed local checks are recorded in [VALIDATION.md](VALIDATION.md).

## Task

Each episode supplies eight synthetic assets over 120 trading sessions:

| Input | Work required |
| --- | --- |
| `prices.csv` | Reconstruct returns across stock splits; resolve shuffled observations. |
| `securities.csv` | Apply dated identity mappings, including renamed and reused tickers. |
| `features.csv` | Handle duplicates, missing releases, delayed publication and revisions using availability time. |
| `contract.json` | Apply the portfolio, cost, validation and submission rules exactly. |

The four generated families contain a usable signal, no signal, a signal overwhelmed
by costs, and a revised-data trap. Days 0–71 are available for exploration; days
72–119 form the prescribed validation window. The solver must evaluate all 16
declared feature/direction/gross candidates, report data-audit values and trial
metrics, and select the highest positive validation screening score or abstain.

Portfolios hold two equal-weight longs and two equal-weight shorts, with gross
exposure at most one and per-asset exposure at most 0.25. Costs include entry,
daily rebalancing, final liquidation and daily short borrow. Sharpe uses arithmetic
daily net returns, sample standard deviation and a 252-session annualization.
The screening score is mean net return minus two standard errors; it is a specified
selection heuristic, not a significance claim or a multiple-testing correction.

The solver writes `policy.json` and `research.py`. Native grading requires a
nonempty regular UTF-8 source file of at most 1 MiB and reruns it in a separate,
fresh public-data container. The rerun receives only the original four input files
and the submitted source, with no submitted policy, evaluator package, secret key
or hidden paths. It must recreate `policy.json` within 30 seconds using Python's
standard library. Numeric outputs use the scoring tolerances; row ordering does
not matter, and the selected decision must match exactly.

The independent standalone reference in `src/financial_research/reference.py`
demonstrates a solution using only public inputs. It is packaged for model-free
validation and is not staged into normal solver tasks.

## Reward

| Component | Weight | Check |
| --- | ---: | --- |
| Data audit | 0.25 | Thirty point-in-time feature and return values. |
| Trial calculations | 0.35 | All four metrics for each of 16 candidates. |
| Validation decision | 0.20 | Correct choice or abstention under the declared rule. |
| Withheld-path quality | 0.20 | Selected policy versus the bounded candidate set on six independent 64-session paths. |

Economic credit requires all preceding checks to pass. The withheld score is the
mean of each path's own screening score. Deployment earns its positive score
divided by the best positive score, clipped to [0,1]. Abstention earns this component
only when every candidate's mean withheld score is nonpositive. A correct validation
decision may therefore receive less than full economic credit; the model is never
asked to reproduce a hidden outcome exactly.

Invalid policy schemas earn zero. Boolean numbers, nonfinite numbers, overflow,
duplicate rows, unknown candidates and incomplete ledgers are rejected. Native
JSON parsing also rejects duplicate object keys and policies larger than 2 MiB.
Failed reproduction earns zero. The local `grade` command remains policy-only
and does not certify execution; use the native taskset for the complete protocol.

## Install and local checks

Run from the repository root with Python 3.12 and Docker available:

```sh
uv venv --python 3.12 environments/financial_research/.venv
uv pip install --python environments/financial_research/.venv/bin/python -e 'environments/financial_research[prime]'
source environments/financial_research/.venv/bin/activate
financial-research init-key --out /tmp/financial-research.key
export FINANCIAL_RESEARCH_KEY="$(cat /tmp/financial-research.key)"
financial-research self-check --count 16
python -m unittest discover -s tests -p 'test_research*.py' -v
RUN_RESEARCH_DOCKER_TESTS=1 python -m unittest discover -s tests -p 'test_research*.py' -v
vf-validate financial-research --runtime.type docker --taskset.num-tasks 4 -n 4 -c 1 -o /tmp/financial-research-validation
```

`init-key` creates a fresh file with owner-only permissions and refuses to overwrite
an existing file. Store a persistent evaluator copy outside the repository when results
must be reproducible. Never put the key in model prompts, runtime environment
variables, solver mounts, task metadata or uploaded traces. The trusted coordinator
and trusted grading workers must receive the same key. The generator refuses to
run without a key of at least 32 characters. Key rotation changes task identifiers
and generated data; it creates a different dataset.

For a public-only local export:

```sh
financial-research export --seed 0 --split train --out /tmp/financial-research-input
```

The command prints the task ID and prompt and writes only the four input files.
Run a candidate in a container containing those inputs, then grade its returned
policy on the trusted host with `financial-research grade TASK_ID policy.json`.
Do not give a model the entire repository, installed environment package or key.

## Prime integration

`FinancialResearchTaskset` is the exported v1 taskset. Solver inputs contain the
opaque task ID, prompt and four public files. No seed, family label, answer key or
continuation is carried in `TaskData`. Grading reconstructs the registered task on
the trusted worker, verifies the public projection, and stages trusted code and
private data into a fresh isolated verifier. Candidate Python is never executed
there. A third, independent runtime executes candidate analysis against public
data. Its root-owned controller launches the analysis under an unprivileged UID,
with a 256 MiB address-space limit, CPU/process/file limits, discarded console
output and a wall-clock deadline. It stops detached child processes before bounded
output capture. Candidate code cannot replace the controller or its result file.
All runtime policies restrict external egress; host mounts and forwarding the
evaluator key through runtime configuration are rejected.

`vf-validate` runs model-free setup, reference-artifact and no-op checks. The
reference validation hook uses a known policy and the independent public-input
reference source. It exercises artifact transfer, source execution, reproduction
comparison and private scoring. Empty submissions exercise the zero-reward path.

An instruct-model smoke test, after configuring a model endpoint, can use:

```sh
vf-run financial-research -m MODEL_ID -n 4 -r 1 -c 1 \
  --env.taskset.num-tasks 4 \
  --env.agent.runtime.type docker --env.agent.harness.id bash \
  --env.agent.max-turns 24 \
  --client.base-url ENDPOINT_URL --client.api-key-var MODEL_API_KEY \
  -o /tmp/financial-research-model-smoke
```

Replace the placeholders with an available instruct model and its endpoint. This
quick runner saves results locally; it does not publish the environment. Full
evaluation and training orchestration belong to the compatible `prime-rl` workflow.
Consult the [Verifiers taskset documentation](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/tasksets.md)
and [Prime evaluation documentation](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/eval.md).
Keep the installed API pinned when reproducing a run.

## Failure accounting

After the runner finishes, account for every planned attempt before comparing models:

```sh
financial-research report /tmp/financial-research-model-smoke --expected 4
```

For the proposed 12-episode, three-attempt calibration, the expected count is 36.
A trusted sibling `attempt_manifest.json` containing `{"expected_attempts":36}`
can declare this before a run. The report reads native Episode/Trace records,
including failed, dropped, malformed and missing attempts. Candidate artifacts,
logs and configuration files cannot supply the expected denominator.

Confirmed candidate failures contribute zero reward. Infrastructure failures and
unclassified framework errors remain unresolved; they do not become candidate
zeros or disappear from the report. Duplicate records, missing attempts, conflicting
counts or unknown planned coverage block the overall mean and return exit status 2.
`mean_reward_scored_only` is a conditional diagnostic, not a model comparison.
`passed` means scoring completed after reproduction, and can include partial
financial credit; full-credit counts are reported separately.

Native artifact collection can fail before task-specific grading, for example on
an unsafe archive. These failures remain explicit even when their cause cannot be
attributed automatically. Resolve or rerun them with recorded attempt identities
before accepting an overall result. Model-free `vf-validate` records are labelled
`validation_only` and never reported as model performance. Hosted training must
preserve this accounting policy; this package does not change an external trainer's
sample-dropping behavior.

## Splits and limits

The default registry has 128 training, 32 development and 32 test seed slots,
cycling through the four families. Family-specific runs are also possible within
each split. Treat related family variants as related generated episodes, not
independent evidence of broad capability. The splits use disjoint seed ranges;
they do **not** hold out new financial mechanisms or markets. Never train against
the development/test registry. Reserve a separate key for final evaluation when
developing new training generators.

This is a bounded research pilot: a fixed strategy grammar, mandatory trial ledger
and prescribed decision rule. Gross 0.5 and 1 are linear rescalings, so this does
not meaningfully test risk-budget optimization. It does not establish real-world
alpha, capacity, unrestricted signal discovery, statistical research judgment
or out-of-distribution performance. Reproduction checks consistency on the supplied
public episode; they do not prove the submitted algorithm generalizes, uses its
inputs honestly or avoids hard-coded answers. Missing features
carry forward the latest eligible observation; no maximum staleness filter is
implied. Prices have complete coverage and no delistings or dividends.

Docker's runtime does not enforce a total writable-disk quota. Per-file limits do
not bound the aggregate size of many files; use an enforced storage quota before
large adversarial training runs. Docker reproduction is exercised locally; hosted
Prime runtime behavior needs its own deployment validation.
No live model calibration or RL training is implied by reference-control success.

Next calibration work is an instruct-model pilot followed by targeted harder
episodes: weaker signals, correlated tests, regime changes, nonlinear execution
costs, risk-budget choices and mechanism-held-out evaluation. Keep these as new versions instead of changing
the frozen question banks or overwriting scored pilot runs.
