# Constrained portfolio research experiment

`financial-portfolio` 0.3.0 is a separate experimental environment. It adds nonlinear
execution costs, liquidity limits, training-based covariance constraints and sizing
choices to a point-in-time synthetic research task. The frozen `financial-research`
0.2.0 package and the original evaluation banks are unchanged.

The four mechanisms are impact-sensitive sizing, tight liquidity, concentrated risk
and no signal. Each episode declares 48 policies: four features, two directions,
three gross sizes and two allocations (equal and inverse volatility). These are
bounded research choices, not unrestricted strategy discovery. Constraint-equivalent
labels can produce identical executed paths; diagnostics report distinct paths.

## Local installation

Python 3.12 or later. From the repository root, install the sibling packages together:

```sh
uv venv .venv-portfolio
uv pip install --python .venv-portfolio/bin/python ./environments/financial_research './environments/financial_portfolio[prime]'
```

The experimental package pins `financial-research==0.2.0` and its optional native
adapter pins `verifiers==0.4.1`. Both source packages are needed unless the pinned
dependency is already installed. No hosted publication or live model run is required
for the local commands below.

Create an evaluator key outside the repository and solver inputs, then load it only
in the evaluator shell. An existing pilot key can be reused: the experimental IDs
and random paths use a distinct versioned namespace.

```sh
.venv-portfolio/bin/financial-research init-key --out /private/tmp/portfolio-evaluator.key
export FINANCIAL_RESEARCH_KEY="$(cat /private/tmp/portfolio-evaluator.key)"
.venv-portfolio/bin/financial-portfolio self-check --count 4
.venv-portfolio/bin/financial-portfolio export --seed 0 --out /private/tmp/portfolio-public
```

Only `prices.csv`, `features.csv`, `securities.csv` and `contract.json` belong in the
solver workspace. Copy the installed `financial_portfolio/reference.py` there as
`research.py` for a model-free example and run it in that directory. It uses only
the Python standard library and public files. Grade its policy using the opaque id
printed by export:

```sh
.venv-portfolio/bin/financial-portfolio grade TASK_ID /private/tmp/portfolio-public/policy.json
```

Direct grading checks structured outputs. Native grading also requires the submitted
source to reproduce those outputs in a fresh public-only container. It inherits
the artifact handling, separate private verifier, failure accounting and bounded
reproduction controller from the frozen pilot. See its [runtime documentation](../financial_research/README.md).

With Docker available, the native model-free check is:

```sh
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_portfolio*.py' -v
.venv-portfolio/bin/vf-validate financial-portfolio --runtime.type docker --taskset.num-tasks 4 -n 4 -c 2 -o /private/tmp/portfolio-validation
```

For future model rollouts, use `financial-portfolio report OUTPUT --expected N` to
retain missing and failed attempts in accounting. Model-free validation records are
identified separately and cannot establish model performance.

## Financial contract

The full machine-readable rules accompany every episode. Returns use split-adjusted
prices and dated security mappings; signals use release-time eligibility and revision
ordering. A sample covariance from 72 training returns is shrunk halfway to its
diagonal, then held fixed for both validation and withheld paths. The strategy
allocates each long and short leg separately, scales the entire target to gross,
per-asset and estimated-volatility limits, and applies one common execution fraction
to satisfy every asset's participation cap while retaining neutrality.

Costs include a one-way spread, square-root participation impact on actual trades,
and borrow on executed shorts. Exposures are explicitly fixed-reference-capital
return notionals; they do not model drifting physical share holdings or compounding
wealth. After 48 validation intervals, a fixed number of zero-market-return synthetic
closeout intervals liquidates every policy under the same execution limits and costs.
All policies use the same full-episode denominator, including already-flat intervals.
This closeout convention is a controlled experiment, not a real-market liquidation
forecast. Volatility limits constrain the frozen model estimate, not realized risk.

Scoring checks both source-data reconstruction and portfolio checkpoints, all trial
metrics, the validation decision and six independent withheld paths. Screening scores
are rounded to 12 decimals before lexicographic tie breaking. Perfect public analysis
can earn less than full total reward if its selected policy underperforms on withheld
paths. The screening statistic is a prescribed heuristic, not a significance test.

## Validation and remaining work

See [VALIDATION.md](VALIDATION.md) for checks actually performed. The finite registry
has 128 training, 32 development and 32 test seeds; explicit family overrides yield
768 distinct identifiers. Those are seed-disjoint variants of four mechanisms,
not 768 independent research problems or mechanism-held-out tests. Keep evaluator
keys and packages out of candidate contexts; visible generator code and disclosed
keys are unsuitable for contamination-resistant assessment.

The [frozen calibration protocol](../../configs/financial_portfolio/calibration_v1/README.md)
provides dry-run configs for four smoke attempts and 64 development attempts.
Live instruct-model calibration, practitioner review and hosted RL remain pending.
The local fixed-notional simulator omits order books, intraday scheduling, changing
ADV, funding/margin and wealth dynamics. Same-episode source reproduction is not a
test of algorithmic generalization. Docker disk quotas and external trainer failure
accounting require deployment checks before adversarial training at scale.
