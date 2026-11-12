# Staged protocol validation

Applies to `financial-portfolio-staged` 0.4.0 with portfolio 0.3.0, research 0.2.0
and Verifiers 0.4.1. Validation records are evaluator-local and are not distributed
with this package. These checks establish software behavior, not model performance.

| Area | Checks exercised |
| --- | --- |
| Structural validation | Original-byte inventories, exact schemas, unique coverage, integer types, duplicate keys, nonfinite numbers, depth limits and exact file-size boundaries. |
| Source handling | UTF-8, nonempty Python syntax, missing files and unsafe archives; checkpoint validation does not execute source. |
| Coordinator | Ordered single-call checkpoints, committed assistant messages, concurrent runtime isolation, final artifact hashes and resistance to writable-state changes. |
| Protocol failure | Malformed arguments preserve their original bytes, block the whole tool batch and produce a zero-reward candidate failure. |
| Harness | Native endpoint/tool-gate wiring, bash/edit execution, unsupported-setting rejection, pinned dependency installation and retained third-party license. |
| Transport | Local fake HTTP 429, HTTP 500 and broken-stream fixtures make one request without retry. |
| Native Docker validation | Reference submissions exercise staged validation, isolated financial scoring and fresh-container reproduction; untouched submissions exercise the zero-reward path. |

All four final artifacts use bounded, individual collection. Transfer-limit violations
record an omitted payload and candidate failure; unrelated collection errors retain
their infrastructure classification. Regression coverage distinguishes these paths.
Reference submissions bypass model behavior and cannot demonstrate that a model can
complete the stages. Their expected reward comes from the unchanged financial scorer
and need not equal one.

## Reproduce

Install the three packages and load an evaluator-only key as described in the README.
From the repository root:

```sh
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_staged*.py' -v
RUN_STAGED_DOCKER_TESTS=1 .venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_staged_harness.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/staged_v1/smoke.toml --dry-run
.venv-portfolio/bin/vf-validate financial-portfolio-staged --runtime.type docker --taskset.seed-start 4 --taskset.num-tasks 4 -n 4 -c 2 -o /private/tmp/portfolio-staged-validation
uv build environments/financial_portfolio_staged --out-dir /private/tmp/portfolio-staged-build
```

The opt-in Docker test installs and exercises the actual bundled dependencies against
local fake endpoints; it does not call a live model. Without that flag, the Docker
bundle test is skipped. Native adapter tests require the installed pinned dependency.

Live runs require separate attempt accounting, usage coverage and sample-level review.
The four-seed diagnostic is neither a paired comparison with earlier protocols nor
an estimate across independent financial research problems. Structural completeness
does not establish financial correctness, strategy profitability or generalization.
