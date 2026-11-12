# Experimental portfolio validation

Applies to `financial-portfolio` 0.3.0, `financial-research` 0.2.0 and Verifiers 0.4.1.
These are software and synthetic-mechanism checks, not model evaluation results.

| Check | Result |
| --- | --- |
| Hand-calculated trusted portfolio regressions | 10 passed. |
| Independent public solver and mechanism regressions | 15 passed, including four-family subprocess parity. |
| Native adapter, privacy and frozen dependency checks | 12 passed. |
| Native Docker `vf-validate`, four families | Four reference and four empty submissions passed; no errors, timeouts or missing outcomes. |
| Frozen calibration protocol | Four integrity/schema regressions passed; both native `vf-run` dry-runs completed without task generation or model calls. |
| Repository `make check` | 224 case definitions and 2,163 controls passed; 106 tests passed and 42 optional tests skipped in the base environment. |
| New suite with pinned native dependency | All 41 tests passed; none skipped. |
| Frozen packages | Research v0.2.0 resources are byte-identical; original question and evaluator banks unchanged. |
| Wheel and source distribution | Built successfully; wheel installation and local CLI checked separately. |
| Native validation output privacy | Actual evaluator-key bytes absent from output files. |

The native reference checks exercise solver staging, source reproduction in a fresh
public-only container, and isolated private grading. Expected reference reward is
computed from the actual selected policy; it need not be one. Empty submissions
receive zero. Runtime and scoring infrastructure is inherited from the frozen
dependency rather than copied into this experiment.

Regressions cover sample covariance and training boundaries, separate long/short
inverse-vol normalization, common risk scaling, convex impact, actual fills and
borrow, neutrality, participation limits, entry and exit costs, fixed denominator,
full liquidation, missing signals, no exposure drift, numeric tie canonicalization,
split adjustments and reward gates. Negative controls detect omitted costs,
ignored liquidity and risk estimation using validation returns.

## Mechanism check

An evaluator-keyed batch used training seeds 0–31: eight instances of each mechanism,
with six independent withheld paths per instance. No development or test seeds were
used for this check. Generation parameters were not retuned after inspecting it.

| Mechanism | Validation selections | Best mean withheld score |
| --- | --- | --- |
| Impact-sensitive sizing | Three quarter-size, one half-size, four full-size; both allocations appear. | Three quarter-size, five full-size. |
| Tight liquidity | Four quarter-size, four half-size; both allocations appear. | Six half-size, two full-size. |
| Concentrated risk | Eight full-size inverse-vol allocations. | Eight full-size inverse-vol allocations. |
| No signal | Eight abstentions. | Eight abstentions. |

Every instance had 48 distinct executed exposure paths when coordinates were rounded
to ten decimals for the diagnostic comparison. This is an empirical result for this
batch, not a guarantee for every seed. The generator reports effective policy count,
the gap to the next candidate label, and the gap to the next distinct executed path.
Independent seeds and paths are related instances of four mechanisms, not independent
financial research problems. The full-size risk-constrained policies can execute at
substantially lower exposure than their nominal requested gross.

The differing validation and withheld winners are expected under sampling noise and
path-dependent execution. The batch demonstrates that size and allocation affect
the task; it does not establish model difficulty, profitability or out-of-distribution
generalization. An instruct-model calibration, independent practitioner review and
hosted deployment checks remain outstanding.

## Reproduce

Install the two local packages using [README.md](README.md), set an evaluator-only
key, then run:

```sh
make check
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_portfolio*.py' -v
.venv-portfolio/bin/financial-portfolio self-check --count 32
.venv-portfolio/bin/vf-validate financial-portfolio --runtime.type docker --taskset.num-tasks 4 -n 4 -c 2 -o /private/tmp/portfolio-validation
uv build environments/financial_portfolio --out-dir /private/tmp/portfolio-build
```

Different evaluator keys produce different data and selection frequencies. Keep key
files, exported episodes and native validation outputs outside the repository and
candidate access. The generation key used for the reported mechanism batch is not
distributed with this package.
