# Financial evaluation suite

A public reference suite for financial reasoning, quantitative analysis and reproducible research workflows. The frozen bank contains **224 cases and 1,924 structured outputs**, plus **six related diagnostic variants**. Separate generated environments test data repair, signal selection, costs, portfolio constraints and saved research programs.

Use the static bank for regression tests, teaching and transparent evaluation development. Its questions **and evaluator answers are public**, including cases with historical `holdout` labels. It is not an uncontaminated held-out benchmark. Keep evaluator material outside solver context during a controlled run; that does not undo prior public exposure.

## Start here

- [Quickstart](docs/quickstart.md): verify the bank, export a question and grade a response.
- [Benchmark card](docs/benchmark-card.md): coverage, counting rules and limitations.
- [Case catalog](docs/case-catalog.md): browse every case and its recorded review status.
- [Evaluation protocol](docs/evaluation-protocol.md): information boundaries, metrics and reporting.
- [RL readiness](docs/rl-readiness.md): generated environments and remaining training gates.
- [Source policy](docs/source-policy.md): evidence redistribution and conservative packet export.
- [Corrected scorer](scoring_v3/README.md): versioned changes over the frozen baseline.
- [Practitioner review packet](docs/practitioner-review.md): forwarding note, case selection and blank review record.
- [Verification evidence](docs/verification.md): completed checks and the claims they support.

## Coverage

| Track | Cases | Typical work |
| --- | ---: | --- |
| Source-grounded | 36 | Reconcile filings, contracts, corporate actions and release vintages with cited evidence. |
| Quantitative methods | 100 | Portfolio risk, Sharpe ratios, estimation, derivatives, allocation and execution calculations. |
| Executable workflows | 6 | Submit a program and reproduce its outputs on controlled inputs. |
| Research scenarios | 58 | Thirty single-turn and 28 multi-turn cases, with 142 scored turns in total. |
| Decision core | 24 | Information acquisition, sequential decisions, execution state, funding and portfolio repair. |
| Related diagnostics | 6 | Adaptive variants, counted separately from the main bank. |

The generated environments are separate from these counts. Repeated seeds, turns, output fields and attempts do not increase the number of independent financial mechanisms.

The [case-level audit](quality/README.md) independently recomputes 119 cases and 539 declared outputs, including all 100 quantitative cases. It records the remaining 111 entries as lacking independent recomputation. [Alternate methods](quality/crosschecks/README.md) provide additional reproducible checks for five of those cases and 28 outputs. This is scoped mathematical evidence under the supplied assumptions, not practitioner validation or empirical trading-performance evidence.

## Verify and use

From the repository root with Python 3.11 or later:

```sh
make check
python3 -B -m scoring_v3.grade validate
python3 -B -m scoring_v3.grade export q076 > question.json
python3 -B -m scoring_v3.grade grade q076 response.json
```

Construct `response.json` using the exported response contract. The base checks need no credentials or model service. Native integration tests require Python 3.12, pinned environment dependencies and Docker; skipped integration tests are not a passing runtime validation. See the [quickstart](docs/quickstart.md) for explicit verification tiers.

`questions/` and `private_evaluator/` are frozen. Scorer v3 adds exact semantic-integer checks and duplicate-citation handling over the preserved v2 corrections. Corrections are opt-in and separately versioned. Keep original results, corrected regradings and new runs distinguishable. Local reference checks establish implementation consistency; they do not establish model difficulty, trading performance or independent practitioner approval.

## Generated environments

The [research pilot](environments/financial_research/README.md) generates messy synthetic observations with point-in-time joins, revisions and costs. The [portfolio environment](environments/financial_portfolio/README.md) adds nonlinear impact, funding and covariance constraints. The [staged protocol](environments/financial_portfolio_staged/README.md) enforces checkpoints, and [prerequisites](environments/financial_prerequisites/README.md) test basic program reproducibility. [Clarified instructions](environments/financial_portfolio_clarified/README.md) and [operational controls](environments/financial_portfolio_operations/README.md) retain separate version identities. The [runtime integration](environments/financial_portfolio_runtime/README.md) binds task schedules, provider requests, tool lifecycles and isolated grading in a separately frozen protocol.

These evaluation integrations target native Verifiers 0.4.1 tasksets. The separate [training adapter](training/README.md) targets exact later trainer revisions in its own dependency environment. Its 41 local tests include actual spawned workers, Docker, scripted training-token transport and failure admission. A controlled rollout, isolated financial grading and fresh-container reproduction are distinct gates. No actual model training, optimizer update or checkpoint reload has been demonstrated.

## Evidence and reuse

Publication dates, availability times and financial conventions are part of the tasks. Source manifests bind retained excerpts to their provenance; they do not grant redistribution rights. The [source policy](docs/source-policy.md) distinguishes reviewed excerpts from link-only material. The conservative exporter is the supported way to prepare a source packet; a complete repository clone has a broader evidence footprint.

The project is currently **unlicensed**: no project-level reuse license has been granted. Public repository access does not imply blanket reuse permission. Project licensing and third-party source permissions are separate questions; required third-party notices remain with their respective components.
