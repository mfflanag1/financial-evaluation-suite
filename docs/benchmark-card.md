# Benchmark card

## Purpose and unit of evaluation

The suite tests whether a solver can apply supplied financial rules, reconcile evidence, compute quantities and maintain consistent decisions as information changes. It is intended for reproducible research and evaluation development. It does not validate investment advice or predict live strategy returns.

The main bank has 224 cases: 36 source-grounded, 100 quantitative, six executable workflows, 58 research scenarios and 24 decision-core cases. Six adaptive variants are separate diagnostics. The main bank declares 1,924 structured outputs. An output field, sequential turn or new attempt is not an additional independent case. Correlated variants should be grouped with their parent mechanism in uncertainty estimates.

## Data and exposure

Source-grounded cases use retained public-source excerpts and authored evidence anchors. Other tracks use authored scenarios and synthetic data. Publication time, knowledge cutoffs and retrospective availability differ between cases; use each packet's stated information conditions.

All committed evaluator material is publicly inspectable. Historical `development` and `holdout` fields describe original protocol assignments, not current secrecy. Static-bank scores are public-reference results, not contamination-free benchmark performance.

Generated environments use evaluator-held keys and separately registered split ranges. Fresh keys produce new realizations, while the generator and financial mechanisms remain public. Keep keys, private continuations and grading workers outside candidate access. A key is not a substitute for an isolated runtime.

## Financial coverage

Coverage includes filing reconciliation, corporate actions, contracts, liquidity, credit, rates, currencies, energy, derivatives, execution, portfolio repair and research validity. Quantitative cases include Sharpe conventions, covariance and risk attribution, statistical corrections, discrete optimization and capacity constraints.

The generated workflow tests shuffled records, dated security mappings, stock splits, missing releases, duplicates and revisions. It evaluates a declared family of candidate signals and portfolio policies. This is narrower than unrestricted signal discovery or a complete production research platform.

The separate selection workflow tests selective outcome observation, dated population reconstruction, sampling-adjusted estimates, disclosed population-mixture changes, shared estimation covariance and funded integer allocations. Its program is checked on three distinct input packets. These are prescribed synthetic calculations, not evidence that missing returns can be recovered exactly or that the resulting allocation will profit in live markets.

## Scoring and verification

Use a versioned public export, matching grader and matching session controller. Report the scorer version, content binding, case IDs, information condition and run settings. Preserve baseline results when regrading saved responses.

Report financial correctness, evidence support, structural validity, action feasibility, program reproduction and economic outcomes separately. A valid response may be financially wrong; correct arithmetic may cite insufficient evidence; a reproducible policy may have poor withheld-path performance.

| Level | Evidence supplied | Not established |
| --- | --- | --- |
| Manifest and schema checks | Content integrity and contract consistency | Correct financial reasoning |
| Reference replay and negative controls | Expected scorer behavior on fixtures | An independent answer derivation |
| Independent calculation | Separate calculation using task inputs | Practitioner endorsement or market validity |
| Scripted native rollout | Local tool, artifact and scoring behavior | Language-model capability |
| Model evaluation | Performance under a specified protocol | Training success or profitability |
| Optimizer integration | An actual learning step with recorded rewards | Generalization to unseen mechanisms |

Case-level audit records identify completed levels. A catalog check alone is not numerical certification.

## Known limitations

- Public exposure limits static-bank ranking claims.
- Authored and synthetic rules simplify markets and may reward contract following more than open research judgment.
- Source scoring matches retained quotes and authored anchors; it is not a general entailment model.
- Source redistribution is restricted to exact excerpts accepted by the source policy.
- Sequential branches, variants and repeated seeds are statistically dependent.
- Generated portfolio policies use bounded menus; prescribed selection is not a multiple-testing correction.
- A correct public decision can receive less than maximum reward when the economic component uses withheld paths.
- Partial-credit rewards can be substantial for cheap baselines; selection results must include exact correctness and the published cash baseline.
- Local containers are not a certification for hostile multi-tenant hosting.
- Practitioner review and broadly replicated model calibration remain separate tasks.

See [evaluation-protocol.md](evaluation-protocol.md) and [rl-readiness.md](rl-readiness.md) for reporting and validation gates.
