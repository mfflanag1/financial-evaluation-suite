# Case-level quality evidence

The inventory covers **224 main cases and six related variants**, with 1,924 main outputs and 48 variant outputs. Every entry has an explicit review status. Generated environment seeds, related variants and repeated attempts are not additional independent static cases.

**All 230 entries have independent numerical recomputations**: 100 quantitative cases, 24 core cases, 36 retained-source cases, six executable workflows, 58 research scenarios and six related variants. All **1,972 declared outputs**, including every authored turn, match the stored references within their specified tolerances. Arithmetic implementations consume public question inputs and governing contracts, separately from stored solutions and production grading algorithms. Derived results were frozen before reference comparison. One source-audit field was subsequently converted from dollars to the public schema's required millions of dollars; its certificate records that correction. The frozen reference was unchanged.

This establishes the scoped calculations under stipulated assumptions. It does not establish market realism, empirical signal efficacy, source licensing, adversarial scorer robustness, practitioner validation or full coverage of sequential paths. Multi-turn checks cover authored public snapshots and cumulative disclosed amendments. They do not validate arbitrary candidate-action trajectories, simulator transitions or executable workflow hidden-input replay.

Later turns share files with evaluator answers. An allowlisted [projection helper](public_contracts.py) copies only public context; core output templates contribute field names and top-level types, without answer values. Corruption controls verify that changing stored answer values or solution metadata cannot change those projections or numerical derivations. Retained-source transcriptions are bound to exact question/excerpt hashes; they are fixed-version reviews, not general document parsers. Audit revision `1.1.0` extends the evidence without changing frozen scoring rules.

## Read the evidence

| Artifact | Purpose |
|---|---|
| [inventory.json](inventory.json) | All 230 entries, track, stage schemas, input/reference hashes, public exposure and verification status |
| [independent_checks.json](independent_checks.json) | Per-case methods, recomputed fields, reference comparisons and numerical certificates |
| [numerical.py](numerical.py) | Independent quantitative calculations, with no scorer imports |
| [allocation_numerical.py](allocation_numerical.py) | Financing ledgers, constrained allocations, active sets and convex optimality certificates |
| [risk_numerical.py](risk_numerical.py) | Conditional tails, collateral, netting, settlement and stress identities |
| [remaining_numerical.py](remaining_numerical.py) | Finite contingent policies, rates, execution and capacity calculations |
| [core_numerical.py](core_numerical.py) | Independent portfolio repair, temporal-eligibility and joint-sign calculations |
| [decision_numerical.py](decision_numerical.py) | Adaptive paid research, accepted order state, settlement and constrained cash decisions |
| [source_numerical.py](source_numerical.py) | Retained-excerpt arithmetic and declared conventions; no external source revalidation |
| [advanced_source_numerical.py](advanced_source_numerical.py) | Multi-document EPS/cash reconciliation, acquisition perimeter, conversion entitlements and separate basis blocks |
| [source_remaining.py](source_remaining.py) | Remaining 25 source cases: corporate actions, financing, energy, cash bridges and data-vintage portfolios |
| [core_remaining.py](core_remaining.py) | Remaining 16 core cases, including conditional decisions and 52 authored state snapshots |
| [variant_numerical.py](variant_numerical.py) | Exact backward induction for six related information-acquisition diagnostics |
| [workflow_numerical.py](workflow_numerical.py) | Six public workflow calculations, including cash, covariance and ridge-estimation controls |
| [research_numerical.py](research_numerical.py) | All 58 research scenarios and 142 authored turns, with cumulative public amendments |
| [crosschecks/](crosschecks/README.md) | Second methods for five already audited cases and 28 outputs, with input hashes and finite-search certificates |
| [discrete_fields.json](discrete_fields.json) | Semantic review of all 1,972 stage/output fields, including mixed-array selectors |
| [defects.json](defects.json) | Confirmed scorer-v2 defect reproductions and remaining audit scope |

The independent methods include exact rational discrete probabilities, complete finite-choice searches, covariance identities, residual orthogonality, path-dependent pricing, cash-flow ledgers, and absorbing first-passage probabilities. Tests check invariants and alternate inputs as well as reference agreement. A deliberate reference corruption test confirms that recomputation does not simply return stored answers.

## Reproduce

From the repository root, using Python 3.11 or later and no external services:

```sh
python3 -B -m quality.inventory --check
python3 -B -m quality.discrete --check
python3 -B -m quality.reproduce_defects
PYTHONPATH=private_evaluator python3 -B -m unittest discover -s tests -p 'test_quality_*.py' -v
```

`--check` is read-only. Omit it to regenerate audit JSON after an intentional, reviewed change. Structural metadata and source hashes must match exactly. Independent floating-point evidence comparisons allow only small platform variation (`1e-8` absolute plus the `math.isclose` relative rule at `1e-9`); financial reference comparisons continue to use each question's declared tolerances.

The defect reproduction command intentionally demonstrates acceptance of fractional semantic integers by preserved `scoring_v2`. A passing reproduction test means the defect is reproducible. It does not mean that v2 was repaired. Successor scoring must be labelled separately.

The discrete inventory approves 201 stage/field rules. All 185 initially pending quantity-like fields have been reviewed: six gained integer rules, 153 need no additional integer constraint, and 26 require a clarified contract or a separate closure proof before stricter enforcement. A global prompt review added 12 exact integer-thousand payroll rules. The remaining fields include seven explicit continuous exclusions and 1,585 outputs without an identified integer requirement. Only `approved_integer` rows recommend integral validation. Unit labels such as shares or months are insufficient by themselves: a continuous order-size limit and an exponential half-life remain continuous. Selectors allow integer leaves inside otherwise mixed arrays. Fractional values and JSON booleans fail integer validation; exactly integral JSON numbers such as `2.0` may pass.

## Public benchmark boundary

All answer keys and reserved-labelled case material in this repository are published. A missing question-directory packet does not make a case private when its full evaluator specification is present. These cases are usable as public reference and training material, but no entry in this inventory is represented as a secret or contamination-free holdout. Keep these evaluator-side audit artifacts out of a solver's assigned context; that context restriction does not undo prior public exposure.
