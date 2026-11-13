# Validation scope

This is an experimental evaluation release for one synthetic mechanism,
not a validated investment process or an empirically calibrated model benchmark.
No model API, external-model delegation or GPU was used for these checks.

## Completed mathematical and replay checks

- A separately implemented verifier was frozen before comparison with reference
  outputs. Eighteen hand-authored controls passed. Prior reference authorship was
  disclosed; this is separate implementation, not blind independent authorship.
- A different reviewer derived eight positive-probability sampling masks and
  56 feasible cash ledgers independently. The HT mean was 9 basis points; actual
  sampling variance and expected estimated variance both equalled 76 basis points
  squared. Every comparison matched.
- Fifteen generated condition/split reference crosschecks matched. These use
  public fixture keys and do not establish held-out model performance.
- Complete-population diagnostics matched seven independently derived ledgers,
  including the true-economics oracle and regret. Hidden diagnostics do not
  determine calculation reward.
- Eleven replay tests passed, including actual containers for current-policy
  reproduction and two fresh inputs, missing output, timeout and solver isolation.
  These tests overlap the package suite and must not be added to its test count.
- Malformed UTF-8 and oversized CSV fields now consistently raise public packet
  errors; regression controls cover both cases. Frozen bank scores are unaffected.
- Bounded rational arithmetic has a structural size argument in `CONTRACT.md`;
  it is not a worst-case runtime guarantee for arbitrary adversarial inputs.

## Native lifecycle

The native adapter gate passed 23 tests with no skips, including two real Docker
lifecycles. A reference submission passed bounded output capture, isolated native
grading and three source replays, with reward 1 and exact success on every packet.
A second lifecycle submitted a source symlink, an oversized policy and unrelated
large logs: only the two declared outputs were considered, the invalid artifacts
earned candidate zero, and no source replay occurred. Runtime/service failures
have separate controls and cannot silently enter the financial denominator.

Independent CPU review covers opaque identity recovery, spawned-worker resolution,
public task payloads, rejected runtime/scoring overrides, single replay under
concurrent hooks, strict-success metrics, and sanitized failure/cancellation.
The review reproduced and closed unsafe-archive collection, timeout-skip override
and inconsistent result-admission defects. It is assistant peer review, not an
external practitioner endorsement. The mathematical size argument was separately
checked; a conservative correct-policy serialization envelope is below 400,000
bytes, within the 2 MiB limit.

## Installed package

The package and its local replay dependency were built as wheels with the pinned
backend and installed noneditable into a fresh Python 3.12 environment, using the
suite's recorded runtime dependency constraints. Dependency checking passed.
Before release integration, the reviewed package, run outside the source checkout
with isolated Python imports, passed **63 tests with zero skips**, including every
real-container check. Its CPU-only run passed 58 tests with five explicitly skipped
container tests; these counts overlap and must not be added.

The release adds a public baseline command and focused baseline controls, plus a
native validation configuration. These additions do not change the nine reviewed
financial and runtime modules. The public command and exact invocation are in
[README.md](README.md); release-level checks include the added controls.

The integrated package suite contains 65 tests; its CPU-only run passed 60 with
five explicit container skips. A fresh noneditable suite installation passed
dependency checks and isolated verification of all eight packages, 52 resources
and ten native plugins. The installed baseline command reproduced the table below
from outside the checkout.

The documented `vf-validate` configuration also passed with the installed package,
a fresh worker key and an immutable local container image. Its single planned
task was recorded and terminal: the reference was valid, the empty submission was
valid with reward zero, and missing, unchecked, invalid, timeout and error counts
were zero. The fresh key was absent from saved outputs. This validates the scoped
local lifecycle, not an external model service or training job. See the suite's
[verification evidence](../../docs/verification.md) for the full release gate.

## Reward baseline

An outcome-blind cash policy was tested on 100 public-fixture packets: 20 in each
condition. It reconstructs release eligibility, support, counts and ESS, but never
reads markout values or solves the allocation objective.

| Condition | Mean shaped reward | Exact full-policy successes |
| --- | ---: | ---: |
| Positive | 0.52500 | 0/20 |
| Mixed | 0.54000 | 0/20 |
| Negative | 0.82500 | 0/20 |
| Unsupported | 0.74625 | 0/20 |
| Sparse | 0.73646 | 0/20 |
| All conditions | 0.67454 | 0/100 |

This follows the declared partial-credit rubric. Preserve it as a baseline;
report exact correctness and components before interpreting reward changes.
`python -B -m financial_selection.baseline` regenerates this table from its public
fixture key and prints aggregate results. A control changes every observed markout
and verifies that the cash policy's financial outputs remain identical; only its
required input digest changes.

## Remaining evidence limits

There are no live instruct/reasoning model results, independent practitioner
ratings, real-market signal validation or end-to-end optimizer/training results.
The finite generated families share a financial mechanism; split separation
does not demonstrate generalization to other markets or new mechanisms. Public
fixture keys and reference programs cannot produce contamination-free holdouts.
Private key custody and complete attempt accounting remain operator duties.
