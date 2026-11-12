# Pilot validation record

Applies to the `financial-research` 0.2.0 implementation and Verifiers 0.4.1.
These are software/reference checks, not model-performance results.

| Check | Result |
| --- | --- |
| Frozen bank definitions and reference/negative controls | Passed; 2,163 controls, baseline packages unchanged. |
| Public-input solver and core grader regressions | 16 passed. |
| Native adapter contract and artifact checks | 20 passed, including an opt-in real Docker test. |
| Fresh-container reproduction and privacy regressions | Nine passed, including real Docker execution, timeout, detached-child, Unix-socket and error-trace probes. |
| Failure accounting and report CLI | 23 passed, including malformed, dropped, duplicate and missing attempts. |
| Native `vf-validate`, four generated families | Four reference submissions and four empty submissions behaved as expected through the solver, public rerun and private verifier; no errors or timeouts. |
| Wheel and source distribution build | Passed; wheel installed and imported in a separate Python 3.12 environment. |
| Native taskset loading | Passed against the pinned package. |
| Evaluator key in native validation output files | No matches for the actual secret bytes. |

All 68 pilot tests were exercised across the unit/API and opt-in Docker runs.
Core regressions cover late publications, revised data, splits, dated ticker reuse,
transaction costs, borrow, turnover, Sharpe conventions, missing/duplicate outputs,
large integers, boolean/nonfinite numbers, reward gates, split separation and key
rotation. Adapter probes cover isolated placement, public/private separation,
malformed policies, absent or invalid source artifacts, successful source
reproduction in a separate public runtime, and non-execution of candidate analysis
in the private verifier. Provider exception-chain tests use native Trace error
serialization and confirm sentinel secret text is excluded.

Accounting checks use the actual v1 Episode/Trace representation and verify that
duplicate UUIDs across directories, copies of a trace inside and outside an Episode,
and candidate-owned attempt manifests cannot manufacture complete results. Confirmed
candidate failures receive zero; unresolved infrastructure and unknown failures
block overall performance. Both report CLI exit paths are covered.

The normal repository `make check` completed with 77 passes and 30 skips in the
base Python environment. The pilot suite with the pinned Prime dependency completed
with 64 passes and four opt-in Docker skips; all four were exercised separately.
The five pre-existing workflow Docker tests were not rerun for this addition.

No live model evaluation, practitioner validation, RL training or hosted environment
deployment is represented by these results. Artifact collection errors remain
visible, including errors before the task's grading hooks. Reproduction verifies
same-episode output consistency, not algorithmic generalization. Hosted runtime,
external trainer accounting and aggregate writable-disk limits require additional
deployment checks before large adversarial training runs.

Run instructions and the remaining scope limits are in [README.md](README.md).
