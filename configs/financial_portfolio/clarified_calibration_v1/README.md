# Clarified portfolio development calibration v1

This separately frozen calibration uses the unchanged
`financial-portfolio-clarified` 0.5.0 taskset and
`financial-portfolio-staged` 0.4.0 harness. It retains the clarified pilot's public
instructions, checkpoints, final scorer and execution settings, and expands to
the development schedule originally specified in
[`calibration_v1`](../calibration_v1/README.md). It does not amend either earlier
condition or its results.

## Fixed schedule

| Split | Seeds | Episodes | Attempts per episode | Total attempts |
| --- | --- | --- | --- | --- |
| Development (`dev`) | 128–159 inclusive | 32 | 2 | 64 |

The default family rotation supplies eight episodes and 16 attempts per mechanism:

| Mechanism | Seeds |
| --- | --- |
| `impact_sizing` | 128, 132, 136, 140, 144, 148, 152, 156 |
| `liquidity_tight` | 129, 133, 137, 141, 145, 149, 153, 157 |
| `risk_concentration` | 130, 134, 138, 142, 146, 150, 154, 158 |
| `no_signal` | 131, 135, 139, 143, 147, 151, 155, 159 |

These development episodes are distinct from pilot training seeds 4–7 and the
earlier smoke seeds 0–3. They use the same mechanisms and generator assumptions.
This is a development calibration, not the reserved test set. Test seeds 160–191
remain excluded. Two attempts on one episode are related measurements: there are
32 episode pairs, not 64 independent financial problems. A claim that episodes
were unseen by a model additionally requires checking the prior run history.

Use `moonshotai/kimi-k2-0905`, the model that completed the prerequisite and
clarified pilot. Bind its endpoint and available deployment metadata before live
execution. The model label does not pin a gateway deployment. A model, prompt,
tool, generation, scoring or budget change requires another labelled condition.
Do not select an attempt based on its score or replace a failed attempt.

## Execution and binding

Task selection is ordered, with concurrency two. Each attempt has 32 turns,
65,536 output tokens, 262,144 total tokens, 8,192 tokens per response and
temperature 0.2. Episode, agent, scoring, SDK and stream retries are zero. Docker
uses `python:3.12-slim`, no network allowlist entries and no forwarded host
environment variables; scoring is isolated. Preserve the pilot's resolved image
digest, evaluator-key fingerprint and pinned dependencies. These ceilings do not
constitute a monetary budget; retain actual provider usage and cost coverage.

Before calls, record the source revision, protocol digest, exact local bound
configuration, dependency versions, image identity, endpoint/model metadata,
evaluator-key fingerprint and helper-file hashes. Record all 32 task IDs with
their seed/family, public-file hashes, user/system-prompt hashes and two expected
attempts each. Verify the complete schedule and the inherited pilot readiness
gate: all four pilot attempts accounted for, functioning infrastructure and at
least one verified source reproduction. The gate imposes no financial-score
threshold.
Credentials and evaluator secrets remain only on the trusted host. Solver context
contains public inputs only. Keep live configurations, traces, artifacts and
reports outside the repository, in a fresh run directory.

Preserve all 64 scheduled attempts, including protocol failures, malformed
outputs, missing attempts, provider failures and unresolved outcomes. No ad hoc
response repair, automatic retry or selective rerun belongs to this batch. If an
interruption prevents completion, report the unfilled attempts explicitly;
diagnostic work is separate and does not replace original observations. Do not
change the condition between the two attempts on an episode.

## Preregistered reporting

Account for all 64 attempts first, and verify exactly two attempts for each of the
32 planned task IDs. Generic attempt accounting alone does not verify that
schedule. Report candidate failures, infrastructure errors, malformed records,
missing attempts and unresolved outcomes separately. Preserve failed calls and
missing usage. An unresolved attempt makes the aggregate performance estimate
unavailable; it is not silently dropped or converted to a financial zero.
Explicitly scored candidate failures remain in the reward denominator at zero.

Report overall and per-family results for:

1. Ingestion, trials and submission checkpoint completion, then source-artifact
   validity and fresh-container reproduction. Structural success is separate from
   correct finance.
2. Full-credit attempts (exact official reward of 1) out of 64 overall and 16 per
   family, continuous financial reward, requested audit accuracy,
   trial accuracy, decision correctness and withheld economic quality. Report
   missing component metrics as missing; do not invent zeros for unavailable data.
   Requested audit accuracy includes both source and portfolio audits. Correct
   public audit, trial and decision outputs do not guarantee full credit when
   withheld economic quality reduces the final reward.
3. The mean of the two rewards within each episode, followed by the distribution
   of those 32 episode means: mean, median, minimum, maximum and quartiles. Family
   summaries use eight episode means each. Show both attempts and their absolute
   within-episode reward differences; native completion order does not identify
   a first or second replicate. Report both/one/neither full-credit and reproduction
   counts across episode pairs. Report exact decision and selected-candidate
   agreement only among episodes with two valid scored policies, and state that
   eligible denominator. If no pair is eligible, report agreement as unavailable;
   this secondary metric does not change the accounting gate. Two absent policies
   are not agreement. Do not select the
   better score or report pass@k for
   continuous reward. An episode with an unresolved attempt has no complete mean.
4. Failure reasons, token usage, elapsed time and provider-reported costs, with
   missing usage and cost coverage explicit. Report deployment and abstention
   decisions separately, including the `no_signal` mechanism.

Full-credit counts describe exact outcomes; the continuous reward distribution
retains partial performance. Neither establishes real-market profitability,
population accuracy or independent replications across shared mechanisms. This
run is not a paired before/after comparison against the four pilot episodes.

Review every failure and a sample of successes. Distinguish scorer defects,
ambiguous instructions, shortcuts, workflow failures and substantive financial
errors. Freeze these findings before any targeted edits. Corrected cases or
scoring belong to a new version; retain this calibration and leave reserved tests
unused during development.

## Preparation and integrity

The checked-in TOML remains dry-run-only with an invalid endpoint and model
placeholder. Validate the native configuration and local protocol checks, then
freeze before live execution:

```sh
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_portfolio_clarified_calibration_protocol.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/clarified_calibration_v1/development.toml --dry-run
python3 configs/financial_portfolio/clarified_calibration_v1/freeze_manifest.py --write
python3 configs/financial_portfolio/clarified_calibration_v1/freeze_manifest.py --check
```

The manifest binds this protocol and its test, the clarified pilot manifest and
every frozen predecessor payload. `--write` refuses to overwrite a manifest.
After review and binding, the authorized live run uses a fresh local config with
`vf-run @ LOCAL_CONFIG --dry-run false`. No publication or model training is part
of this calibration.
