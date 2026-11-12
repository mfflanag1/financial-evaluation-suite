# Portfolio clarification protocol v1

This separately frozen condition tests one public wording change with
`financial-portfolio-clarified` 0.5.0. The task adds one system paragraph defining
the direction of the existing split factor and the adjustment formula:

> shares_per_original is the cumulative number of current shares corresponding to
> one original share; close is price per current share. Define adjusted_close =
> close * shares_per_original.

All public files, question text, trial definitions, checkpoint rules, reproduction
requirements and financial scoring remain those of the frozen
[`staged_v1`](../staged_v1/README.md) condition. The harness remains
`financial-portfolio-staged` 0.4.0. Existing attempts and scores are preserved.

## Fixed comparison and interpretation

Run the same `moonshotai/kimi-k2-0905` model on training seeds 4–7, in order, with
one attempt per episode and concurrency two. These are the same four episodes
used in that model's earlier staged portfolio pilot after it passed the frozen
prerequisite condition. Pair by episode ID and public-input digest; retain the
same evaluator-key fingerprint. Record the earlier traces and summary digests in
the trusted local preflight before any new call. Record the requested model,
endpoint, current deployment metadata when available, image digest, installed
dependencies, source revision, protocol digest, bound config and prompt digests.

The comparison is descriptive. Four stochastic attempts against a historical
gateway baseline cannot establish that clarification caused any score change,
nor estimate general model ability. The gateway's remote deployment is not
immutably pinned. Differences can reflect sampling, provider routing or other
unobserved changes. Retain provider-reported reasoning-token usage even if a
catalog describes the model as non-reasoning.

Review metrics in this order:

1. Account for all four scheduled attempts, including candidate failures,
   infrastructure errors, missing attempts and calls without usage.
2. Report ingestion, trials and submission checkpoint completion, then successful
   fresh-container source reproduction. Structural success is not financial accuracy.
3. Report full-credit count, financial reward and its audit, trial and decision
   components for each episode, alongside the matching historical result.
4. For the targeted clarification, the primary objective diagnostic is correctness
   of the two fixed requested `forward_return` rows: `E02` on day 36 and `E05` on
   day 74. Evaluate both in each of the four paired episodes: eight observations
   per condition. Derive expected values directly from each episode's saved public
   prices and securities CSVs. Resolve each price row to an entity using ticker
   validity intervals inclusive at that row's day, compute
   `adjusted_close = close * shares_per_original`, then calculate
   `adjusted_close(d+1) / adjusted_close(d) - 1`. Compare saved policy answers with
   absolute tolerance `1e-8`. Report the eight individual results and correct
   counts by episode and condition. The historical baseline has 2/8 correct,
   distributed as 0, 0, 2, 0 across seeds 4–7. Missing or malformed answers are
   not correct; preserve their reason and the fixed eight-observation denominator.
   This descriptive diagnostic does not change the frozen financial scorer.
5. As secondary evidence, inspect the saved source and trace for multiplying or
   dividing by the split factor. Distinguish clear multiplication, clear division,
   inconsistent handling and indeterminate evidence. Preserve relevant source or
   trace locations. Do not execute candidate source for this error analysis.
   Correct aggregate output alone does not prove the operation used.

The full financial outcomes remain the overall performance measure. The fixed
split-sensitive return diagnostic measures the specific clarification target;
source reading supports interpretation and does not replace that objective check.

Every scheduled attempt is eligible for accounting. Keep unsuccessful attempts;
do not repair responses, choose the better of repeated runs, or rerun failures as
replacement observations. Report an unavailable score as unavailable, with its
reason, rather than zero or a fabricated difference. Separate the unchanged
scorer's financial metrics from descriptive manual error analysis. Do not
retroactively rescore or amend the baseline.

The 64-attempt calibration remains a later decision. This condition supplies and
launches only four attempts. A complete account and at least one reproducible
submission establish technical readiness; they do not establish acceptable
financial accuracy. Review remaining ambiguities and the paired error analysis
before deciding whether a larger separately bound calibration is useful.

## Execution and freezing

Limits match the staged baseline: 32 turns, 65,536 output tokens, 262,144 total
tokens, 8,192 tokens per response and temperature 0.2. Episode, agent, scoring,
SDK and stream retries are zero. The Docker image is `python:3.12-slim`; no
network entries are allowed or host environment variables forwarded. Scoring is
isolated. Use the same resolved image digest as the baseline. A mismatch requires
an explicitly documented new condition rather than an unqualified paired claim.

The checked-in config is dry-run-only with an invalid endpoint and model
placeholder. Keep the exact bound config and all live outputs outside the repo.
Credentials remain in the trusted host environment. Never include private
evaluator material in solver files or context.

Install pinned packages with Python 3.12 or later and run local validation:

```sh
uv pip install --python .venv-portfolio/bin/python ./environments/financial_research ./environments/financial_portfolio ./environments/financial_portfolio_staged ./environments/financial_portfolio_clarified
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_clarified*.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/clarified_v1/smoke.toml --dry-run
```

The manifest binds all new package/protocol files and clarification tests. Its
freezer checks the staged and prerequisite manifest chains back through the
original calibration payload. Freeze only after implementation, review and local
checks, before the first live call:

```sh
python3 configs/financial_portfolio/clarified_v1/freeze_manifest.py --write
python3 configs/financial_portfolio/clarified_v1/freeze_manifest.py --check
```

`--write` refuses to overwrite a manifest. After freezing, verify again before
execution. Any change after live execution requires a new versioned condition.
Use a fresh local output directory and `vf-run @ LOCAL_CONFIG --dry-run false` for
the four authorized attempts. The checked-in placeholder is never a live config.
