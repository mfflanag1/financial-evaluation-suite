# Portfolio calibration protocol v1

This freezes the endpoint-independent evaluation design for `financial-portfolio`
0.3.0 with `financial-research` 0.2.0 and Verifiers 0.4.1. No live model results are
included. The model, endpoint and evaluator key remain unbound; the complete run is
frozen only after those are recorded in evaluator-local run metadata before calls.

| Stage | Split and seeds | Episodes | Attempts per episode | Expected attempts |
| --- | --- | --- | --- | --- |
| Smoke | Train, 0–3 | 4 | 1 | 4 |
| Development | Development, 128–159 | 32 | 2 | 64 |

Default family rotation gives one smoke episode and eight development episodes per
mechanism. Test seeds 160–191 are reserved and are not selected. Two attempts on an
episode are related measurements; the development sample has 32 episodes, not 64
independent financial problems. The four mechanisms also share implementation and
generation assumptions.

## Fixed execution settings

- One instruct-model agent, bash tools, no external search, no forwarded host
  environment, Docker solver and isolated scoring.
- Temperature 0.2; maximum 8,192 output tokens per model response, 32 agent turns,
  65,536 aggregate output tokens and 262,144 aggregate total tokens per attempt.
- Concurrency two, deterministic task selection, no automatic scoring retries.
  Native episode and agent retries default to zero in the pinned runtime.
- Task limits remain the package defaults, including 600 seconds for the agent,
  180 seconds for scoring and a 30-second fresh-source reproduction limit.

These are token ceilings, not a monetary budget. Actual usage depends on the bound
model's tokenizer, context limit, provider behavior and price. Token-budget checks
can stop at model-response boundaries; inspect native usage when reporting cost.

The two checked-in TOMLs have `dry_run=true`, an invalid placeholder endpoint and a
credential environment-variable name only. Never commit credential values, private
key files, materialized tasksets or rollout outputs. `outputs/` is ignored by Git.

## Offline preflight

Install the sibling packages as described in the [environment README](../../../environments/financial_portfolio/README.md).
From the repository root:

```sh
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_portfolio_calibration.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/calibration_v1/smoke.toml --dry-run
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/calibration_v1/development.toml --dry-run
```

Native dry-run validates and writes resolved configuration, then exits before task
generation, containers or model calls. It does not test endpoint compatibility or
calculate expected attempt counts. A dry-run directory can be reused; any directory
with a nonempty `traces.jsonl` requires a new `--run.dir`. Saved native
`configs/resolved/run.json` omits the `dry_run` flag; the
checked-in TOMLs are the safe preparation entry points.

`manifest.json` binds the exact package resources, protocol and TOMLs by SHA-256.
The regression check verifies all listed hashes and package file inventories. The
manifest excludes itself and tests. Subsequent case, scorer or protocol changes
require a new package/protocol version and new manifest rather than rewriting this
baseline. It does not bind a remote provider's implementation or all transitive
dependencies.

## Bind and run later

Live execution remains deferred. Before it is enabled:

1. Copy both TOMLs outside the repository. Bind the same instruct model, immutable
   provider revision when available, endpoint and credential variable name. Confirm
   bash tool support, context capacity and the declared sampling settings. An endpoint
   that requires different settings starts a separately labelled protocol.
2. Record the repository commit, manifest digest, resolved configurations, dependency
   versions, Docker image digest, provider/model identity and evaluator-key
   fingerprint. Keep the actual evaluator key secret and unchanged between stages.
   Record per-episode public-input hashes and opaque task IDs on the trusted side.
   Distinct keys produce different datasets; they are not interchangeable replicas.
3. Use distinct, fresh output/run directories for preflight and live attempts. Enable
   live execution only in the local run configuration. Preserve all native traces,
   model usage, artifacts and errors. Do not place the evaluator key in client
   headers, container environment, mounts or model-visible messages.
4. Run smoke first. Advance when all four attempts are accounted for, infrastructure
   works, and at least one source artifact reproduces successfully. If all responses
   fail the interface, inspect those failures before expanding. Do not require a
   financial-reward threshold or change questions to pass the smoke gate.
5. Run the fixed development configuration. Do not alter prompts, generation,
   sampling or tools between its two attempts or selectively replace poor results.

An explicit later authorization to run is separate from offline configuration
validation. Hosted publication and RL training are outside this protocol.

Once authorized and bound, invoke each local copy with `vf-run @ LOCAL_CONFIG
--dry-run false`, smoke first. Changing any fixed setting starts a separate run
condition and must be labelled accordingly.

## Analyse and decide

Run accounting on each actual native run directory, using `financial-portfolio
report RUN_DIRECTORY --expected 4` for smoke and `--expected 64` for development.
Missing, malformed, dropped and unresolved infrastructure attempts block an overall
performance estimate. Candidate failures remain in the denominator with zero
reward. Diagnostic retries must be reported separately from the frozen batch.

Also verify the episode schedule: exactly the four planned smoke task IDs once,
or exactly the 32 planned development task IDs twice. The generic accounting command
checks saved-attempt identities and counts; it does not enforce that task-ID schedule.

Report source-audit accuracy, trial accuracy, validation-decision accuracy, source
reproduction rate, withheld economic quality and total reward separately. Include
failure counts, tokens, elapsed time and cost where provider usage/pricing permits.
Separate deployment and abstention decisions, and show results per mechanism. First
average the two attempts within each development episode. Report episode-level
distributions and paired changes; do not use binary pass@k for continuous reward or
claim statistical independence across attempts and related mechanisms.

Review every failure and a sample of successes. Prioritize incorrect scoring,
ambiguous instructions, source leakage and reward shortcuts. Only then identify cases
that consistently produce accurate reasoning with little difficulty. Freeze findings
before making targeted changes under a new version, and compare against this baseline.
Reserved tests remain untouched during these iterations.
