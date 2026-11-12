# Staged portfolio research protocol

`financial-portfolio-staged` 0.4.0 adds enforced public checkpoints, explicit malformed
tool-call termination and zero request retries. It uses the frozen portfolio 0.3.0
data, research 0.2.0 financial scorer and fresh-container source reproduction.
Earlier packages, protocols and results remain separate.

Checkpoint acceptance establishes structural completeness, earns no reward and does
not establish financial accuracy. All checkpoints must pass before final scoring.

## Install and check

Use Python 3.12 or later. From the repository root:

```sh
uv venv .venv-portfolio
uv pip install --python .venv-portfolio/bin/python ./environments/financial_research ./environments/financial_portfolio ./environments/financial_portfolio_staged
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_staged*.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/staged_v1/smoke.toml --dry-run
```

The native adapter pins `verifiers==0.4.1`. Its bundled chat harness also pins its
SDK and transport dependencies. Both native taskset and harness IDs are
`financial-portfolio-staged`.

Create an evaluator key outside the repository, or reuse an existing evaluator key.
Load it only in the trusted evaluator shell:

```sh
.venv-portfolio/bin/financial-research init-key --out /private/tmp/portfolio-evaluator.key
export FINANCIAL_RESEARCH_KEY="$(cat /private/tmp/portfolio-evaluator.key)"
.venv-portfolio/bin/vf-validate financial-portfolio-staged --runtime.type docker --taskset.seed-start 4 --taskset.num-tasks 4 -n 4 -c 2 -o /private/tmp/portfolio-staged-validation
```

`vf-validate` uses reference and untouched submissions without calling a model.
Docker is required. Validation records stay outside this repository. See
[VALIDATION.md](VALIDATION.md) for checks and their limits.

## Solver contract

The solver receives `prices.csv`, `features.csv`, `securities.csv` and `contract.json`
in `/workspace`. It must use Python's standard library and persist a `research.py`
that derives its outputs from those inputs. The task prompt includes the exact
checkpoint schemas; `contract.json` defines all financial calculations and requests.

| Stage | Required saved files | Acceptance check |
| --- | --- | --- |
| `ingestion` | `research.py`, `ingestion.json` | Nonempty, syntactically valid Python; exact public-file inventory. |
| `trials` | Prior files plus `trials.json` | One complete metric row for every declared candidate. |
| `submission` | Prior files plus `policy.json` | Every requested audit, matching trial rows and a structurally valid decision. |

After writing and running the script, issue the stage's exact command through the
`bash` tool as the sole tool call in that assistant response:

```text
financial-checkpoint ingestion
financial-checkpoint trials
financial-checkpoint submission
```

These are virtual commands handled by the trusted coordinator, not installed shell
programs. Issue each separately after its writes finish and the preceding checkpoint
passes. Rejected checkpoints return a reason and the next required stage. Every
stage revalidates earlier artifacts. Passing `submission` ends the attempt; prose
completion or budget exhaustion before all stages pass earns zero.

`ingestion.json` contains CSV SHA-256 digests and data-row counts, the contract digest,
distinct entity and candidate IDs, and declared window boundaries. `trials.json` is
an array with exact fields `id`, `mean_net`, `sharpe`, `mean_turnover` and
`screening_score`. `policy.json` has exactly `audit`, `trials`, `decision` and
`selected`; each audit row contains `entity_id`, `day`, `field` and `value`.
Trial rows must match the current `trials.json`. Decisions are `deploy` with a public
candidate ID or `abstain` with `selected: null`.

Files must be regular UTF-8 files. JSON rejects duplicate keys, extra fields,
nonfinite numbers, Boolean numbers and nesting beyond 32 levels. Integer fields
require integers; audit values may be null. Lists require complete, unique coverage
and may appear in any order.

| File | Maximum bytes |
| --- | ---: |
| `research.py` | 1,048,576 |
| `ingestion.json` | 262,144 |
| `trials.json` | 1,048,576 |
| `policy.json` | 2,097,152 |

Only these four declared artifacts are collected, individually with bounded transfers;
there is no workspace sweep. Oversized transfers omit the payload, retain an explicit
violation record and fail the candidate. Other collection failures remain
infrastructure errors. Accepted submission hashes bind all four final artifacts.

## Runtime and trust boundary

The coordinator validates artifacts against the original public inputs without
executing candidate code. Checkpoint progress is held outside the solver filesystem
and writable tool state. The private financial verifier remains isolated; submitted
source runs only in a separate public-only reproduction container. Keep evaluator
keys, generator packages and private grading data out of solver context.

Malformed tool-argument JSON terminates the entire assistant tool batch before any
command runs. The trace retains the raw argument string; the failure receives zero
candidate reward without repair or retry. Provider HTTP and stream errors remain
infrastructure failures. The supplied protocol sets SDK, stream, agent, episode and
scoring retries to zero. The harness disallows external tools, environment forwarding,
search and compaction.

## Evaluation interpretation

The supplied smoke configuration uses training seeds 4–7, one attempt per mechanism,
within one shared 32-turn budget per attempt. It is an unpaired diagnostic: different
seeds and several protocol changes prevent attributing differences from earlier runs
to one intervention. Checkpoint completion, source reproduction, financial accuracy
and withheld-path quality are separate outcomes. Preserve missing and infrastructure
attempts in accounting; neither reference validation nor a checkpoint pass measures
model performance.

The checked-in smoke configuration has an invalid placeholder endpoint and defaults
to dry run. Bind a model, endpoint and credential variable in an evaluator-local copy;
record task, input and prompt hashes, dependency/image versions and four expected
attempts before running `vf-run @ LOCAL_CONFIG --dry-run false`. Use a fresh output
directory and `financial-portfolio report OUTPUT --expected 4` for failure accounting.
The package includes no live calibration result or training claim.
