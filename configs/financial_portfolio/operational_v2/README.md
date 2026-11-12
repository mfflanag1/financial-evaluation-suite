# Portfolio operational protocol v2

This successor preserves the clarified calibration's public tasks, financial
scorer, model settings and development schedule while adding trusted-host request
telemetry and numerical reference controls. Earlier manifests, attempts, reports
and score definitions remain frozen. `policy.json` is the machine-readable
failure and reference-control contract.

## Schedule and runtime

The schedule remains 32 development episodes, seeds 128–159, with two attempts
each, ordered selection and concurrency two. Reserved seeds 160–191 are excluded.
There are eight episodes per mechanism. Retain the earlier instruct model,
temperature 0.2, 8,192 response tokens, 32 turns, 65,536 output tokens and 262,144
total tokens per attempt. The host harness changes to
`financial-portfolio-operations` 0.1.0; the taskset remains
`financial-portfolio-clarified` 0.5.0.

Effective deadlines remain setup 120s, agent 600s, finalization 30s and scoring
180s. Public source reproduction remains 30s and 256 MiB. Use the original pinned
Docker image and dependency versions, no network allowlist and no forwarded host
variables. All episode, agent, scoring, SDK and stream retries remain zero.

## Predeclared failure handling

| Observation | Financial result | Operational interpretation |
| --- | --- | --- |
| Verified reproduction and grading | Recorded financial reward | Completed workflow; correctness is separate |
| Explicit candidate failure with authoritative zero | Zero | Candidate workflow/reproduction failure |
| Explicit native provider error | Unavailable | Provider failure |
| Native attempt deadline or unclassified timeout | Unavailable | Deadline reached; cause remains unresolved |
| Journal start without terminal event | No inferred reward | Interrupted or unfinished request; usage unknown |
| Missing, duplicate or malformed attempt | Unavailable | Accounting defect or unfilled schedule |

Use the unchanged `financial_research.accounting.report_path` for financial
accounting. Operational telemetry supplements it and never replaces its native
outcome. An explicit reproduction-controller timeout is a candidate failure;
an overall native deadline is not equivalent evidence. A cancelled request does
not prove provider fault. Successful earlier checkpoints cannot conceal a later
provider error or timeout.

Keep all 64 attempts in accounting. Any unresolved financial outcome withholds
the aggregate financial mean. Report workflow completion, correct public
calculations and exact full credit separately. Request-journal errors or unmatched
starts also prevent claiming complete telemetry or total spend. Sum only known
costs and state missing coverage. Do not use the native CLI's printed mean as the
accepted financial result.

Retries and selective replacement are forbidden in this condition. Any future
retry policy needs another frozen condition; its original and replacement
attempts must remain distinct. Diagnostic runs never overwrite this batch.

## Numerical reference scores

Before a new model run, persist a numerical reference record for every scheduled
episode. Bind its task identity, family, input/prompt hashes, key fingerprint,
source hashes and protocol digest. Execute the public-input reference in a fresh
isolated container, require public audit/trial/selection scores of one, and
compare its official score against the trusted gold policy.

Store the exact overall reward and each component, including withheld economic
quality. The prescribed public selection can produce a reference reward below
one. This is an attainable score ceiling under the unchanged scorer, not a model
failure or a reason to rescale earlier results. Keep reference artifacts and
hidden grading information on the trusted host. No reserved episode is generated.

## Binding and commands

`development.toml` has an invalid endpoint, placeholder model and `dry_run=true`.
Keep credentials out of the file. Bind an evaluator-local copy outside the repo
to the model, endpoint and fresh result directory. Preserve the baseline key
fingerprint, image identity and dependency versions. Record exact source hashes,
the frozen protocol digest and whether the source tree is committed or dirty.
Local uncommitted preparation is not evidence of a published release.

```sh
"$EVAL_PYTHON" -B configs/financial_portfolio/operational_v2/freeze_manifest.py --write
"$EVAL_PYTHON" -B configs/financial_portfolio/operational_v2/freeze_manifest.py --check
financial-portfolio-operations references \
  --repo "$REPO" --out "$REFERENCE_DIR" \
  --protocol "$REPO/configs/financial_portfolio/operational_v2/manifest.json" \
  --baseline-preflight "$BASELINE_PREFLIGHT" --key-file "$EVALUATOR_KEY_FILE"
```

The manifest refuses replacement and binds every frozen predecessor plus this
package, protocol and tests. Results and journals stay in fresh local directories.
This local preparation performs no live model calls, training, hosted upload or
publication. A future instrumented run must use the package's capture wrapper;
calling `vf-run` directly with the operations harness fails closed.
