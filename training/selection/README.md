# Selection training contract

This separate package connects the published selection-adjusted research task to
the exact native trainer revisions in [the shared dependency pins](../dependencies/README.md).
It subclasses the selection task without changing its public files, equations,
reproduction gate, three-packet reward, or strict-success metrics. It adds a
worker lifecycle journal and a fail-closed native training-admission gate.
The existing portfolio training package is unchanged.

This is an experimental training contract. CPU tests use synthetic token graphs,
a local tokenizer and a scripted token server, not a learned policy. Passing
these checks establishes protocol behavior only. No optimizer step, gradient,
GPU execution, checkpoint or learning result has been observed. The pinned
native trainer requires a supported CUDA/NVIDIA runtime for the actual optimizer
milestone; this package does not substitute a CPU or MPS optimizer.

Observed validation: all 54 tests passed with no skips in the clean installed
Python 3.12 environment, including the actual spawned-worker/Docker fixtures and
scripted native TrainClient transport. The same-task reward-1 and reward-0
siblings remained trainable; the interrupted tool call was dropped. The fresh
installation verified 592 wheel resources and all three native plugins. The
source archive was checked against its explicit package-only allowlist, and all
nine training wheel resources matched the installed package. These are local
protocol and packaging results; the optimizer and learning milestones remain
unverified.

## Separate installation

Use Python 3.12 and a new environment. The published evaluation package's runtime
extra pins a different Verifiers release and must not be installed here. The
bootstrap installs selection and research **core** wheels with the exact trainer
sources and unchanged CPU dependency lock. It performs normal dependency
resolution, pip compatibility checks, isolated installed-resource verification,
and native plugin/config loading; it does not call a model or download weights.
From the repository root, point these variables at clean checkouts of the
commits in `training/dependencies/sources.json`:

```sh
python3.12 training/selection/scripts/bootstrap.py \
  --out "$SELECTION_TRAINING_INSTALL" \
  --prime-rl-source "$PRIME_RL_SOURCE" \
  --verifiers-source "$VERIFIERS_SOURCE" \
  --renderers-source "$RENDERERS_SOURCE" \
  --pydantic-config-source "$PYDANTIC_CONFIG_SOURCE"
```

The output must be a new private directory outside the checkout. Add `--offline`
when its build and wheel dependencies are already cached. Set
`SELECTION_TRAINING_PYTHON` to that directory's `venv/bin/python`, then run:

```sh
"$SELECTION_TRAINING_PYTHON" -I -B -m unittest discover \
  -s training/selection/tests -p 'test_*.py' -v
```

The CPU suite includes native worker task reconstruction, binary/delta transport,
payload retention, exact provider-token graph matching, policy-version coverage,
scoring consistency, cancellation and budget controls. Its loopback-only provider
requires local socket permission. Synthetic fixture logprobs do not demonstrate
calibrated probabilities or policy inference.

## Serial native integration

The opt-in integration fixtures use an actual spawned native worker, Docker,
native TrainClient renderer and scripted loopback token service. They exercise
the same task twice with a correct program and an incorrect submitted policy,
and separately interrupt a tool call. They verify sampled IDs, logprobs, sampler
support and policy versions through the native wire/payload store, original
rewards, drop admission and absence of the worker key from persisted artifacts.
Use an existing local Python 3.12 image; no image pull is performed:

```sh
SELECTION_TRAINING_DOCKER_IMAGE="$(docker image inspect python:3.12-slim --format '{{.Id}}')" \
RUN_SELECTION_TRAINING_DOCKER_TESTS=1 \
"$SELECTION_TRAINING_PYTHON" -I -B -m unittest discover \
  -s training/selection/tests -p test_spawned_worker.py -v
```

Keep container concurrency one. Optional `SELECTION_TRAINING_FIXTURE_EVIDENCE_DIR`
retains journals, episodes, logs and payloads in an unused private directory
outside the checkout. These are operator evidence, not distributable benchmark
inputs. The fixtures generate and discard their own private worker key. Actual
workloads supply `FINANCIAL_SELECTION_KEY` only to the trusted worker; the key
must not enter task data, solver/replay environments, harness forwarding, source
archives, logs or public artifacts.

## Admission and limits

`Env.start()` verifies the installed adapter, financial and native source bytes,
requires the worker key, identical immutable solver/verifier images, networkless
mountless Docker, isolated scoring, no inference overrides and zero retries.
Each minted trace binds its task, exact outbound messages and model. The scoped
TrainClient wrapper returns the original response object unchanged. Every
sampled token requires a nonempty, contiguous policy-version span; untruncated
sampling may omit sampler support, and its exact presence/content is preserved.
Each response must match the native model-call node, ordered prompt ancestry,
unsampled prefix, sampled suffix, logprobs, support and version metadata.

`Env.finalize()` requires complete harness, runtime-health, tool, artifact and
isolated scoring observations, and reconciles the published evaluator's admission
predicate plus reward and exact-success/reproduction metrics. Resolved correct and genuinely incorrect
candidate outcomes remain trainable, including candidate failures with zero
reward. A controller-observed program analysis timeout can be a candidate zero;
host/tool-runtime/async scoring timeouts and unknown service outcomes are dropped.
Model-free reference validation is never trainable. Dropping does not rewrite
financial rewards or diagnostics. Active episode finalizers drain before worker
shutdown; malformed token graphs fail the worker delivery boundary rather than
entering unsafe native serialization. These are trusted-local consistency checks,
not cryptographic attestation of a remote worker.

The [configuration](configs/training_v1/contract.json) is a schema contract,
not a ready launch command. Model path and image digest are placeholders. It
specifies group size two, batch size two, one serial worker, no retries,
untruncated sampling, at most eight episode attempts and 900 seconds per worker.
Worker replacement resets those local budgets. A separate whole-run supervisor
must cap planned attempts, elapsed time, worker restarts and pre-trace startup
failures; the local trace journal alone does not account for unminted attempts.
A future optimizer launcher must supply that supervisor before use.

One surviving sibling or equal sibling rewards yields no GRPO reward difference.
Zero-advantage filtering can otherwise keep collecting indefinitely. An actual
training readiness claim additionally requires policy-generated same-task reward
diversity, finite nonzero advantages and gradients, an observed optimizer update,
a reloadable checkpoint, and synchronized weight versions. Report shaped reward
alongside strict three-packet success and the documented public baselines; shaped
partial credit alone is not evidence of complete research-policy correctness.

The project remains unlicensed. Bundled native chat code retains its
[third-party notice](src/financial_selection_training/THIRD_PARTY_NOTICES.txt).
