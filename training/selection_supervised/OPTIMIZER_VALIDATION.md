# One-step optimizer evidence

This gate checks whether retained **trusted local observations** consistently
describe one native optimizer update, a complete checkpoint, a fresh load without
another update, and inference after the receiver applies the new policy. It does
not execute training, authenticate a remote GPU, prove collector honesty, or
establish learning quality. A successful LoRA run validates a LoRA update, not a
full-parameter update. Fixture tests validate the checker only.

The contract is deliberately narrow: one trainer rank, first optimizer step,
one selection environment, policy version 0 to 1, no resume or fake data, and
native GRPO with no length penalty, no length-weighted baseline, and token loss
aggregation. Unsupported configurations fail instead of being approximated.
The inference engine may use a separate GPU. The supervisor's resource and
process limits apply independently of this evidence gate.

Run the installed inspector against a private bundle and an independently
retained prelaunch plan:

```sh
python -m financial_selection_supervised.optimizer_evidence \
  /private/run/bundle.json --expected-plan /private/approved-plan.json
```

Exit 0 means `trusted_observations_consistent`; exit 1 means rejected; exit 3
means an internally consistent **fixture**, which cannot satisfy the live gate.
Every report sets `gpu_execution_independently_verified` and
`training_readiness_claim` to false. No result from this checker alone should be
described as independently attested training or as a performance improvement.

## Versioned input contract

All JSON uses schema `selection-optimizer-evidence-v1`. Duplicate JSON keys,
non-finite observations, Boolean values used as integers, missing artifacts,
digest mismatches, path traversal, and symlink artifacts are rejected. The
checker hashes files without loading executable checkpoint objects. Keep the
bundle, raw traces, worker journals, model files, and generated tasks private.

The independently retained plan contains:

| Field | Meaning |
| --- | --- |
| `schema`, `run_id` | Contract version and unique, prelaunch run identity. |
| `model_id`, `model_revision`, `model_path`, `initial_weights_sha256` | Exact model/revision, resolved local snapshot path, and immutable initial weight inventory binding. |
| `update_scope` | `lora` or `full`; must agree with resolved model configuration. |
| `config_sha256` | Hash of the full resolved native configuration JSON. |
| `sources_sha256` | Hash of the verified installation's source-binding manifest. |
| `collector_sha256` | Hash of the actual instrumentation source shipped to the host. |
| `task_sha256` | Canonical JSON hash of the single shared task's `Trace.task.data`. |
| `prime_rl_commit` | `c225aaa45d2160380b2c9c160d072b12af5c785f`. |
| `world_size`, `step` | Both exactly integer 1 for this contract. |

`bundle.json` contains `schema`, `kind` (`observed` or `fixture`), `binding`
(the entire plan), `artifacts` (relative path to SHA-256), `config`, `sources`,
`collector` (artifact paths), and `records` (the eight stage names below mapped
to distinct artifact paths). Every referenced artifact must be inventoried.
The bundle's plan must equal the separate expected plan; copying the plan out
of an untrusted bundle after a run defeats this check.

Every stage JSON contains `stage`, `run_id`, `kind`, `binding_sha256` (canonical
plan hash), and an actual host `monotonic_ns`. The checker joins all stages to
the same plan and enforces causal ordering. All processes must be on the same
host monotonic clock. A distributed extension needs a different clock contract.
`after.optimizer_step_monotonic_ns` separately records the instant the optimizer
returns. Receiver apply must follow that instant. It may precede the final
`after` snapshot: native broadcast occurs before completed-step Progress is
counted and checkpoint state is captured. Do not backdate the state snapshot.

| Stage | Required observations |
| --- | --- |
| `runtime` | CUDA backend, trainer world size, device UUID, CUDA/PyTorch versions, trainer process identity, exact loaded model and initial weights binding. |
| `cohort` | One cohort and task identity; at least two unique, resolved, admitted policy siblings with different financial rewards. Each carries its raw native trace and raw journal artifact binding, provider request identity, prompt hash, sampled token IDs/log probabilities/version 0, reward, scalar advantage and transported token advantages. |
| `before` | Cohort ID, ordered trace IDs, optimizer step, actual batch digest, sampled-token and micro-batch counts; complete pre-update AppState inventory, optimizer counters and native Progress. |
| `after` | The same actual batch/cohort join; finite loss, positive finite gradient norm, gradient element/finite-element counts, positive finite parameter delta norm, changed model and optimizer state, counters advanced exactly once, and completed batch Progress. |
| `checkpoint` | Step 1; full post-update state inventory; complete file inventory under `checkpoints/step_1/`, including trainer `.metadata`, DCP shard(s), and `orchestrator/progress.pt`; canonical orchestrator state hash. |
| `reload` | A different process, native `CheckpointManager.load_from_path`, the identical checkpoint inventory, zero learning updates, and exact model/optimizer/scheduler/progress/counter equality; separately counted native zero-lr optimizer initialization with byte-identical parameters; independently reloaded orchestrator state hash. |
| `receiver` | `WeightWatcher.on_update` hook after apply, receiver identity and applied version 1. For LoRA: published adapter tensor inventory joins the changed trainable state, and published/applied file inventories join identical retained artifact bytes. This is an artifact/apply/version join, not a comparison of every inference-engine tensor. |
| `inference` | A new real policy request against that receiver/model, with nonempty sampled token IDs, finite nonpositive log probabilities, and version 1 on every sampled token. |

State inventory format is a mapping with exactly `model`, `optimizers`,
`scheduler`, and `progress`. Each maps canonical state-entry names to SHA-256.
Record all entries, including buffers and optimizer state; do not hash a
hand-picked parameter subset. Canonical tensor hashing must bind name, dtype,
shape and exact contiguous bytes, with deterministic ordering for nested
containers. `progress` inventory hashes each JSON scalar in the separately
recorded `step`, `total_tokens`, and `total_samples` mapping. Use the identical
canonicalization before saving and after loading. The collector must bind its
implementation and preserve the resulting full inventory for independent
review; the inspector cannot discover a tensor that the collector concealed.
The native trainer increments `Progress.total_samples` by the number of
consumed micro-batches, which can differ from the number of siblings after
packing. Capture and reconcile `batch_micro_batches`; do not substitute cohort
size. Multi-turn siblings retain a list of actual `provider_request_ids`.

The raw `native_trace` is the native Trace JSON, not a reconstructed summary.
The `admission_journal` is a JSON list of unchanged worker-journal rows. Native
trace ID, task data, resolved status, financial reward and admission must
reconcile with the cohort and exactly one worker start, score and terminal
event. The checker does not replace the native adapter's graph/token admission
checks; retain the complete trace/journal for that review. The synthetic
`Bundle` builder in `tests/test_optimizer_evidence.py` is a structural fixture,
not a collector or a sample of actual GPU evidence.
Each sibling also names `provider_responses`, a list of separately captured raw
response artifacts. Their `trace_id`, actual `provider_request_id`, prompt hash,
model, sampled token IDs, log probabilities and policy-version arrays must
reconcile with the sibling. Concatenation follows actual call order. Capture
provider responses when returned; do not generate them from the later trace.

## Instrumentation points in the pinned native stack

The inspector binds five critical source hashes from the published training
installation. Validate the complete installed source manifest before launch;
these five checks alone do not attest every executed dependency.

1. **Cohort and actual batch.** In
   `prime_rl.orchestrator.algo.grpo.GRPOAlgorithm.score_group`, rewards are
   float32 and advantages are reward minus the group mean. There is **no
   standard-deviation normalization** in this branch. Capture after native
   scoring and join through `annotations.stamp_batch` to the shipped step and
   each effective trace. Capture the batch actually consumed by the trainer,
   including its masked token advantage streams. Different tasks, dropped
   siblings, scripted references, stale token versions and all-zero advantages
   cannot support this gate.
2. **Optimizer boundary.** Native `trainer/rl/train.py` scales/clips gradients,
   calls `optimizer.step()`, clears gradients, and steps the scheduler. Capture
   gradient finiteness and norm **before** clearing gradients; capture exact
   state before the update and again after scheduler/progress accounting.
   The monitor sanitizes non-finite metrics, so a missing `optim/grad_norm` is
   a failure, not evidence that the norm was finite. Enable `optim.max_norm`.
3. **Native checkpoint.** `trainer.ckpt.CheckpointManager.save` writes the
   distributed checkpoint through `AppState`, containing model, optimizers,
   scheduler and Progress. `skip_optimizer` must be false. Inventory every
   checkpoint file after the save completes; `load_weights` is insufficient.
   Preserve the separate orchestrator checkpoint's progress, train-source
   state and configuration state as well.
4. **Fresh load only.** Initialize the same native model, optimizer and
   scheduler in a fresh process, call
   `CheckpointManager.load_from_path(path, model, [optimizer], scheduler, progress)`,
   compare every state entry and counter, and exit **before any update**.
   Normal trainer resume increments the completed step and still enters the
   training body; setting `max_steps=1` does not turn resume into a safe
   load-only check. Native checkpoint files are executable/trusted artifacts;
   only the trusted loader, never the JSON inspector, deserializes them.
5. **Receiver and new inference.** `WeightWatcher.on_update` runs after
   `receiver.receive` and after `policy.version` advances. Observe that hook
   and then a fresh inference response with native token version spans.
   Sender `.finished` markers, a clean process exit, or watcher timeout warnings
   do not prove the inference engine applied the weights. The receiver's
   LoRA artifact inventory must come from the actual files passed to the
   receiver; read the published adapter tensor state and compare its canonical
   names/bytes to the trainer. Copying a trainer digest into a receiver record
   is not an observation. Native adapter loading does not expose the engine's
   full tensor state, so the LoRA result cannot claim full engine equality.

The collector must fail on missing hooks, unresolved siblings, unjoined
cohorts, truncated arrays, lost records or reload differences. Do not invent
success flags or substitute CPU arithmetic, a toy optimizer, patched rewards,
or reference-generated siblings to make a failed live run pass. A rejected
live run is useful diagnostic evidence and remains rejected.

The fresh load-only process loads the pinned initial model before native checkpoint
restoration. Torch 2.14 DCP may call `optimizer.step(closure=None)` once to
allocate previously empty Adam state. The observer allows only that native
initialization: every optimizer learning rate must be exactly zero, every
trainable gradient must be finite and exactly zero, and every model parameter
must remain byte-identical. It records `optimizer_initialization_calls` and the
full before/after parameter inventories separately from `optimizer_updates = 0`.
Any further call, closure, nonzero gradient/rate, or parameter change fails the
load-only gate. This is not a claim of zero Python `step()` invocations.
