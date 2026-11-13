# Portfolio training adapter

This separate package targets the exact native trainer dependency revisions in
`dependencies/`. It preserves the financial generator, public research task,
structural checkpoints, public reproduction and isolated financial scorer.
It uses its own worker lifecycle and training admission protocol.

Use a separate Python 3.12 environment; the frozen evaluation environment is
incompatible with this trainer revision. The [dependency instructions](dependencies/README.md)
describe exact source commits, the CPU dependency lock and the clean bootstrap.
It installs wheels with ordinary dependency resolution and verifies installed
resources. It does not install GPU packages or download a model.

Run the CPU contracts with the new environment's interpreter:

```sh
"$TRAINING_PYTHON" -I -B -m unittest discover -s training/tests -p 'test_*.py' -v
```

The tests use native Env orchestration, worker task reconstruction, binary/delta
transport, payload retention and the installed trainer's admission predicate.
Synthetic token graphs test data plumbing; their logprobs are fixture values.
They do not demonstrate model inference, reward diversity, gradients or learning.

The serial integration fixtures also use an actual spawned native worker, Docker,
the native TrainClient renderer, a local synthetic tokenizer and a scripted HTTP
token server. They complete all public checkpoints, reproduce the submitted
program in isolation, retain a positive and an incorrect zero-reward sibling,
and drop a tool timeout before native training admission. Token IDs, logprobs,
sampler support and policy versions are checked through the native wire and
payload store. They require an existing `python:3.12-slim` image and pin its
concrete image ID for both solver and verifier:

```sh
RUN_TRAINING_DOCKER_TESTS=1 "$TRAINING_PYTHON" -I -B -m unittest discover \
  -s training/tests -p test_spawned_worker.py -v
```

Run these fixtures with container concurrency one. Optional
`TRAINING_FIXTURE_EVIDENCE_DIR` retains journals, binary episodes and payloads in
an unused private directory outside the checkout; these are evaluator artifacts,
not a public data export. A host or Docker startup timeout remains an unresolved
execution and fails the integration test. Inspect and remove any identified
fixture container left by an interrupted upstream startup before retrying.

`Env.start()` installs host journaling inside each worker process. Each trace
registers its task, expected outbound prompt and model before provider access.
Only the native renderer TrainClient is accepted. The wrapper returns the exact
native response, retaining token IDs, logprobs, sampler support and policy
version spans. `Env.stop()` restores the resolver and closes unfinished attempts.
Completed attempt details leave memory after their append-only terminal record.
Shutdown drains active episode finalizers before closing the journal. Admission
joins each provider response to the native model-call node, exact prompt ancestry,
sampled token suffix, logprobs, sampling support and policy versions. Host mounts,
mutable or mismatched solver/verifier images, inference overrides and retries
are rejected. These checks assume trusted installed code and host journals;
they are consistency checks, not cryptographic attestation of a remote worker.

`Env.finalize()` admits a trace only after prompt, request, tool, runtime and
financial-scoring evidence reconcile. Unknown executions receive a native
`trace.drop()` flag. Their original reward and diagnostics remain unchanged.
Completed incorrect or malformed submissions can retain their genuine zero
reward and participate in training. A failed admission gate fails the episode.

The [training-v1 configuration](configs/training_v1/contract.json) is a schema
contract, not a ready launch file. Its local model path and image digest are
placeholders. It specifies two rollouts per task, batch size two, one serial
worker, untruncated sampling, zero retries and isolated scoring. Missing or
dropped siblings leave at most one admitted reward in this two-rollout contract,
which gives no GRPO reward difference. Equal rewards also give no advantage.
Default zero-advantage pruning can keep collecting indefinitely; an eventual
training launcher needs a separate duration and attempt cap. Stop a smoke run
on any unknown outcome. Larger groups require a separately reviewed group policy.

No model training, optimizer step, GPU execution or financial performance result
is claimed. Actual training requires separately authorized local-policy rollouts,
different finite rewards for the same task, nonzero advantages and gradients,
an observed optimizer update, a reloadable checkpoint and weight synchronization.

The project remains unlicensed. Bundled third-party code retains its own notices.
