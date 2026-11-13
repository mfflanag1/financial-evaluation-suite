# Bounded selection training launcher

This separate package supervises the published `financial-selection-training==0.1.0`
adapter on its exact Prime RL / Verifiers stack. It does not replace financial
scoring or change admission, and does not establish that a policy learned.

The first contract is intentionally narrow: one local node, two GPUs (one trainer,
one inference), rank-8 LoRA, one static worker, one training step, zero worker replacements and
zero episode/model/scoring retries. Existing local policy weights, tokenizer and
immutable Docker image are required. Hub downloads and remote monitors are disabled.
An actual CUDA run requires separately authorized resources. The frozen profile
uses Flash Attention 2, no compile or CPU offload, full activation checkpointing,
AdamW at 1e-6 with gradient clipping, and filesystem weight broadcast. Only LoRA
parameters are updated; successful evidence would not establish a full-weight update.

Install into the separate pinned training environment, without changing the eval
runtime. Build and install this package as a normal wheel, with its dependencies
already installed. All implementation is original wrapper code; native packages
are dependencies, not vendored copies. This project remains unlicensed.

```sh
selection-supervise prepare \
  --model /absolute/path/to/local-policy \
  --image sha256:<local-image-id> \
  --root /absolute/private/new-run \
  --out /absolute/private/config.json
selection-supervise check /absolute/private/config.json \
  --root /absolute/private/new-run
```

`check` binds configuration/source bytes and checks local assets. It does not run
training. To execute after resource authorization, set a fresh 32-byte hexadecimal
`FINANCIAL_SELECTION_KEY` in the trusted host environment, then use:

```sh
selection-supervise run /absolute/private/config.json \
  --root /absolute/private/new-run --execute \
  --max-attempts 8 --startup-seconds 300 --total-seconds 900 --finalize-seconds 30
```

The launcher creates a fresh directory with mode 0700. The SQLite ledger is 0600;
children inherit a private umask. Each attempt reserves budget transactionally
before native `Env.run` validates or serializes a task. A context-local reservation
binds the exact wire request identifier before the first send. Returns, exceptions
and cancellation are accounted without rewriting a reward; process death leaves
reserved attempts visible as unresolved. Equal-reward groups cannot bypass the
attempt or whole-run time budget.

A boot callback composes the native worker logging and `PayloadHook` initialization.
The environment marks ready only after the unchanged adapter starts. Startup also
requires an actual reserved attempt before its deadline. A healthy
broker alone never counts as worker readiness. The watchdog stops admission on
timeout, worker death, or exhausted completed attempts; then it terminates the
entire observed process tree, escalating to KILL after the finalization allowance.
Cleanup has a further bounded four-second process reap and up to three ten-second
Docker commands. Only containers labelled with this run's random identifier are
removed, including any surviving after forced worker termination.

The generation key travels to the launcher through an inherited pipe, not argv,
config, logs or SQLite. Only the orchestrator taskset and pool/worker receive it in
their environments. Trainer and inference environments have the key and any aliases
removed. No key is passed into solver or verifier containers. Run reports, native
logs, model traces, weights, checkpoints and ledgers stay private and must not be
committed. The local operator is trusted: these are not tamper-proof receipts.

`supervisor-report.json` reports execution accounting and cleanup separately. Even
a successful native exit leaves `optimizer_validated` false: the independent
optimizer evidence checker must establish finite nonzero advantages/gradients,
parameter change, a saved and reloaded checkpoint, and inference weight sync.
An independently prepared private `--optimizer-plan /absolute/private/plan.json`
enables the observer on the trusted orchestrator, worker and trainer. The key stays
absent from trainer and inference. The plan and evidence are copied into the run's
private directory. A separate fresh-process checkpoint-load check must follow the
native run; the supervisor never substitutes a fixture for that check.
No fixture or static import test can close those gates.

## Reproducing the local verification

Use the separate Python 3.12 environment produced by the unchanged selection
training bootstrap, with its pinned dependencies already installed. From the
repository root, build and test the installed wheel:

```sh
uv build --offline --out-dir /absolute/private/dist training/selection_supervised
uv pip install --offline --no-deps --python /absolute/private/venv/bin/python \
  /absolute/private/dist/financial_selection_supervised-0.1.0-py3-none-any.whl
RUN_SELECTION_SUPERVISED_DOCKER_TESTS=0 \
  /absolute/private/venv/bin/python -I -B -m unittest discover \
  -s training/selection_supervised/tests -p 'test_*.py' -v
```

The installed CPU/native suite ran 77 tests: 75 passed and two were explicitly
skipped. The skips were the separately enabled Docker fixture and a native Torch
scheduler check, because this CPU environment does not install Torch. Coverage
includes transactional caps, cancellation, failed requests before trace creation,
healthy-broker/failed-worker detection, process-tree cleanup despite ledger errors,
strict source/configuration checks, secret isolation, separate-process plan staging,
and observer/checker rejection controls. Synthetic tensor doubles test collector
logic; they are not evidence of gradients or GPU execution.

The serial Docker fixture was also run separately and passed. To reproduce it,
provide the already built immutable selection image and an empty private evidence
directory, with no other fixture using its resources:

```sh
RUN_SELECTION_SUPERVISED_DOCKER_TESTS=1 \
SELECTION_TRAINING_DOCKER_IMAGE=sha256:<local-image-id> \
SELECTION_SUPERVISED_FIXTURE_EVIDENCE_DIR=/absolute/private/docker-fixture \
  /absolute/private/venv/bin/python -I -B -m unittest discover \
  -s training/selection_supervised/tests -p 'test_supervised_docker.py' -v
```

That fixture uses the actual wrapped native broker, worker and orchestrator with
a scripted token provider. It retained one reward of 1 and one candidate-caused
reward of 0, accounted for both attempts and request IDs, rejected a third
reservation, and cleaned up its processes and labelled containers. It does not
run a policy, optimizer or checkpoint reload. Actual optimizer validation remains
a separate CUDA gate described in [OPTIMIZER_VALIDATION.md](OPTIMIZER_VALIDATION.md).
