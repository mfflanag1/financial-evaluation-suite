# Selection calibration v1

This is an offline handoff for five instruct-model evaluation episodes on the
training split: one index-0 current packet for each of mixed, positive, negative,
unsupported and sparse. Each source is graded on its current packet and two
fresh packets by the existing environment. There is one planned attempt per
condition. Final and development episodes are not selected. There are no repair
runs, resume, retries, difficulty changes or automatic sample expansion.

No live inference has been performed. Five observations cannot establish model
quality, condition-level performance or generalization. The generator conditions
are five diagnostic settings of one mechanism, not five independent mechanisms.

The five TOMLs are safe dry runs. They pin the published selection task and a
restricted Bash/edit harness with zero SDK retries; the harness does not add
portfolio checkpoints. Network notices are disabled because the exact selection
prompt already states the network restriction. Docker networking remains denied.
Environment, agent, scoring and transport retries are all zero. The native
RunnerConfig supports no resume argument in these configs. Only the prepared
five configs are admissible; reruns create duplicate/unplanned records and block
accounting. The operator must not discard earlier attempts or rewrite records.

## Prepare without inference

Use the clean, noneditable Python 3.12 installation of the suite and pinned
Verifiers 0.4.1 dependencies. The preparation command compares installed package
resources and native source files with the frozen source hashes. Configure
`FINANCIAL_SELECTION_KEY` locally; never put it in a config, trace or repository.
Choose an immutable local Docker image ID and an instruct model/endpoint. This
command does not inspect Docker, use a model credential, contact the endpoint,
execute a submitted program, or load fresh/final packets:

```sh
python -B scripts/selection_calibration.py check
python -B scripts/selection_calibration.py prepare \
  --output /absolute/external/new-calibration-directory \
  --model INSTRUCT_MODEL_NAME --endpoint https://endpoint.invalid/v1 \
  --image sha256:REPLACE_WITH_64_HEX_DIGIT_LOCAL_IMAGE_ID
```

Retain the printed `binding_sha256` separately before any launch. Preparation
creates five local TOMLs and a binding containing opaque task IDs, public-input
hashes, the exact task/system prompts' hashes, configuration hashes, and package
manifest identity. It stores no evaluator key, private population, fresh packet,
policy or solution. Reusing or overwriting a preparation directory is rejected.

For a native model-free parser/write check, invoke each prepared config once:

```sh
vf-run @ /absolute/external/new-calibration-directory/mixed.toml
```

Repeat this dry-run command for positive, negative, unsupported and sparse.
All prepared files retain `dry_run = true`. Native dry runs write resolved
configuration files only, before environment loading or client construction.
They do not constitute attempted model evaluations and cannot produce a score.

Live inference is deferred and requires separate authorization. No utility here
launches it. Once authorized, an operator may run each prepared config exactly
once with only the native `--dry-run false` override, serially. The key must be
unchanged and `SELECTION_MODEL_API_KEY` must be configured locally. No other CLI
or environment override, retries, resume, model substitution or endpoint change
is permitted. A failed or interrupted attempt remains unresolved; it is not
replaced. Changing model/endpoint/image requires a new protocol decision before
any attempt, not a repaired report of this frozen schedule.

## Account for all planned attempts

```sh
python -B scripts/selection_calibration.py report \
  /absolute/external/new-calibration-directory \
  --binding-sha256 RETAINED_PRELAUNCH_SHA256
```

The checker joins each native episode and its sole trace to its exact task,
prepared/resolved config, resolved agent/model/provider settings, actual call-node
prompt ancestry and
immutable image. It also checks observed call model, endpoint dialect and
sampling bounds. It rejects missing/duplicate/unplanned/conflicting attempts,
native failures, unbound records and inconsistent success metrics. It always
shows the denominator of five. Any unresolved attempt blocks all performance
means; genuine scored zeros and explicit candidate failures remain included.

Native selection traces expose aggregate reward, exact success and reproduction
metrics. They do **not** expose the four financial components across the three
packets, so the report leaves components null. Candidate-failure zeros are shown
separately. Exact success must accompany shaped reward: the published outcome-
blind cash control earns substantial partial credit. Regenerate that control
with `python -B -m financial_selection.baseline`; its 100 public development
fixtures are a contextual control, not a paired comparison with these five
keyed train episodes. Its documented results are bound through VALIDATION.md.

These are locally trusted native records, not cryptographically authenticated
provider receipts. Native `dry_run` is excluded from resolved configuration serialization, so the
permitted execution toggle does not change its binding. All other settings stay
bound. The checker cannot detect an operator fabricating mutually
consistent files, deleting an earlier attempt, or relabeling a provider's model.
The retained preparation digest anchors the intended task/configuration, not
proof of execution. No claim of endpoint evaluation, provider delivery, training,
optimizer progress, profitability or production readiness follows from the
model-free tests. Accounting tests use explicitly synthetic records.
