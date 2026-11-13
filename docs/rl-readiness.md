# RL readiness

The generated research and portfolio packages expose native Verifiers 0.4.1 tasksets: task generation, public inputs, structured rewards and isolated reproduction/grading components. This supports environment integration work; it does not establish successful model training.

## Scope

The research pilot tests point-in-time data repair, a declared signal menu, trading/borrow costs, policy selection and saved programs. The portfolio successor adds nonlinear impact, liquidity, funding and covariance constraints. Staged protocols add checkpoints.

Four generating mechanisms and registered train/development/test seed ranges provide data variation. New seeds do not establish broad strategy discovery. A fixed policy menu bounds the search problem.

## Boundaries

The solver receives public files, task identity, instructions and authorized tools. Keys and private continuations remain on trusted workers. Replay the saved program in a fresh public-data container; grade finances separately. Do not expose evaluator packages, grader source, hidden data or keys to the candidate runtime.

Record exact messages sent to the provider, including framework additions. Guidance must remain idempotent across task copying and launch. Terminal diagnostics must distinguish intentional completion from disappearance during an unanswered tool call.

## Release gates

| Gate | Required evidence |
| --- | --- |
| Install | Clean Python 3.12 environment; local wheels install; dependency check passes. |
| Discovery | Pinned native loader discovers the intended taskset, task and harness. |
| Contract | Data boundaries, schemas, integer handling, artifact limits and prompt identity pass. |
| Native execution | Scripted-provider rollout completes checkpoints, artifact collection, fresh reproduction and isolated scoring in real containers. |
| Failure accounting | Malformed calls, unanswered tools and runtime failures have distinguishable outcomes. |
| Fresh setup | New evaluator key and local images work without historical private run files. |
| Training adapter | A specified trainer revision consumes trajectories/rewards and handles failures. |
| Optimizer smoke | An actual step records finite losses/gradients, usable rewards and checkpoint output. |
| Generalization | Preregistered held-out realizations and mechanism tests follow the selection policy. |

Configuration files alone do not establish adapter compatibility. Optimizer and generalization gates require their own observed runs. Claim optimizer validation only after observing and reviewing the actual run.

## Verified training adapter

The separate [portfolio training package](../training/README.md) passes 41 local tests with no skips, including actual spawned native workers, Docker and a scripted TrainClient. A same-task positive reward and genuine zero reward retain their native training records; an unresolved tool timeout receives `trace.drop()` and is excluded by the trainer's admission predicate. Prompt/completion token IDs, log probabilities, sampler support and policy versions survive native transport. Malformed graphs fail before worker serialization. Independent review found and verified fixes to trajectory identity, container isolation, dependency binding and lifecycle handling.

Its clean installation verifies 591 resources and eight native source bindings, using exact trainer and Verifiers development revisions. Follow its [separate bootstrap](../training/dependencies/README.md); it is incompatible with the frozen evaluation environment. The root seven-package bootstrap and test tiers do not install or test this training successor.

This establishes scripted compatibility at the adapter gate. No real policy rollout, advantage calculation, gradient, optimizer update, checkpoint reload or weight synchronization has been observed. The included configuration is a contract with placeholders, not a ready training launch. Actual training remains a separately authorized validation step.

## Reward and curriculum

Retain separate data-audit, calculation, decision, reproduction and economic metrics. A correct public decision need not maximize hidden-path reward. Do not retrospectively normalize scores to make reference solutions perfect.

Start with program prerequisites, then research and staged portfolio tasks. Use development data for protocol selection; reserve fresh tests before tuning. Compute any difficulty filters from training data and record exclusions. Do not optimize and evaluate prompts on the same small episode set.

A substantive expansion should add versioned mechanisms for less prescribed signal discovery, temporal leakage controls, multiple testing, capacity/funding stress and replay on changed realizations. Independently verify new mechanisms before expanding near-identical seed counts.
