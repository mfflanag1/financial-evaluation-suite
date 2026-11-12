# Staged portfolio protocol v1

This separately labelled condition uses `financial-portfolio-staged` 0.4.0 with
the unchanged portfolio 0.3.0 financial data and research 0.2.0 final scorer.
It does not replace or regrade `calibration_v1` or earlier workflow diagnostics.

The smoke schedule is four training seeds, 4–7, one attempt each: one per mechanism.
These differ from the earlier diagnostic seeds 0–3; development and reserved test
seeds remain unused. Different seeds and a bundled protocol change preclude a
paired causal claim against the earlier smoke.

## Execution contract

The native taskset and harness IDs are both `financial-portfolio-staged`. The
overall limits remain 32 turns, 65,536 output tokens, 262,144 total tokens,
8,192 tokens per response, temperature 0.2 and concurrency two. Agent, episode,
scoring and SDK retries are explicitly zero. The bundled SDK version and dependency
pins are defined in the new harness. HTTP or stream failures remain infrastructure
errors; failed requests with absent usage are counted and disclosed.

Malformed tool-argument JSON terminates the whole assistant tool batch before any
call is delivered to the harness. The exact argument string stays in the native
trace. No repair, command execution or retry occurs. This is a zero-reward candidate
protocol failure, distinct from a provider error.

Three checkpoints inspect persisted files against original public inputs:

1. `financial-checkpoint ingestion`: valid Python source and exact public inventory.
2. `financial-checkpoint trials`: all declared candidate metric rows, with strict types.
3. `financial-checkpoint submission`: every audit request and trial, legal decision
   structure, and consistency with the saved trial table.

Call checkpoints individually, after writes finish. They are virtual bash commands
handled by the native trusted tool gate, which supplies the result without executing
an OS command. The coordinator stores progress outside the solver filesystem and
the writable tool-server state. Passing the final checkpoint ends the attempt.
Incomplete stages earn zero. Checkpoint acceptance is structural, adds no reward and
provides no financial answer. Final artifact hashes prevent post-checkpoint changes;
fresh-container reproduction and financial scoring remain required.

## Local preparation and execution

Install the three packages together with Python 3.12 or later:

```sh
uv pip install --python .venv-portfolio/bin/python ./environments/financial_research ./environments/financial_portfolio ./environments/financial_portfolio_staged
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_staged*.py' -v
.venv-portfolio/bin/vf-run @ configs/financial_portfolio/staged_v1/smoke.toml --dry-run
```

The checked-in configuration is dry-run-only and has an invalid placeholder endpoint.
Copy it outside the repository, bind the instructed model and endpoint, and record
the manifest digest, dependency and image versions, evaluator-key fingerprint,
per-episode IDs/public hashes/system-prompt hashes and an expected-attempt count of
four before enabling live execution. Never include key values or evaluator data in
solver context. Use a fresh run directory and `vf-run @ LOCAL_CONFIG --dry-run false`.

No large calibration or training configuration is supplied. Review all four attempts
first. Expansion requires accounted-for attempts, functioning infrastructure and at
least one final source artifact that reproduces successfully. Report checkpoint
completion separately from source reproduction, financial accuracy and hidden-path
quality; preserve missing or unresolved infrastructure attempts in accounting.

`manifest.json` binds this new package and protocol plus the frozen predecessor
manifest. Changes after freezing require a new versioned condition. It does not
freeze the remote model deployment or every transitive dependency.
