# Runtime v1 protocol

This successor binds exact outbound prompt identity, host execution diagnostics,
fresh-key bootstrap and conservative outcome accounting. Predecessor protocols
and stored results remain unchanged. Default optional guidance is empty.

The smoke schedule is training seeds 4–7, one attempt each, one active solver
container, zero retries. Bootstrap creates a fresh key and records the concrete
Docker image ID, source hashes, public input hashes and raw financial reference
ceilings. It requires no earlier private run. Reference controls are labelled
separately from native scripted-provider integration and model evaluation.

Use the package [instructions](../../../environments/financial_portfolio_runtime/README.md)
to bootstrap and run. Changes to frozen payloads require a new protocol version.
The manifest is written once after review, then checked read-only:

```sh
"$EVAL_PYTHON" -B configs/financial_portfolio/runtime_v1/freeze_manifest.py --check
```

An empty or malformed completed response can be a candidate failure. Interrupted
harness execution, unanswered tool execution, dead runtimes and unresolved
timeouts cannot be converted to zero financial reward. Exact runtime accounting
must accompany native result files; native footer averages are insufficient.
