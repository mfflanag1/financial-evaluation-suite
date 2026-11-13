# Verification evidence

The following are completed local checks, not model-performance or optimizer-training results. Commands and tests remain in the repository so reviewers can repeat them. Raw provider journals, evaluator keys, responses and run directories are intentionally excluded.

| Check | Observed result | Scope |
| --- | --- | --- |
| Clean installed packages | Seven packages; dependency check passed; nine native plugins discovered | Python 3.12, Verifiers 0.4.1; isolated noneditable imports and installed resource hashes |
| Core release gate | 356 passed, 125 optional native/container skips | Initial hardening snapshot; no failed tests |
| Expanded core release gate | 453 passed, 125 optional native/container skips | Includes subsequent numerical audits, alternate methods and corrected source packets; no failures |
| Native release gate | 457 passed, 24 Docker skips | Same snapshot; no disallowed skips |
| Full serial Docker gate | 481 passed, zero skips | Same snapshot, real containers and local scripted providers |
| Numerical expansion | 123 quality tests passed | Subsequent independent recomputation for all 100 quantitative cases; 110 cases and 461 outputs overall |
| Further numerical coverage | 169 quality tests passed; nine more cases and 78 outputs recomputed | Five decision cases and four multi-document source cases; 119 cases and 539 outputs overall |
| Complete authored-case calculations | 270 quality tests passed; all 230 entries and 1,972 outputs recomputed | Includes every authored research/core turn and six related variants; excludes arbitrary simulator trajectories and practitioner certification |
| Complete-audit release gates | Core: 554 passed, 125 optional skips; native: 655 passed, 24 Docker skips | 679 discovered tests per run; frozen manifests and installed-package checks passed |
| Alternate numerical methods | Five cases and 28 outputs agree; 16 regression tests passed | Separately implemented exhaustive searches, analytic pricing and exact score comparisons; a subset of the numerical audit |
| Source packet | 31 boundary tests passed; 35-file packet verified | Seven complete cases; 20 eligible exact sources across the inventory, 17 included in complete-case packets |
| Corrected source packet | 13 additional tests passed; 44 distribution tests overall | Same seven cases and 17 excerpts, matching v3 contracts, fresh-process version binding and a separate verified manifest |
| Runtime evidence replay | Five saved reports reconstructed identically | Completed reference, nonreproducing program, malformed tool, empty completion, interrupted tool |
| Fresh runtime setup | Four reference controls and installed CLI dry-run passed | New evaluator key and frozen protocol; zero provider requests during the dry-run |
| Separate training adapter | 41 tests passed, zero skips; clean installation checked 591 resources | Exact later native dependencies; actual spawned workers, Docker and scripted TrainClient transport/admission |

These rows overlap and must not be summed into a test count. The full Docker run preceded the numerical expansion; the expansion was checked separately and did not change runtime or scorer code. Additional independent review of that expansion used alternative numerical methods without importing the answer key or production scorer.

The runtime protocol is `portfolio-runtime-v1`, with manifest SHA256 `56fcc2101f5dad44ecca0007f8e477820f3cf8907055b8bcd4a4b63d7a684b83`. The corrected static scorer is `financial-reasoning-v3.0.0`, binding `c7db1aa46acc7669b0b42dbda47a556269daec18121a4d108fe278d4fe46ec2b`. Frozen predecessor banks, scorers and protocol payloads were preserved.

Reference economic rewards are not artificially normalized to one. The five-fixture runtime reference earned `0.9996936313902082`; the four fresh-key controls earned `1`, `0.984448057463567`, `1` and `1`. These are deterministic reference-program outcomes under their financial contracts, not measured model scores. An interrupted tool remained an unknown outcome with no aggregate financial score.

## Repeat the checks

Follow the [quickstart](quickstart.md) for a clean environment and images, then run:

```sh
python3 -B scripts/check.py --tier core
../financial-evals-venv/bin/python -B scripts/check.py --tier native
../financial-evals-venv/bin/python -B scripts/check.py --tier docker
```

Each command reports its own current counts and explicit skips. Use the [numerical audit](../quality/README.md), [source policy](source-policy.md) and [runtime protocol](../configs/financial_portfolio/runtime_v1/README.md) for narrower reproductions and fresh-key setup. A filtered test run does not replace the full release gate.

## Remaining evidence

No independent practitioner validation, fresh live-model comparison, empirical trading-performance study or actual RL optimizer update is established by these checks. Public answers remain exposed. Numerical review covers specified models and retained excerpts; it does not certify arbitrary inputs or every sequential branch. The separate training successor has passed scripted dependency, worker, token-trajectory and failure-admission gates. Real policy trajectories, gradients, optimizer updates and checkpoint reload remain unverified.
