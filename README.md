# Financial reasoning evaluations

A local, vendor-neutral collection: **224 main-bank cases**, 1,924 structured outputs, and six separately counted adaptive-research variants. Topics include source-grounded analysis, quantitative methods, portfolio decisions, funding, execution and sequential research.

The original `questions/` and `private_evaluator/` packages remain frozen. Opt-in corrections live in [`scoring_v2/`](scoring_v2/README.md), with separate commands, scorer labels, session bindings and regression tests. Use that version for corrected scoring; keep original baseline results separately labelled.

A separate [generated research pilot](environments/financial_research/README.md) tests point-in-time data repair, signal validation, costs and portfolio decisions, with a native Prime/Verifiers integration. It has its own version and checks and does not change the main-bank counts.

An opt-in [portfolio experiment](environments/financial_portfolio/README.md) adds nonlinear execution costs, liquidity limits, covariance-based constraints and allocation choices. It uses a separate package and version; the original research pilot remains frozen.

A separate [staged workflow](environments/financial_portfolio_staged/README.md) adds enforced public checkpoints, explicit malformed-tool handling and zero request retries. Its intermediate checks measure structural completion; final financial scoring and fresh-container reproduction remain required.

Four [workflow prerequisites](environments/financial_prerequisites/README.md) check file inspection, saved calculations, script repair and complete structured output before portfolio evaluation. Each requires the saved script to work on original and alternate inputs in fresh containers. These synthetic diagnostics are separate from the financial case bank.

A separate [split-convention clarification](environments/financial_portfolio_clarified/README.md) adds an explicit adjusted-close formula to the staged instructions. It retains the same public data, financial scorer and workflow, with separately labelled results.

Its [development calibration](configs/financial_portfolio/clarified_calibration_v1/README.md) fixes 32 generated episodes and two attempts per episode, with unchanged instructions and scoring. Results are kept outside the repository; reserved tests remain excluded.

Separate [operational controls](environments/financial_portfolio_operations/README.md) add trusted-host request journals and numerical reference-score ceilings for all 32 development episodes. They preserve the frozen scorer and earlier results; local control checks are not model-performance measurements.

The candidate core contains 16 development and eight reserved cases. Reserved question packets are absent from public inputs and require explicit evaluator release. Practitioner validation is pending. This is research evaluation material, not evidence of production trading performance.

## Layout

- `questions/catalog.json`: main-bank inventory and per-track paths. Legacy tracks retain their own schemas.
- `questions/`: supplied-evidence solver inputs; sequential cases export opening packets only.
- `questions/variants/`: six related diagnostics, outside the independent main-bank count.
- `private_evaluator/`: answer keys, source provenance, scoring and trusted session coordination. Includes reserved material.
- `private_evaluator/amendments/`: optional source-sufficiency amendments, separate from frozen scored banks. Adopting one requires a new scorer version; label offline regrading separately.
- `tests/`: diagnostic reference, feasibility, evidence and integration checks.

Keep evaluator files, future states and reserved cases outside solver context. The complete collection is intended for a private repository. A folder name alone does not create an access boundary.

## Verify

Python 3.11 or later. The included runtime and checks use the standard library; no credentials or external services are required.

```sh
make check
```

This validates manifests and all 224 case definitions, runs 2,163 reference/negative controls, and checks the separate diagnostic family. Local reference controls are not solver trials or practitioner reviews.

## Grade static responses

Follow the schema supplied with each question. The main entry point dispatches legacy tracks and single-turn core cases:

```sh
python3 -B private_evaluator/grade.py grade CASE_ID response.json
python3 -B private_evaluator/grade_variant.py c225 response.json
```

## Sequential and retrieval cases

The trusted coordinator releases current public information and commits actual submitted actions:

```sh
python3 -B private_evaluator/sessions.py start c219 --condition retrieval --session ./sessions/private --out ./sessions/public
python3 -B private_evaluator/sessions.py submit --session ./sessions/private --out ./sessions/public --revision 0 --response response.json
python3 -B private_evaluator/sessions.py report --session ./sessions/private
```

Use the exact `session_revision` in `sessions/public/current.json`; zero applies only to the first submission. Keep private and public session locations separate, outside both package directories. Legacy multi-turn cases use `private_evaluator/legacy/sessions.py`; consult its `--help` for the track-specific contract.

Report strict correctness, financial content, evidence support, response validity and actual economic outcomes separately. Do not aggregate monetary regret across unlike objectives or equate expected regret with realized loss. Related variants and repeated attempts are not independent cases.

Source documents retain publication, availability and retrieval metadata needed for evidence verification. Financial dates and retrospective-availability qualifications are part of the specification. Manifests bind file contents; they do not assert authorship or historical performance.
