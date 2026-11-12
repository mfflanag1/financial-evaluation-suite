# Corrected scoring

`financial-reasoning-v2.0.0` is an opt-in scorer over the frozen collection. It preserves the original question bank, answer keys, manifests, evaluator commands and historical results. `baseline.json` pins the original manifests and identifies the baseline commit. New reports include a scorer version and a content binding; sessions reject mismatched versions or bindings.

## Run

From the repository root:

```sh
python3 -B -m scoring_v2.grade validate
python3 -B -m scoring_v2.grade export q076 > question.json
python3 -B -m scoring_v2.grade grade q076 response.json
python3 -B -m scoring_v2.grade grade h2-001 response.json
python3 -B -m scoring_v2.grade grade c227 response.json
```

Use the v2 export for corrected public contracts. Source-grounded exports still use the retained source documents under `questions/`. Static workflow grading is explicitly an **executed-answer** mode accepting `question_id` and `answers`; code submissions use the separate workflow controller.

Sequential core and research cases use a trusted coordinator:

```sh
python3 -B -m scoring_v2.sessions start c207 --condition supplied --session sessions/v2-private --out sessions/v2-public
python3 -B -m scoring_v2.sessions submit --session sessions/v2-private --out sessions/v2-public --revision 0 --response response.json
python3 -B -m scoring_v2.sessions report --session sessions/v2-private
```

Read the current revision from the public packet. Core cases also support `--condition retrieval`. Research cases use supplied evidence and release one turn at a time. Reserved core cases still require explicit `--allow-holdout`. Private state, evaluator files and reports must remain outside solver access; separate directories are not an access-control boundary.

## Workflow programs

Build the image locally once, then run the controller:

```sh
docker build -f scoring_v2/Dockerfile.workflow -t financial-evals-workflow:v2 scoring_v2
python3 -B -m scoring_v2.workflow start q101 --session sessions/workflow-private --out sessions/workflow-public
python3 -B -m scoring_v2.workflow submit --session sessions/workflow-private --out sessions/workflow-public --revision 0 --response first-code.json
python3 -B -m scoring_v2.workflow submit --session sessions/workflow-private --out sessions/workflow-public --revision 1 --response final-code.json
python3 -B -m scoring_v2.workflow report --session sessions/workflow-private
```

Each response is exactly `{"code":"def solve(data): ..."}`. The first submission runs on public inputs and returns the program's outputs or error. The final program runs on the base and both hidden variants, with results available only in the private report. Invalid submissions and candidate execution failures consume their attempt. Sessions retain the immutable image ID, and execution never falls back to the host or automatically pulls an image.

Containers use a non-root user, no network or host mounts, a read-only filesystem, dropped capabilities, and limits on memory, processes, CPU, output and elapsed time. The image contains only the worker and its Python/NumPy runtime, not the evaluator or answer bank. Build and operate it on a trusted evaluator host with Docker available. These are local isolation controls, not a certification for hosting hostile tenants.

Run the real-container regression suite explicitly:

```sh
RUN_WORKFLOW_DOCKER=1 python3 -B -m unittest discover -s tests -p test_v2_workflow.py -v
```

## Corrections

| Cases | Corrected behavior |
| --- | --- |
| q101–q106 | A separate isolated code controller implements the two-submission workflow; static answer scoring remains explicitly distinguished. |
| c207, c212 | Exact integral JSON numbers are normalized before execution. Fractional quantities and booleans fail discrete-action scoring and execution. |
| q076, q083, q088 | Count outputs require exact integer values; half-counts no longer pass. |
| q056 | The v2 public packet explicitly enumerates all permitted share counts from 0 through 100. |
| Core, diagnostic and research tracks | Oversized integers fail numeric validation without an overflow exception. |
| Core and diagnostic tracks | Response-contract validity requires every declared answer with the correct structure and every per-field citation list. Correctness and source sufficiency remain separate metrics. |
| h2-001 | Annual-summary evidence is an additional permitted route, requiring the annual heading, prior-year column, non-GAAP section, income row and value. The recast-policy evidence requirement remains. |
| q138–q141, q144, q146, q160 | Specific verified supporting sources are permitted; q139 also accepts the sufficient current-turn source alone for stipulated new revenue. |
| c217, c224, c227, c229 | The retained source-sufficiency amendments are applied only by v2. |

Citation changes retain required support and reject unknown, unavailable and duplicate source IDs. They do not accept arbitrary extra citations. The source-grounded scorer checks retained quotes and authored anchors; it is not a general entailment model.

## Baseline comparison

Keep original commands and reports as the frozen baseline. For static responses, run the same saved response through both graders and label the v2 output as corrected regrading. Never overwrite a baseline result or merge scores from the two versions.

Stateful cases require a new v2 session: corrected action execution can change subsequent observations and economic outcomes. Replaying an old response sequence is a counterfactual replay, not a fresh solver run. Workflow code evaluation and executed-answer evaluation also have different information and execution conditions and must be reported separately.

## Verification

```sh
make check
```

This runs the frozen checks plus v2 regression reproductions. The v2 tests cover reference paths and negative controls, the observed false positives and false negatives, overflow handling, session revisions, public-data boundaries and baseline preservation. These checks are implementation validation, not practitioner review or evidence of model difficulty.
