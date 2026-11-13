# Corrected scoring v3

`financial-reasoning-v3.0.0` adds exact semantic integer checks and duplicate source-citation rejection to the preserved v2 scorer. Financial answer keys, frozen bank bytes and all v2 files remain unchanged. Public packets, results and sessions carry a distinct version and content binding. Use the matching v3 public export when evaluating with v3.

## Commands

Run from the repository root using Python 3.11 or later:

```sh
python3 -B -m scoring_v3.grade validate
python3 -B -m scoring_v3.grade export q076 > question.json
python3 -B -m scoring_v3.grade grade q076 response.json
python3 -B -m scoring_v3.grade grade credit-001 response.json
```

Source-grounded exports contain the question contract; the underlying source files remain subject to the [source distribution policy](../docs/source-policy.md). The full public repository includes evaluator answers. An answer-free export is an isolation mechanism for an evaluation run, not proof that a model has never seen the benchmark.

Sequential core and research cases require the trusted coordinator:

```sh
python3 -B -m scoring_v3.sessions start c207 --condition supplied --session ../session-private --out ../session-public
python3 -B -m scoring_v3.sessions submit --session ../session-private --out ../session-public --revision 0 --response response.json
python3 -B -m scoring_v3.sessions report --session ../session-private
```

Use the current `session_revision` from `session-public/current.json`. Private state must remain inaccessible to the solver. Later research turns are released by the coordinator. A v2 session cannot be resumed, submitted or reported through v3; start a new session and label any replay separately from a fresh evaluation.

## Semantic integer rules

The scorer pins [the explicit field map](../quality/discrete_fields.json) and enforces only its **201 `approved_integer` stage-field entries**. Candidate entries and explicit exclusions do not acquire new restrictions. Classification comes from task declarations, not from whether a stored answer happens to be integral.

Public packets expose `scoring_corrections.integer_fields` and per-output `integer_selectors`. A selector `[]` targets a scalar; `['*']` targets each array element; numeric positions select a component of a mixed array. For example, `c224.action[1]` requires an integer quantity while its action label remains a string, and `c220.diagnostics[2][*]` targets binary eligibility entries without imposing integer constraints on continuous diagnostics.

Selected leaves require finite integral JSON numbers and exact reference comparison. Booleans and fractional values fail; `2.0` and `2` are equivalent and are never rounded. Continuous siblings keep their original tolerances. Existing bounds, finite menus, multiples and economic feasibility rules still apply. Integer checks do not introduce an arbitrary magnitude cap; inappropriate large values fail their numerical or financial contract without an overflow exception.

The same checks apply to static quant/source/workflow scoring, research turns and core stage scoring. Accepted integral action forms are normalized before simulated execution. A fractional action cannot receive integer-field credit or be rounded into an executable order.

## Source citations

Within each source-grounded answer field, a `(source_id, whitespace-normalized quote)` pair may appear once. Normalization uses the existing source scorer convention. Distinct necessary quotes from the same document remain permitted, and a quote may support different answer fields. Invalid duplicate evidence cannot convey proof to dependent fields through `support_refs`.

Unknown or unavailable source IDs, invalid quotes, missing anchors and invalid support references continue to fail. This remains deterministic quote and authored-anchor checking, not a general semantic entailment model. Financial value accuracy and evidence accuracy remain separate.

## Legacy positive controls

**27 frozen positive controls across four source cases contain duplicate citations:** `equity-002`, `h2-001`, `h2-002` and `h2-005`. Their original responses intentionally fail v3's new citation contract. This is a response-format change, not a change to the financial answers.

[The control-change manifest](reference_control_changes.json) identifies every affected control, its frozen-file hash, duplicated fields and the hash of an explicitly deduplicated reference copy. Tests verify that those repaired reference copies pass. The originals are preserved, and the candidate grading path never removes duplicates automatically. Well-formed gold values with unique valid source citations pass across all 36 source cases.

## Isolated workflow programs

V3 reuses the unchanged v2 Docker executor and image, while its controller, session binding and final scoring are v3:

```sh
docker build -f scoring_v2/Dockerfile.workflow -t financial-evals-workflow:v2 scoring_v2
python3 -B -m scoring_v3.workflow start q103 --session ../workflow-private --out ../workflow-public
python3 -B -m scoring_v3.workflow submit --session ../workflow-private --out ../workflow-public --revision 0 --response first-code.json
python3 -B -m scoring_v3.workflow submit --session ../workflow-private --out ../workflow-public --revision 1 --response final-code.json
python3 -B -m scoring_v3.workflow report --session ../workflow-private
```

Responses contain exactly `{"code":"def solve(data): ..."}`. First-submission feedback exposes only execution on public inputs. Final scoring runs the base and two changed-input instances. Infrastructure failures preserve the attempt; candidate failures consume it. Execution never falls back to the host or automatically pulls an image. The trusted controller keeps hidden inputs and results outside the solver's context.

Static `grade` on workflow cases is explicitly executed-answer scoring; it does not run code and must not be reported as program evaluation.

## Verification and comparison

```sh
python3 -B -m unittest discover -s tests -p 'test_v3_*.py' -v
RUN_WORKFLOW_DOCKER=1 python3 -B -m unittest discover -s tests -p 'test_v3_workflow.py' -v
```

The tests exercise all approved selectors through their actual track graders, retained reference paths and negative controls, mixed-array semantics, duplicate-proof propagation, cross-version rejection, sequential CLI isolation and unchanged v2 behavior in a fresh process. The opt-in Docker test executes a reference program and a fractional-count mutant through the real controller.

`dependencies.json` pins the predecessor files and the field map. Updating either requires explicit review of a new version/binding. Keep v2/v3 results separately labelled. Static regrading compares the same submitted response under different contracts; a changed sequential execution path requires a fresh session or an explicitly labelled counterfactual replay. Software controls do not establish model difficulty, practitioner validation or trading performance.
