# Contributing

Changes should make a financial distinction testable and its result reproducible.
Explain the task's information condition, governing conventions, answer schema
and failure modes before expanding the number of examples.

## Preserve existing results

The question/evaluator banks and existing scorer/protocol versions are frozen.
Do not edit their payloads or regenerate their manifests to conceal a change.
Add a separately versioned correction with a minimal defect reproduction,
corrected public contract, grader checks and version comparison. Changes to
stateful execution require new sessions.

Do not add original datasets, employer material, credentials, evaluator keys,
private traces or local machine paths. Use public or authored synthetic inputs
with explicit provenance and distribution status. New third-party text needs
item-level review before it enters an exported packet.

## Review a case

1. Solve from the candidate-visible inputs before inspecting its reference answer.
2. Record assumptions, units, availability times and rounding/tie conventions.
3. Check the entire declared output schema, including discrete quantities and
   mixed arrays. An integral reference value does not establish an integer rule.
4. Test plausible wrong interpretations and boundary conditions, including valid
   alternative solutions. Prefer financial counterexamples over implementation-mirroring tests.
5. For sequential tasks, verify actual action effects and information release.
6. Record what remains unverified. Assistant review is not practitioner review.

## Verify a change

Use `scripts/check.py` with the tier appropriate to the changed component.
Native integration changes require the pinned install and real Docker checks.
Record test counts and skips. A local scripted provider tests the protocol;
language-model performance and optimizer training require separate observed runs.

Independent numerical changes regenerate `quality/` artifacts; render the
browsable catalog with `python scripts/catalog.py`. Source policy changes require
review of exact source/question hashes, deterministic inventory regeneration and
export-boundary tests. Keep unreviewed sources link-only.

Before release, inspect the diff, frozen bindings, installed package contents,
source packet and generated documentation. Run reports belong outside the repo.
Do not claim an exposed static case is an unseen test, or silently change a
historical result when a successor scorer is introduced.
