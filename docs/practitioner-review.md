# Practitioner review packet

This is a prepared review request, not evidence that practitioner review has occurred. Independent calculations and container tests establish specific implementation properties; a practitioner must assess whether the financial distinctions are sound, clear and useful. The published answer keys mean reviewers may already know a case. Record prior exposure rather than describing the review as blinded without evidence.

## Forwarding note

> Please review the supplied financial reasoning cases independently from their candidate inputs before opening the reference solutions. Record your solution, material assumptions, any alternative defensible answer, and a concrete correction for each defect. Distinguish an intentionally simplified financial model from an ambiguous or internally inconsistent one. For sequential cases, review the actual information and action sequence. Your review will be recorded only after your findings are received; it does not imply endorsement of the full suite.

## Assemble the inputs

Use the [quickstart](quickstart.md) to export corrected scorer-v3 question contracts. Provide the scorer version and binding, case IDs, exact public-input hashes, available tools, and permitted evidence. Include a blank review record for each case. Keep reference answers, audit certificates and later sequential states outside the assigned input packet. This limits the review context; it does not reverse the repository's public exposure.

For source cases, use the [source exporter](source-policy.md) and supply the corrected v3 question contract alongside its retained evidence. The two source cases below are included in that reviewed subset. Do not forward the full source archive as if it met the packet allowlist. The project remains unlicensed; this review packet grants no new reuse rights.

Sequential cases must be delivered by the session controller, one state at a time. Workflow programs use their execution controller and controlled container. Give reviewers the declared schema and conventions, not an unlabelled concatenation of every turn or hidden execution input.

## Suggested first review set

This selection spans distinct failure modes. It is not a representative performance sample or a difficulty ranking. A reviewer may mark a case outside their expertise; assign another reviewer rather than treating that response as acceptance.

| Case | Review focus |
| --- | --- |
| [q001](../questions/legacy/quant100/questions/q001.json) | Restricted collateral and financing resources in a long-short book |
| [q015](../questions/legacy/quant100/questions/q015.json) | Robust allocation with ellipsoidal mean uncertainty |
| [q045](../questions/legacy/quant100/questions/q045.json) | Risk bounds when correlations are not fully identified |
| [q055](../questions/legacy/quant100/questions/q055.json) | Delta conventions when volatility changes with spot |
| [q076](../questions/legacy/quant100/questions/q076.json) | Sharpe uncertainty and track-record assumptions |
| [q095](../questions/legacy/quant100/questions/q095.json) | Noisy factor measurement and prediction |
| [portfolio-002](../questions/legacy/source_grounded/cases/portfolio-002/case.json) | Joint curve/funding stress; liquidity versus terminal economics |
| [systematic-001](../questions/legacy/source_grounded/cases/systematic-001/case.json) | Acceptance versus public availability; synthetic replay costs |
| [c201](../questions/challenges/c201.json) | Portfolio repair with a settlement constraint |
| [c207](../questions/challenges/c207.json) | Executed trades changing later funding and feasible actions |
| q151 — see [catalog](case-catalog.md) | Multi-turn energy delivery and nomination quantities |
| q164 — see [catalog](case-catalog.md) | Multi-turn execution and integer-order conventions |
| q103 — see [workflow instructions](../scoring_v3/README.md) | Saved program reproduction on controlled alternate inputs |

For consequential or disputed cases, obtain two independent reviews before showing reviewers each other's findings. A calculation author or another model run is not an independent practitioner.

## Review record

Complete this record separately for each assigned case. Store identifying reviewer details and completed responses outside the public repository unless publication is separately agreed.

| Field | Reviewer response |
| --- | --- |
| Case ID, input hashes, scorer version/binding | |
| Reviewer role/expertise; prior case or answer exposure | |
| Independently derived answer or saved program | |
| Units, timing, financing, settlement and rounding assumptions used | |
| Missing inputs or alternative defensible interpretations | |
| Financial calculation or economic-model defect, with reproduction | |
| Plausible wrong answer that should fail, and why | |
| Valid alternative answer the scorer should accept, if any | |
| Sequential action/information or program-replay concerns | |
| Value for the intended research/portfolio workflow | |
| Recommendation: usable / revise / reject / outside expertise | |
| Required correction and evidence needed to close it | |

Record factual error, internal inconsistency, ambiguity and limited realism separately. A precise toy problem may still be mathematically sound; that does not establish empirical trading relevance. An expert finding a case straightforward is not a measured model ceiling.

## Resolve findings

Reproduce each substantive disagreement against the exact packet. Resolve financial or information-contract defects before using that case for a capability claim. Add corrections as a new case/scorer version and retain the original result; do not silently repair frozen questions. Keep unresolved disagreements visible and report how many cases were actually reviewed, by what kind of reviewer, under which version. Publish no practitioner-validation claim based on this blank packet alone.
