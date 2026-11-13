# Selection environment practitioner review

This packet extends the [static-case review](practitioner-review.md) to the
[selection environment](../environments/financial_selection/README.md). It is a
review procedure and blank record, not a completed practitioner assessment.
The five generated conditions share one synthetic financial mechanism. Reviewing
one example of each does not establish empirical trading relevance or model
difficulty.

## Forwarding note

> Please derive a policy or saved program from each assigned input packet before
> consulting the published reference implementation. Record every assumption
> needed to reach your answer, alternative defensible interpretations, and
> concrete financial or scoring defects. Assess whether the prescribed sampling,
> financing and decision model is internally consistent and whether it tests a
> useful research skill. Record prior exposure to the public answers. An
> assignment outside your expertise can be reassigned; it is not an acceptance.

## Prepare and deliver an assignment

Use five current packets from the `train` split, covering mixed, positive,
negative, unsupported and sparse conditions. Keep their condition labels in the
facilitator's private assignment record. Give each packet a neutral case ID.
This set is a mechanism review, not a held-out performance test. Do not consume
the `final` split for reviewer preparation or tune the protocol on it.

Each reviewer receives:

- The exact task prompt and normative `CONTRACT.md`.
- The five public input files: `contract.json`, `deployment.json`,
  `universe.csv`, `feature_releases.csv` and `observed_labels.csv`.
- Package version, source commit, input and document SHA256 hashes, permitted
  tools, and a blank review record.

Exclude reference answers/programs, evaluator code, generation keys, complete
populations, audit conclusions, model responses and the two fresh replay packets.
Retain facilitator records outside the public repository. Public reference code
already exists, so withholding it from an assignment does not make a review
blinded. Do not claim independence without recording the reviewer's role and
prior exposure.

Reviewers may use a local spreadsheet or program to derive the policy. Any
submitted code must be inspected and run through the controlled execution path;
do not execute an unsolicited program on the trusted evaluator host. Ask for
both a saved program and its current `policy.json` when reviewing reproducibility.

## Financial questions to resolve

| Area | Review question |
| --- | --- |
| Information timing | Does the census include the right matured observations, and do feature/label revisions obey their distinct cutoffs? Are duplicate releases treated consistently? |
| Selective observation | Are inclusion probabilities, full-cell denominators and zero/unknown support treated correctly? Is the distinction between an estimator and recovery of missing returns clear? |
| Uncertainty | Are the design-based variance, effective sample size and eligibility rules consistent with the stated sampling law? Is sampling uncertainty distinguished from market-return risk? |
| Population change | Are the disclosed mixture weights sufficient under the stated within-cell invariance assumption? Is that assumption explicit rather than presented as an empirical finding? |
| Dependence | Does shared-cell estimation induce the correct covariance between baskets? |
| Funding | Are principal, entry/exit costs, lending, borrowing and the credit limit applied at the correct stages without double counting? |
| Allocation | Are integer capacities, eligibility, feasibility, the objective and tie-breaking sufficient to define a unique accepted policy? |
| Reproduction | Does the same submitted program compute from its inputs, and are current-policy reproduction and fresh-packet correctness distinct requirements? |
| Usefulness | Which real research skill is tested, and which simplifications limit transfer to actual portfolio work? |

The exact task prompt already includes the reward weights and field-credit rules.
After the independent derivation is recorded, request a separate rubric assessment
and disclose the [public cash baseline](../environments/financial_selection/VALIDATION.md).
Ask whether partial credit rewards useful intermediate work or an unhelpful
shortcut. The baseline's substantial shaped reward and zero exact successes must
be considered together. Do not interpret a high partial-credit score as expert
agreement or successful learning.

## Review record

Copy this record for each assignment. Completed records and reviewer identities
remain private unless their publication is separately agreed.

| Field | Reviewer response |
| --- | --- |
| Neutral case ID; package version; source commit; manifest hash | |
| Reviewer expertise; prior exposure; tools used | |
| Independent derivation, policy or saved program | |
| Information cutoff, sampling, transport and funding assumptions | |
| Alternative defensible interpretation or answer | |
| Factual or mathematical defect, with minimal reproduction | |
| Ambiguity or missing convention | |
| Valid alternative the verifier should accept | |
| Plausible wrong result the verifier should reject | |
| Code reproduction or information-boundary concern | |
| Utility and limitations of the mechanism | |
| Rubric assessment after initial derivation | |
| Recommendation: usable / revise / reject / outside expertise | |
| Required correction and evidence needed to close it | |

Obtain two independent reviews for material disagreements before sharing either
reviewer's conclusions with the other. Separate internal inconsistency,
ambiguity, limited realism and rubric preference. Preserve the original packet
and result; corrections require a new version. Report actual returned reviews
and unresolved findings, rather than counting assignments as completed reviews.

This packet grants no new reuse rights. The project remains unlicensed.
