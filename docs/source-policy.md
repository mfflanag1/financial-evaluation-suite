# Source distribution policy

Use the versioned source exporter when preparing a new source-grounded solver packet. It allows only the exact reviewed files listed in `distribution/source_policy.json`. Public availability, a government-hosted URL and a repository license do not by themselves establish permission to redistribute a third-party document.

The inventory covers **72 source manifests, 72 retained evidence files and 72 public evidence mirrors across 36 cases**. It also records paths and hashes for associated public question/metadata files and evaluator cases, gold answers and controls. Those evaluator artifacts may contain source quotations and are never included in a solver export. This is a scoped source inventory, not a legal audit of every string in the repository or its history.

**58 original manifests contain unresolved or local-research-only language.** The new policy does not overwrite those frozen statements. It independently identifies 20 exact plain-text excerpts as eligible under the cited BLS, BEA, EIA, Federal Reserve Board and SEC publisher policies. It leaves **52 sources link-only**. An item remains excluded until its specific bytes and reuse basis are reviewed; this exclusion does not mean its reuse is necessarily unlawful.

The packet includes seven complete cases: `macro-001`, `macro-002`, `macro-003`, `energy-001`, `energy-002`, `portfolio-002` and `systematic-001`, using 17 eligible excerpts. Three other eligible excerpts belong to incomplete cases and are omitted with those cases. The other 29 source cases remain outside the packet. These counts describe distribution eligibility, not financial correctness, difficulty or independent practitioner validation. The project remains unlicensed; publisher reuse permissions do not grant a license to project code or authored cases.

## Publisher-policy basis

The policy records the observation date and scope of each review. Generated inventories contain no current timestamp and are byte-reproducible. The eligibility decision binds the source URL, original manifest, exact evidence hash and exact public question packet. Changes require a new review and updated policy; newly discovered files do not inherit approval.

- [BLS copyright information](https://www.bls.gov/opub/copyright-information.htm) identifies its publications as public domain except previously copyrighted photographs and illustrations. The five reviewed excerpts contain plain-text employment and CPI releases, with attribution preserved. No images or emblem are included.
- [BEA reproduction policy](https://www.bea.gov/help/faq/145) permits reuse unless otherwise stated and requests attribution. The two reviewed excerpts contain GDP release text and statistical tables; no logo or separately attributed asset is included.
- [EIA copyright and reuse policy](https://www.eia.gov/about/copyrights_reuse.php) permits reuse of EIA information products with acknowledgment, including the publication date, while distinguishing protected private contributions and photographs. The six reviewed excerpts contain storage/petroleum statistics, notes and release metadata. The natural-gas excerpts omit market-price and analyst-survey sections. No image, logo or separately attributed data is included.
- [Federal Reserve Board copyright policy](https://www.federalreserve.gov/disclaimer.htm) permits copying its website information unless otherwise indicated and requests citation. The two reviewed excerpts contain its March 22, 2023 FOMC statement and a jointly issued Treasury/Board/FDIC statement. The latter was checked against the [Board co-publication](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312b.htm); the FDIC URL returned HTTP403 during review. Neither excerpt includes a logo, image, model code or separately attributed third-party expression. This does not approve other FDIC excerpts or material from Federal Reserve Bank websites.
- [SEC Website Dissemination policy](https://www.sec.gov/about/privacy-information) permits copying and further distribution without SEC permission and encourages citation; its [Webmaster FAQ](https://www.sec.gov/files/about/webmaster-faq.htm) also addresses reuse. The five reviewed excerpts contain four SEC-generated filing indexes and SEC staff FAQ prose. File names and document-list rows identify linked issuer documents, but those documents and images are not included in the approved excerpts. Seals, logos, artwork and trademarks have separate restrictions.

Three other FDIC excerpts remain link-only. The retrieved [FDIC Archive copyright statement](https://archive.fdic.gov/Content%20and%20Copyright) applies to that archive; it is not treated as blanket permission for main-site content. Main-site retrieval was limited by HTTP403, and an exact archive edition plus complete passage match was not established. The joint statement's separate Board co-publication does not release the rest of `h2-011`.

These are assistant reviews of exact excerpts and publisher statements, not legal clearance. Eligibility does not extend to other parts of a linked document, embedded third-party assets or all material from that publisher. In particular, a company filing hosted by the SEC remains issuer material for this policy; it is not automatically treated as a government-created work. The [Copyright Office explains that facts and federal-government works are among the material copyright does not protect](https://copyright.gov/history/copyright-exhibit/lifecycle/), but that principle is not a blanket license for issuer-authored expression.

## Build and verify a source packet

Run from the repository root with Python 3.11 or later. The tools use only the standard library, require no credentials and do not fetch remote documents.

```sh
python3 -B -m distribution.inventory --check distribution/source_inventory.json

# Metadata only: source URLs, hashes, dispositions and case availability.
python3 -B -m distribution.export create --out ../source-links

# Seven reviewed solver cases and their 17 source excerpts; no answers.
python3 -B -m distribution.export create --out ../source-packet --mode source-packet
python3 -B -m distribution.export verify ../source-packet
```

The destination must be a **new directory outside the checkout**. Existing directories are never overwritten. A complete export writes its manifest last; a write failure may leave an incomplete directory, which must not be treated as a release. Verification reconstructs the expected bytes from the reviewed policy and rejects missing files, extra files or directories, changed bytes, symlinks and special files such as FIFOs. Publisher-policy metadata has a fixed schema; arbitrary additional fields are rejected. Verification does not merely trust a manifest supplied inside the packet.

`links` mode contains no question prompts or source text. A URL is a locator, not a license or a promise that the original version remains available. Hashes identify the retained excerpts, not necessarily the complete original PDF/HTML or a document downloaded today. The tool neither fetches nor regenerates excluded content. An evaluator who obtains a source separately must assess their own permissions, preserve the expected version and explicitly label any changed evidence condition.

`source-packet` mode contains 31 reviewed public question/metadata/evidence files plus four export metadata files. Its `case_availability.json` lists all 36 source cases and explains each omission. It is a solver-input subset, not the complete suite, a standalone grader or a package of the synthetic tracks. The packet retains frozen question bytes. Export the matching corrected question contract before a v3 run, provide it alongside the retained permitted evidence, and grade separately on the trusted evaluator host, for example:

```sh
python3 -B -m scoring_v3.grade export macro-001 > candidate-question.json
python3 -B -m scoring_v3.grade grade macro-001 response.json
```

The source packet never includes evaluator keys, gold answers, controls, later interaction states or reserved-case packets. The associated evaluator file paths and hashes in the inventory are provenance metadata only. Do not give a solver access to the full checkout when the intended condition is answer-free evaluation.

## Existing public copies and benchmark exposure

The complete repository has already been made public. Its `private_evaluator` directory contains answers, source quotations and material described by older documentation as reserved. Folder names and split labels are not access controls. These materials must be treated as publicly exposed; an answer-free packet does not turn them into an unseen benchmark.

The new policy affects **new exports only**. It has not removed existing evidence, quotations, frozen banks or Git history. The existing repository archive and a full clone therefore do not meet this source-packet allowlist merely because the exporter passes. Removing or replacing currently published source content requires an explicit migration plan; rewriting history would be a separate owner decision and cannot recall copies already obtained. Preserve baseline bytes in an appropriate archive if a future public release changes its contents, and label any revised cases/scoring distinctly.

Until item-level review is complete, share the allowlisted packet for the source track and report the omitted cases. Resolve excluded sources through documented permission, a supported reuse basis for the exact material, or a newly authored evidence/case version. Do not copy excluded quotations into documentation, error messages, examples or replacement metadata as a workaround.
