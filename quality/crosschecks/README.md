# Selected numerical crosschecks

This package supplies second implementations for **five existing cases and 28
outputs**. It adds reproducible assistant review evidence to the arithmetic audit;
these are not new benchmark cases or independent practitioner reviews.

| Case | Second method | Primary audit method |
|---|---|---|
| `q033` | Enumerate every bond and cash grid allocation with rational arithmetic | Enumerate bond pairs and solve their minimum cash completion |
| `q065` | Analytic Bachelier and shifted Black expectations | Numerical integration of terminal payoffs |
| `q098` | Enumerate all 262,144 complete 18-day binary paths | Absorbing-state probability dynamic program |
| `c201` | Enumerate signed trades; aggregate buys, early receipts and late receipts separately | Enumerate final holdings and apply settlement cash rules |
| `c214` | Reconstruct the residual tape and enumerate all 64 shared-block sign vectors; evaluate direct square-root scores | Exact signed squared-score ranks across all inspected policies and checkpoints |

The `c214` implementation uses exact pairwise cross products for decisions and
ties, with 80-digit square-root scores as a separate numerical agreement check.
It never rounds a close score into a decision tie. Both methods apply the same
stipulated randomization test; their agreement does not validate that null model
for actual trading data. `c201` reports all minimum-cost portfolios and chooses
the lexicographically first one only to make the displayed plan reproducible.

Run from the repository root with Python 3.11 or later:

```sh
python3 -B -m quality.crosschecks --check
python3 -B -m quality.crosschecks --derive-only --check
python3 -B -m unittest discover -s tests -p 'test_quality_crosschecks.py' -v
```

Without `--check`, the CLI prints the complete JSON evidence report to standard
output. It does not write files. The report includes alternate answers, compact
certificates, input and implementation hashes, field-level comparison results,
and scope limits. `--derive-only` does not import the primary audit modules.
Comparison failure returns a nonzero exit code.

`reviewed_inputs.json` pins the **complete bytes of each public packet**, including
its prompt and governing source contracts. A changed packet fails before any
derivation or primary comparison. Updating a pin requires reviewing the changed
contract and the applicability of both methods; this package does not offer
automatic pin regeneration. Constants transcribed from core documents are valid
only for the pinned contracts. The functions are not general financial APIs.

All alternate derivations finish before the runner imports or calls the primary
arithmetic modules. It never reads stored answers, imports production scorers, or
uses the inventory's reference-comparison code. Comparisons use each public
question's declared numeric tolerances; core tolerances come from the public
response contract. The tests also corrupt a primary result to confirm that
agreement is checked rather than assumed.

The finite searches certify the enumerated domains. Analytic/numerical agreement
is scoped calculation evidence, not a claim of market realism, empirical signal
efficacy, secrecy, or unrestricted input support. These public answers and
certificates are evaluator-side material and belong outside a solver's assigned
context. This package does not change the repository's licensing status.
