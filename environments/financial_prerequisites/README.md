# Financial prerequisites

Four fixed synthetic tasks test file inspection, saved Python execution, recovery
from a broken starter, and complete typed JSON output. They are a diagnostic gate
before portfolio research episodes, not additional financial benchmark questions.

Each task uses only public CSV/JSON inputs and the Python standard library. The
solver must save `research.py` and produce `policy.json`. Full credit requires the
submitted output to match and the same source to reproduce the expected output in
two fresh isolated containers: one with the original inputs, one with alternate
inputs following the same schema. Integer fields reject booleans and floats;
numeric quantities use absolute tolerance `1e-9`. Keys, list order, nulls, and
coverage are exact. Duplicated JSON keys and nonfinite numbers are invalid.

The package exports the native `financial-prerequisites` taskset for Verifiers
`0.4.1`; use the separately pinned `financial-portfolio-staged` harness. Expected
outputs, alternate fixtures, and reference source remain on the trusted worker.
Candidate source runs only in a fresh public-only container with denied network,
no mounted host data or credentials, and bounded unprivileged execution. The
isolated verifier parses artifacts as data and never executes candidate source.

All four cases must receive binary reward 1 with verified reproduction before a
model proceeds to the portfolio smoke check. Missing/invalid artifacts, malformed
tool arguments, wrong outputs, and unsuccessful replay are candidate failures.
Runtime/provider failures remain infrastructure errors and do not count as zeros.
Model-free reference validation is reported separately from model performance.
The repair case requires a corrected solution for a supplied broken starter;
its score does not prove that the solver first executed the broken version.

Install the three pinned local prerequisite packages first, then this package:

```sh
uv pip install -e environments/financial_research -e environments/financial_portfolio \
  -e environments/financial_portfolio_staged -e environments/financial_prerequisites
vf-validate financial-prerequisites --runtime.type docker \
  --runtime.image python:3.12-slim --max-concurrent 2
```
