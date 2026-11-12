# Clarified portfolio research condition

`financial-portfolio-clarified` 0.5.0 changes one public instruction in the frozen
`financial-portfolio-staged` 0.4.0 environment. Its system prompt appends exactly
two newlines followed by:

> shares_per_original is the cumulative number of current shares corresponding to one original share; close is price per current share. Define adjusted_close = close * shares_per_original.

Task IDs, user prompts, public files, financial calculations, grading, checkpoints,
artifact limits, source reproduction and the existing chat harness are unchanged.
Earlier environments and their results remain separate. The condition tests whether
making this price convention explicit improves performance; it does not establish
the cause of every difference between stochastic model attempts.

## Install and validate

Use Python 3.12 or later. From the repository root:

```sh
uv pip install --python .venv-portfolio/bin/python ./environments/financial_research ./environments/financial_portfolio ./environments/financial_portfolio_staged ./environments/financial_portfolio_clarified
.venv-portfolio/bin/python -B -m unittest discover -s tests -p 'test_clarified*.py' -v
.venv-portfolio/bin/vf-validate financial-portfolio-clarified --runtime.type docker --taskset.seed-start 4 --taskset.num-tasks 4 -n 4 -c 2 -o /private/tmp/portfolio-clarified-validation
```

The evaluator must load its existing `FINANCIAL_RESEARCH_KEY` in the trusted host
environment. Model-free validation uses reference and untouched submissions, calls
no provider and does not measure model performance. Keep outputs outside this
repository. The native taskset ID is `financial-portfolio-clarified`; the harness
ID remains `financial-portfolio-staged`, including its pinned zero-retry behavior.

Both setup and private verifier staging bind the exact clarified system prompt.
All decorated workflow hooks and the financial scorer are inherited unchanged.
The private staging path retains the original episode identity checks and never
executes candidate source; source reproduction remains in a fresh public-only
container. No private episode, evaluator key or grader is sent to the solver.

Trace metadata records `portfolio_condition.protocol_id = portfolio-clarified-v1`
and the clarification digest. `portfolio_workflow.protocol_id` remains
`portfolio-staged-v1`, because checkpoint behavior is the frozen staged protocol.
Use the separate condition protocol and identical model, data and limits when
comparing results. See the [clarified protocol](../../configs/financial_portfolio/clarified_v1/README.md).
