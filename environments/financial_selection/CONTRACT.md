# Experimental contract v1

All records are original synthetic data. Time is an integer tick, not a calendar
date. Money is in cents, gross markouts in basis points, and rates apply to the
single stated holding period. Rational JSON values are reduced `[numerator,
positive_denominator]` pairs of integers. Booleans are never quantities.

## Public inputs

Exactly five UTF-8 files form the public packet:

- `universe.csv`: `event_id,decision_tick,label_maturity_tick,archived,pi_num,pi_den`.
  IDs are unique, archived is `0` or `1`, and a blank probability pair means
  explicitly unknown. Otherwise `0 <= pi_num/pi_den <= 1`. A zero-probability
  event cannot be archived, and a probability-one event must be archived.
  Reduced probability components are at most 1000. Every event's label maturity
  is after its decision.
- `feature_releases.csv`:
  `event_id,release_id,published_tick,revision,signal,regime`. Signal is an
  integer; regime is a declared token. A cell combines the regime and whether
  signal is negative (`negative`) or nonnegative (`nonnegative`). Select the
  latest release no later than the event decision, then greatest revision.
- `observed_labels.csv`:
  `event_id,release_id,published_tick,revision,markout_bp`. Markout is an integer
  number of basis points. Only archived events may have labels. Select the
  latest release no later than `research_cutoff`, then greatest revision.
- `deployment.json`: named baskets, rational cell weights summing to one,
  integer principal per lot, integer capacities from zero through three,
  integer linear/quadratic entry costs and linear exit costs, initial cash,
  credit limit, rational lending and borrowing rates.
- `contract.json`: version, research cutoff, declared regimes/cells, support
  policy, sample gate, screening coefficient and audit event IDs.

The population contains **every** universe row with label maturity at or before
the research cutoff. Its count includes unarchived rows. A matured row without
an eligible feature is invalid input, not a row to drop. An archived matured row
without an eligible label is also invalid. Same event/publication/revision keys
with different values are invalid, even if ineligible. Exact duplicate records
collapse within release tables; universe IDs remain unique. Release IDs identify
one immutable row within their table. Feature and label tables may
be shuffled. Late revisions and unmatured events cannot change population means.

## Identification and estimates

Archive inclusion is independent Bernoulli sampling conditional on the fixed
historical population, with its disclosed row probabilities. Deployment changes
only the disclosed cell mix. Within-cell outcome distributions, including the
distribution of relevant covariates, remain unchanged by stipulation. Neither
arbitrary latent regime changes nor unknown action counterfactuals are identified.

Check support over **every** matured census member, not just archived rows. A cell
with no population, any zero probability or any unknown probability is
`unsupported`. Its mean, variance and ESS are null. All baskets with positive
weight on such a cell are ineligible. Ineligible baskets must receive zero lots.
This is a conservative rule requiring a disclosed sampling design. An unknown
probability does not make a complete, fully observed census mean unidentified;
the policy still excludes it because this task requires the disclosed design.

For a supported cell with population size N and observed sample S:

```
mean_bp = sum(y_i/pi_i for i in S) / N
variance_bp2 = sum((1-pi_i)*y_i**2/pi_i**2 for i in S) / N**2
ess = sum(1/pi_i for i in S)**2 / sum(1/pi_i**2 for i in S)
```

An empty supported archive has mean, variance and ESS zero. Cells with observed
count below `minimum_observations` or ESS below `minimum_ess` are
`insufficient_sample`; retain their numerical estimates but exclude affected
baskets. A census with pi=1 and fewer observations than the declared minimum is
still gated. Defaults are four observations and ESS four; authored mathematical
controls may explicitly set both thresholds to zero.

The variance estimates sampling uncertainty, not future return variance. Shared
cell estimates create covariance between basket forecasts. No self-normalization
by an estimated population size is permitted. These are design-based point
estimates, not exact missing-outcome recovery or valid post-selection intervals.

## Allocation and cash

Cell and basket output order is lexicographic by declared ID. For q_j integer
lots, principal K_j cents and cell weights w_jg, let
`a_g = sum(q_j*K_j*w_jg/10000)`. Estimated gross profit is `sum(a_g*mean_g)` cents
and sampling variance is `sum(a_g**2*variance_g)` cents squared. Per-lot basket
forecast covariance uses the analogous products of both basket exposures.

Entry cost is the sum of each basket's `entry_linear_cents*q` plus
`entry_quadratic_cents*q*q`. Exit cost is `exit_linear_cents*q`. Opening residual
cash is `c = initial_cash - sum(q*K) - entry_cost`. Require
`c >= -credit_limit`, inclusive. Residual cash earns `c*lending_rate` when
nonnegative, otherwise `c*borrowing_rate`. Principal returns on redemption;
counting it as a loss is incorrect.

```
estimated_terminal_cents = sum(q*K) + estimated_gross_cents
                           + c + interest_cents - exit_cost_cents
cash_benchmark_cents = initial_cash * (1 + lending_rate)
estimated_excess_cents = estimated_terminal_cents - cash_benchmark_cents
penalty_microcents = ceil(1e6*lambda*sqrt(sampling_variance_cents2))
objective_microcents = floor(1e6*estimated_excess_cents) - penalty_microcents
```

Compare all feasible allocations by maximum integer objective, then **minimum
invested principal**, then lexicographically smallest lot vector. The all-zero
vector is feasible and has objective zero. Lambda defaults to two and is only
a prescribed screening penalty. The output reports exact squared sampling error
and its rounded penalty rather than an inexact square-root decimal.

## Declarative result and scoring

`policy.json` has exactly `schema`, `input_sha256`, `audit`, `cells`, `baskets`,
`basket_covariance_cents2`, `allocation`, and `ledger`. Arrays follow canonical
cell/basket order; IDs are explicit in their row records. Supported but gated
statistics remain numerical; unsupported means/variances/ESS and affected basket
estimates/covariances are null. Every output rational is reduced with positive
denominator. Money and counts that are integral fields remain JSON integers.

The exact verifier awards population/vintage audit credit 0.20, statistical
calculation credit 0.35, selected-allocation cash-ledger credit 0.15, and canonical
optimal allocation credit 0.30. Invalid schema, fractional quantities, wrong input
identity, duplicate keys, nonfinite values or infeasible allocations earn zero.
Numerically wrong but well-formed statistics can retain unrelated field credit.
The verifier never executes candidate source. CPU scoring alone establishes no
source reproduction or fresh-input execution. The separate replay adapter runs
the submitted source in fresh public-only containers: one reproduces the current
policy and two solve distinct new packets. Native reward averages those three
calculation scores only after successful current-policy reproduction. See the
validation record for the tested scope; this is not unrestricted generalization.

Complete-population expected economics are evaluator-only diagnostics. They do
not determine public calculation credit. Generated private outcomes, keys,
policies and runtime traces must not be checked into the repository.

### Exact nested output schema

Each row has exactly the following keys:

- `audit`: `event_id` (requested ID), `population_member` (boolean),
  `cell_id`, `feature_release_id`, `label_release_id` (identifier or null).
  Rows follow sorted `audit_event_ids`. Unmatured events have three null IDs;
  an unarchived matured event has a null label ID.
- `cells`: `cell_id`, `population_count`, `observed_count` (integer counts),
  `status` (`unsupported`, `insufficient_sample`, or `eligible`), `mean_bp`,
  `variance_bp2`, `effective_sample_size` (each null or rational pair).
- `baskets`: `basket_id`, `eligible` (boolean), `mean_bp` (null or rational).
- `basket_covariance_cents2`: a square matrix in basket order, each element
  null or rational. An entry is null if either basket has unsupported mean.
  A supported but sample-gated basket still has numerical covariance.
- `allocation`: one exact JSON integer per basket in sorted order.
- `ledger`: `invested_principal_cents`, `entry_cost_cents`, `exit_cost_cents`,
  `pretrade_cash_cents`, `penalty_microcents`, `objective_microcents` (integers);
  `interest_cents`, `estimated_gross_cents`, `estimated_terminal_cents`,
  `cash_benchmark_cents`, `estimated_excess_cents`, `sampling_variance_cents2`
  (reduced rational pairs).

Schema and identity errors yield zero. Wrong but syntactically valid values,
including incorrect support statuses or null assignments, lose the corresponding
field credit. The verifier calculates feasibility and cash from trusted public
data; candidate statistics cannot make an invalid trade feasible. Audit credit
averages four non-ID fields per requested row (empty audit earns this component).
Statistical credit averages six non-ID fields per cell, two non-ID fields per
basket, and every entry of the covariance matrix, including both triangles.
Ledger credit averages the twelve named ledger fields for the submitted feasible
allocation. Allocation credit requires the exact canonical optimum.

All JSON floating-point literals are invalid, including integral `1.0` and
overflowing exponents. Input files and output text are each limited to 2 MiB;
output integers have at most 4096 decimal digits. Reduced sampling probability
components are bounded by 1000; other input integers and rational components
are bounded by 10^12. No interpreter-wide integer limit is disabled.

The rational bound is structural, not only a tested example. Let
`D = lcm(1,...,1000) < 10^433`, let `P` be the product of the at most eight
nonzero cell population counts (`P <= 512^8 < 10^22`), and let `W` be the
product of all at most 48 basket-weight denominators (`W <= 10^576`).
Cell-variance denominators divide `D^2 * P^2`; portfolio-variance denominators
divide `10^8 * D^2 * P^2 * W^2 < 10^2070`. Each cell variance is at most
10^30 bp squared and total absolute exposure is at most 1.8e9 cents/bp,
so variance numerators need fewer than 2,120 digits. Rates and the bounded
screening coefficient add at most 24 denominator digits. A conservative
2,200-digit bound therefore covers the computed variance and cash components.
For ESS, write each inverse-probability weight as `A_i/D`: the sum of `A_i`
is below `10^439` and the sum of `A_i^2` below `10^875`. The reduced ESS
numerator is therefore below `10^878` and its denominator below `10^875`.
Together these bounds cover all computed correct-policy rational components.
There are at most 72 rational pairs in a policy, keeping a compact canonical
serialization comfortably below 2 MiB. Arbitrary candidate integers or extra
whitespace need not satisfy that stronger size bound; the parser enforces its
separate submission limits. This does not promise a runtime bound for every
adversarially chosen valid input; native generation uses much smaller ratios.
