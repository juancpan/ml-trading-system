# Revision Policy v0

**Status:** ACTIVE.
**Effective:** 2026-05-16.
**Owner:** solo operator.
**Scope:** US equities / ETFs strategy (IBKR), 17 tickers.
**Repository:** This document is version-controlled. Deviating from it
requires a commit explaining why. The git log is your audit trail.

---

## Purpose

This document is the binding contract between the operator and the
strategy. It defines:

1. What counts as underperformance.
2. What automated triggers fire at each tier.
3. What revision actions are allowed at each tier.
4. The trials-budget accounting that gates every revision.
5. The retirement criterion.

If at any point the operator believes the strategy needs revision **and
the protocol does not authorise the revision**, the correct response is
*not* to revise. The correct response is either to wait for the
protocol to authorise it, or to amend this document (via a commit with
written rationale) before acting.

---

## Operating principles (binding)

1. **Pre-commit thresholds.** No tier threshold is set after observing
   live performance. Thresholds are derived from
   ``algos/wfov/baselines/portfolio_*_baseline.json``.
2. **Trials accounting.** Every revision proposal increments the
   cumulative trial count `N`. The Deflated Sharpe Ratio (DSR)
   significance threshold is recomputed at each proposal against the
   inflated `N`.
3. **Action invasiveness matches attribution.** No re-architecting
   because execution drag is high. The attribution decomposition
   (Phase 1.3) dictates which layer is revised.
4. **Pre-registration.** Candidate revisions are written to
   ``docs/revision_hypotheses.md`` (committed, dated) **before** being
   tested. Post-hoc data mining is not authorised.
5. **Two-tier discipline (amended 2026-07-07).** The protocol's gates
   are split into two tiers:

   **Tier A — advisory (compute and display, do not block):**
   MinTRL, DSR, WFOV trial cooldowns, pre-registration aging rules.
   ``revision_check.py`` computes and prints DSR/MinTRL but does not
   hard-block. An operator may override any Tier A gate by appending
   one written sentence of rationale to the relevant
   ``docs/revision_hypotheses.md`` entry. The discipline is a strong
   default and a personal tool, not a mechanical enforcement. This
   replaces the prior blanket "no revisions before MinTRL elapses"
   rule.

   **Tier B — hard, non-negotiable:**
   - Portfolio kill-switch (5%/8% MTD drawdown — ``kill_switch.py``)
   - Per-position circuit breaker (realized loss ≥ 2% of NAV from one
     position within 10 trading days — ``position_circuit_breaker.py``)
   These are risk-of-ruin protections, not statistical-skill tests.
   They stay mechanical and automatic regardless of operator
   philosophy. Tier B becomes *more* important, not less, as capital
   grows.

---

## MinTRL (Minimum Track Record Length)

**Last computed:** 2026-05-24.
**Source baseline:** `algos/wfov/baselines/portfolio_1d0adc7ddf4b_baseline.json`
**Coverage:** 12/17 tickers (~86.9% by weight). 5 buy_and_hold passive
positions (DEF.VI, IS.MI, GHI.VI, JKL.MC, GLX = 13.1% of weight)
remain in `known_gaps` with `reason="no_summary_for_ticker"`. They are
acceptable gaps for MinTRL purposes because passive positions
contribute nearly-neutrally to the Sharpe distribution.

### Inputs and computed MinTRL across Sharpe percentiles

The aggregated mean Sharpe across all WFOV iterations of the 12
covered tickers is **−0.105** (yes, negative), reflecting that
random-window Monte-Carlo backtests at the individual-ticker level
do not exhibit a positive expectation across the full distribution.
This is consistent with diversified-portfolio theory: individual
tickers run individual models that are mostly noise + small edge;
the portfolio-level Sharpe is what HRP weighting is designed to
elevate.

Because mean Sharpe ≤ 0 makes MinTRL undefined (cannot reject
benchmark at any sample size), we anchor on the **75th percentile**
of the Sharpe distribution as a more representative measure of what
the strategy *can* produce when it works. This is conservative
relative to the 95th percentile and honest about the distribution's
fat-tail asymmetry.

| Percentile | Observed Sharpe | MinTRL (obs) | MinTRL (months) | MinTRL/4 (cooldown) |
|---|---|---|---|---|
| Mean | −0.105 | ∞ (undefined) | n/a | n/a |
| Median | +0.244 | 11,503 | 547.8 (45.7 yr) | not actionable |
| **p75 (anchor)** | **+1.111** | **564** | **26.8 (2.24 yr)** | **~6.7 months** |
| p95 | +2.372 | 128 | 6.1 (0.51 yr) | ~1.5 months |

Inputs used: skewness = −0.185, kurtosis = 4.81, confidence = 95%,
annual_periods = 252. Formula: Bailey & López de Prado (2012), via
`algos.wfov.statistical_tests.minimum_track_record_length`.

### Policy bindings

- **MinTRL_months = 26.8 (≈ 27)**.
- **Cooldown (MinTRL/4) = ~6.7 months (≈ 200 days)** — the minimum
  age before a pre-registered hypothesis may be promoted to a formal
  `revision_proposal.yaml` per `docs/revision_hypotheses.md` aging rule.
- **MinTRL elapses on or after:** Day-0 + 27 months. With Day-0 at
  2026-05-24, MinTRL elapses on **2028-08-24**. Until that date,
  Sharpe-based judgments on the deployed portfolio are not
  statistically meaningful at the 75th-percentile-conservative reading.

### Caveats explicitly recorded

- This MinTRL is anchored on the **75th-percentile** Sharpe across
  WFOV iterations, not the mean. The choice is defensible because the
  mean is negative (undefined MinTRL) and the median (0.244) gives an
  impractical 46-year MinTRL. A more honest framing: if the strategy
  performs like its 75th-percentile WFOV iterations, we need ~27
  months to confirm; if it performs like its mean, no track record
  will ever confirm it. The discipline holds either way: do not deploy
  on faith.
- This MinTRL applies only to the **currently deployed configuration**
  (17 tickers, current model assignments). Any model retraining,
  weight rebase, or universe change invalidates this MinTRL and
  requires recomputation.
- The MinTRL value is recomputed and re-committed to this document
  each time a baseline file is regenerated (e.g., after a model
  retrain or universe revision).

### Operational consequence

Until MinTRL elapses (2028-08-24), Sharpe-based judgments on the
deployed portfolio are not statistically meaningful at the
conservative reading used here. The trigger logic in
`execution/revision_triggers.py` intentionally does NOT include
Sharpe-based criteria for this reason; it uses drawdown, hit-rate,
and execution-drag proxies instead, which are reliable on shorter
horizons.

---

## Tier thresholds (automated)

Implemented by ``execution/revision_triggers.py``. Outputs are
written to ``execution/revision_status.json`` and surfaced in
``revision_dashboard.html``.

| Tier | Condition (ANY fires the tier) | Automated action |
|---|---|---|
| **OK** | None of the below | Trade normally |
| **YELLOW** | MTD return below 25th percentile of backtest monthly returns | Increase logging verbosity; no action |
| **ORANGE** | MTD return below 5th percentile of backtest monthly returns; OR any model's 20-day hit-rate < 0.45 | Notify; run diagnostic attribution; block any pending revision until reviewed by operator |
| **RED** | MTD drawdown < 1.2× backtest 95th-percentile drawdown; OR all ml_signal models simultaneously below 50% hit-rate over 20-day window | Kill-switch fires (see Phase 0); strategy halts; convene formal review |

The kill-switch (`execution/kill_switch.py`) has its own independent
thresholds at the equity level. Red-tier triggers above do not override
the kill-switch — both pathways are present for defence-in-depth.

### Kill-switch thresholds (Phase 0 placeholders)

| Condition | Threshold | Sentinel file | Effect |
|---|---|---|---|
| Hard kill | MTD drawdown ≥ 8% | `KILL_SWITCH_ACTIVE` | Flatten non-{BIL,TLT,GLD}; `main.py` exits |
| Soft halt | MTD drawdown ≥ 5% | `SOFT_HALT_ACTIVE` | Block new ml_signal entries (flip to buy_and_hold in-process) |
| Daily move | One-day NAV move ≥ ±4% | `DAILY_MOVE_ACTIVE` | Skip rebalance for the day |

These placeholders will be replaced in Phase 2.1 with values derived
from the backtest distribution. Both this document and
``KILL_SWITCH_HARD_DD`` / `KILL_SWITCH_SOFT_DD` in
``execution/config.py`` must be updated together, in one commit, with
a rationale.

---

## Allowed revision actions × attribution

When a tier fires, the operator consults this matrix. The
``attribution`` table (`portfolio_oversight`/`execution/attribution.db`)
tells which layer is implicated. The matrix specifies the allowed
action and its trial cost.

| Attribution (Phase 1.3) | Tier required | Allowed actions | Trials cost | Required validation |
|---|---|---|---|---|
| Execution drag dominates | Yellow/Orange | Tune `LIMIT_PRICE_STRATEGY`, slippage params; no model changes | 0 (operational) | A/B against previous week's drag |
| Weight drift, signal OK | Orange | Re-run HRP on updated covariance; deploy via `deploy_models.py --portfolio NEW.json` | +1 | Preflight + 1-week shadow comparison |
| Universe regime change | Orange | Drop tickers failing pre-declared liquidity/data-quality filters only. **No performance-driven drops.** | +1 | Preflight |
| Signal decay (1–2 models) | Orange | Retrain those models on extended data; same architecture, same features | +2 | Full WFOV + DSR + PBO with inflated N |
| Signal decay (≥3 models) | Red | Architecture review; freeze at reduced allocation; ensemble reweighting allowed | +5 | Full WFOV + DSR + PBO; must pass DSR at inflated N |
| Unattributable / unknown | Red | No revisions; halt; investigate root cause | 0 | None — invoke retirement decision |

Any revision proposal MUST pass `scripts/revision_check.py`
(Phase 3) before deployment. That script consults
``algos/wfov/trials_ledger.db`` and refuses the deploy if cumulative `N`
is too high relative to the proposal's DSR.

---

## Capital policy

| State | Maximum live allocation (% of net worth) |
|---|---|
| Build phase (Phases 0–4) | 5–10% |
| Phase 5 — accumulating evidence | scale linearly with months of live data toward target |
| MinTRL elapsed without Red triggers | operator's target allocation |
| After any Red trigger | reset to 5% pending root-cause analysis |

This table is operator-discretion at the upper bound; the protocol
binds only the floor (do not exceed it without writing down why).

---

## Trials counting — what is a trial?

A trial is any of:

- One new WFOV run that proposes a deployable model.
- One new universe definition tested via portfolio optimization.
- One new HRP / Efficient Frontier weight set generated for deployment.
- One change to ensemble composition.
- One re-training run (retraining counts as a trial).
- One change to `signal_conversion` thresholds.

Not trials:

- Operational hot-fixes (config rewriting that doesn't change strategy).
- Bug fixes that restore intended behavior (must be tagged as such in
  the trials ledger with rationale).

---

## Retirement criterion

The strategy is RETIRED (flattened, archived, no redeployment) if ANY:

1. DSR at the current cumulative `N` falls below 0 (i.e., Sharpe
   indistinguishable from zero accounting for multiple-testing inflation).
2. Cumulative trials exceed **50** across configurations tested against
   the *currently deployed* portfolio. **Scope clarification
   (amended 2026-07-07):** the 50-trial cap and the DSR haircut both
   count only trials with `layer IN ('weights', 'universe', 'retrain',
   'architecture')` in `trials_ledger.db`. The `layer='backtest'` rows
   (1145 legacy pre-protocol single-ticker WFOV runs on abandoned
   candidate universes — tickers like `AGI.TO`, `BAB.L` that are not in
   the current 17-ticker portfolio) are excluded from both gates. Current
   scoped count: **7** (6 weights-layer trials from the 2026-05-17
   diagnosis cycle + 1 workflow trial). The full ledger count (1152) is
   retained for audit trail but does not gate revisions.
3. Three Red triggers fire within 12 months.
4. Live Sharpe is materially negative over a full MinTRL window
   following any revision.

Retirement = flatten positions, archive logs/journals, write a
postmortem to `docs/postmortems/`, do not redeploy this strategy
configuration.

---

## Revision workflow (end-to-end)

```text
1. Operator observes a trigger in revision_dashboard.html.
2. Operator consults the attribution table to identify the implicated
   layer.
3. Operator writes a candidate revision into
   docs/revision_hypotheses.md (committed, dated). This is the
   pre-registration step.
4. Operator drafts revision_proposal.yaml with:
      layer: weights | universe | retrain | architecture
      description: ...
      source_wfov_run: path to WFOV summary
5. Operator runs:
      python scripts/revision_check.py revision_proposal.yaml
   The CLI computes the DSR haircut at the inflated trials count and
   prints PASS or REJECT.
6. On PASS, operator runs:
      bash scripts/run_revision.sh revision_proposal.yaml
   which orchestrates WFOV → DSR/PBO → preflight → typed human
   confirmation → deploy → ledger update → git commit.
7. Operator monitors the dashboard daily for the next 4 weeks; if a
   Red trigger fires within that window, the revision is automatically
   considered failed and is rolled back.
```

---

## Amendment procedure

Changes to this document:

1. Open a branch.
2. Edit the file with the proposed change.
3. Write the rationale in the commit message (multi-line).
4. Merge to `feature/revision-protocol` (or successor).
5. If the change weakens any threshold or expands any action allowance,
   the commit message MUST include the cumulative trial count at the
   time of the change.

---

## Sign-off

| Date | Operator | Change |
|---|---|---|
| 2026-05-16 | self | Initial v0 — Phase 0–4 of Revision Protocol shipped. Placeholders for MinTRL and threshold calibration to be filled by Phase 2.1 run. |
| 2026-05-24 | self | MinTRL computed (Day-0 §6) on 86.9%-weight-coverage baseline `portfolio_1d0adc7ddf4b_baseline.json`. Anchored on 75th-percentile Sharpe = 1.11 → MinTRL ≈ 27 months → cooldown ≈ 6.7 months. MinTRL elapses 2028-08-24. Bundled with AAA→BBB rebrand (Track A of 2026-05-18 plan) and `known_gaps` enhancement to `baseline_distributions.py`. See `docs/superpowers/plans/2026-05-24_BASELINE_REMEDIATION_PLAN.md`. |
| 2026-07-07 | self | Q3 quarterly review: MinTRL recomputed (26.83 months, no material change). Orange tier met via GLD/lstm_optimized hit-rate persistence (6 trading days < 0.45) — WFOV diagnostic re-run authorized; retrain/redeploy still gated by Principle #5 pre-MinTRL. Capital scaling divergence identified: actual 52.4% vs expected 5-6% — scale-down action required. IBKR Q2 reconciliation completed (realized P&L +$1,119.42; fees -$940.17). Cumulative trials: 1152 (unchanged this quarter). See `docs/quarterly_reviews/2026-Q3.md`. |
| 2026-07-07 | self | **Trial-count scoping amendment:** retirement criterion #2 and DSR haircut both rescoped to count only `layer IN ('weights','universe','retrain','architecture')` trials, excluding 1145 `layer='backtest'` legacy pre-protocol single-ticker R&D rows on abandoned candidate universes. Current scoped count: 7. Full ledger count (1152) retained for audit trail. DSR at scoped N=7 with deployed-portfolio Sharpe 2.882: strongly significant (DSR > 1.0). Operator decision per Q3 review follow-up. Cumulative trial count at time of change: scoped=7, full=1152. |
| 2026-07-07 | self | **Tier A/B discipline amendment:** Operating Principle #5 replaced with two-tier system. Tier A (MinTRL, DSR, cooldowns) now advisory — `revision_check.py` computes and prints but does not block; override requires one logged sentence in `revision_hypotheses.md`. Tier B (kill-switch 5%/8% MTD DD + new per-position circuit breaker at 2%/10td) stays hard and non-negotiable. DSR-gating made fully optional (warn-only). Rationale: retail trader, no institutional stakeholders, operator's own capital, operator's explicit decision per 2026-07-07 review. See `docs/CAPITAL_ALLOCATION_PLAN.md`. |
