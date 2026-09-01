# Research Manual — From Idea to Deployment to Long-Term Monitoring

**Status:** ACTIVE.
**Last updated:** 2026-06-09.
**Owner:** solo operator (you).
**Scope:** The full research lifecycle for the IBKR global-equities/ETF strategy — from a raw idea, through universe construction, portfolio exploration, OOS validation, model selection, WFOV, apples-to-apples comparison, validation gates, deployment, and into long-term monitoring.

**Who this is for and how to read it.** This manual is a *teaching instrument*, written to you, in the second person. Its job is to make the research discipline a habit — so that *you* run the pipeline correctly, from memory, under your own judgment. It is **not** a script for an agent to execute on your behalf. The division of labor is fixed:

> **Agents debug, troubleshoot, fix root causes, and teach. You operate.**
> An agent may help you find a bug, repair a script, or explain a gate.
> An agent must never silently run your live pipeline, maintain your crontab, re-baseline for you, or deploy for you. Keeping tabs is your job; this manual exists so you can keep them well.

**Document scope.** This is the *research/build front half* of the lifecycle. Its companions:

- `docs/OPERATIONS_MANUAL.md` — the *run* half (daily/weekly/monthly/quarterly/annual operations once a strategy is live).
- `docs/REVISION_POLICY.md` — the *what is allowed* (binding anti-data-mining contract: MinTRL, trials budget, DSR gate, retirement criterion).
- `docs/superpowers/plans/PORTFOLIO_REVISION_DIAGNOSIS_PLAN.md` — the diagnosis playbook (the single most reusable "should I revise?" guide).
- `docs/revision_hypotheses.md` — the pre-registration log.
- `docs/WORKFLOW_GUIDE.md`, `docs/MODEL_SELECTION_GUIDE.md`, `docs/DATA_WORKFLOW.md`, `docs/WFOV_V2_SUMMARY.md` — tool-level guides.

**Precedence rule.** When this manual and `REVISION_POLICY.md` appear to conflict for a **live** strategy, the policy wins. This manual may explore more freely **only** before a strategy holds real capital (see §1).

**Maintenance cadence:** review at each quarterly audit; amend after any lifecycle change. Markdown is the source of truth; you export `.odt`/`.pdf` yourself.


---

## §1 — Philosophy and the two-research-regime rule

Read this section before any research session. It is the frame everything else hangs on.

### §1.1 — The two research regimes (the hard line)

Your research operates in exactly one of two **research regimes** at any moment. Know which one you are in *before* you touch a tool.

> **Terminology — read this once and never confuse it again.** "Regime" in this manual ALWAYS means your **research regime** (greenfield vs live) — the question "which rulebook governs what I'm about to do?". It NEVER means the **market regime** (bull/bear, risk-on/risk-off, high/low volatility). The two are orthogonal: you can be greenfield in a bull market, or live in a bear market. When you read "identify the regime," it means "greenfield or live?" — not "are we bullish?". A market-regime observation ("we're risk-on; I'm down 3% vs the S&P MTD") is an *urge input* (§4.1), never the research regime, and is **never by itself a reason to revise a live strategy** (that is market-timing / sub-MinTRL action — §9, §12).

| | **GREENFIELD** | **LIVE REVISION** |
|---|---|---|
| Definition | No real capital is deployed for this idea yet | A strategy is live with capital, and you are considering changing it |
| Governing doc | This manual (§4–§11) | `REVISION_POLICY.md` is binding; this manual is subordinate |
| Cadence | May explore freely, iterate fast | MinTRL/cooldown/trials/pre-registration **strictly apply** |
| Pre-registration | Encouraged (idea log) | **Mandatory** (`revision_hypotheses.md`) before any test |
| "Deploy when degradation < 20%" | Allowed as a *screening* heuristic | **Forbidden** — deployment is gated by DSR/PBO/MinTRL/trials |

**The reconciliation.** An older, aggressive workflow preaches an aggressive cadence — "optimize Monday, validate Tuesday, deploy Wednesday", "weekly portfolio update", "deploy at degradation < 20%". That cadence is **valid only in GREENFIELD**. The instant a strategy is live, that cadence is **forbidden** and the Revision Protocol governs: you may not revise on sub-MinTRL noise, you must pre-register, every test costs a trial, and the deploy gate is DSR ≥ 0.5 at inflated N — not a degradation threshold.

If you ever catch yourself running the aggressive cadence against a live strategy, **stop**. That is the exact failure the Revision Protocol was built to prevent.

### §1.2 — Core ideals (memorize these)

1. **Capital before code.** Reducing capital at risk is always available, always reversible, and never costs a trial. Code changes are slow, risky, and burn your trials budget. When in doubt, scale down capital, not up complexity.
2. **Survival beats speed.** Compounding only matters if you are still in the game. A slower, surviving strategy dominates a faster, dead one.
3. **The deployed default is "do nothing".** Inaction is a position. If the protocol does not *authorize* a revision, the correct action is **not to revise** — not to find a workaround.
4. **Reproducibility is non-negotiable.** Fixed seeds, auto-saved scalers, pinned data windows. A result you cannot reproduce is not a result.
5. **Every claim must be falsifiable.** "This looks good" is not a claim. "OOS Sharpe ≥ X on a non-overlapping window, surviving 2× transaction costs, with DSR > 0.5 at N trials" is a claim.
6. **Pre-register before you test.** Writing the hypothesis down *first* is the single cheapest defense against data-mining yourself.

### §1.3 — The one-sentence test before any session

> "Am I in greenfield or live? Have I written down what I expect to find, and what would falsify it, *before* running anything?"

If you cannot answer both, you are not ready to run a tool yet.

---

## §2 — Your failure modes (personalized guardrails)

This section is candid by design. These are *your* documented biases — found in your own code review (the Lopez de Prado behavioral guidance) and your own audit trail (the ~6–20 unregistered exploration runs acknowledged in `revision_hypotheses.md`, 2026-05-17). Each is stated as **the lie / the truth / your guardrail**, and names the stage that enforces the fix.

### §2.1 — "More iterations = more confidence"
- **The lie:** 200 Monte Carlo iterations feels rigorous; the table is long.
- **The truth:** 200 *overlapping* windows may contain only 5–15 independent observations. The reported p=0.001 may truly be p=0.15.
- **Your guardrail:** Ask "how many **non-overlapping** test periods do I have?" That is your real sample size. Monte Carlo is for *screening*, not inference. → Enforced in §7, §8.

### §2.2 — "The highest-Sharpe model is the best"
- **The lie:** The ranking sorts by score; the top row must be best.
- **The truth:** The score uses arbitrary weights; change them and a different model wins. Worse: Sharpe 1.0 in a market with B&H Sharpe 1.2 is *negative alpha*. And IS Sharpe > 1.5 on daily long-only equity is almost certainly overfit.
- **Your guardrail:** Use **excess Sharpe** (vs buy-and-hold) and **deflated Sharpe**, never absolute. Treat IS Sharpe > 1.5 as an overfitting alarm. → Enforced in §6, §8, §10.

### §2.3 — "Passing validation means it works in production"
- **The lie:** A Tier-1 "DEPLOY" green check feels like a guarantee.
- **The truth:** WFOV validates a model *class* on historical subsets. Deployment retrains from scratch. There is zero guarantee the retrained model matches the validated performance.
- **Your guardrail:** Treat a pass as "worth **paper-trading**", not "deploy real money tomorrow". Paper-trade 1–3 months first. → Enforced in §8, §12.

### §2.4 — "Low transaction costs are realistic"
- **The lie:** PTC = 0.035% is what the broker charges.
- **The truth:** Commission is 10–30% of real cost. Spread + impact + slippage dominate; a daily long/short flip incurs ~4 cost events per round-trip.
- **Your guardrail:** Screen at **2–3× PTC**. If Sharpe goes negative at 10 bps, there is no margin of safety. → Enforced in §8.

### §2.5 — "Profile A is fine for my use case"
- **The lie:** You're willing to take risk, so you pick the aggressive profile.
- **The truth:** Profile A accepts p < 0.15, hit ratio < 0.50 (worse than random), min Sharpe 0.25 (marginal after costs). It will recommend models with no edge.
- **Your guardrail:** **Profile B is your deployment minimum** (p < 0.05, Sharpe > 0.4, hit ratio > 0.50). Profile A is for exploration only. → Enforced in §8.

### §2.6 — "13 models provide meaningful diversity"
- **The lie:** SVM, XGB, LSTM, DQN… feels like thorough exploration.
- **The truth:** All FeatureEngine models consume the same ~36 indicators. That is 13 function approximators on the *same signal* — an ensemble, not a selection process. If the signal is noise, all 13 fit noise.
- **Your guardrail:** True diversity needs **different information sources** (price-volume vs macro vs sentiment), not different classifiers on the same features. → Enforced in §8.

### §2.7 — "The framework catches overfitting"
- **The lie:** Deflated Sharpe + multiple-testing awareness feels like protection.
- **The truth:** If DSR uses n_trials = iterations instead of n_trials per model, and BH-FDR is never applied, the framework cannot distinguish a good model from the luckiest of twelve.
- **Your guardrail:** Ensure DSR uses the correct trial count, and ask: "Would I trust this if I had tested **one** model instead of twelve?" → Enforced in §10.

### §2.8 — "I'll just check this idea quickly" (the unregistered-run trap)
- **The lie:** A quick backtest "just to see" is harmless.
- **The truth:** Observing live data, pattern-matching a hypothesis, then testing it on the *same* data is a one-step data-mining loop. You did this 6–20 times before 2026-05-17; you logged it as a process failure.
- **Your guardrail:** **No test without a pre-registered entry first.** If you catch yourself running a backtest without an entry in your idea log (greenfield) or `revision_hypotheses.md` (live), stop and write it down. → Enforced in §4.

### §2.9 — Pre-flight checklist (run this before trusting ANY result)

- [ ] Did I use **Profile B or C** (not A)?
- [ ] Did **walk-forward** validation pass (not just Monte Carlo)?
- [ ] Is **excess Sharpe** positive (beats buy-and-hold)?
- [ ] Does it survive **2× transaction costs**?
- [ ] Is the **failure rate** below 5%?
- [ ] Would the **same model win with a different seed**?
- [ ] How many **non-overlapping** windows do I really have?
- [ ] Did I **pre-register** before testing?
- [ ] Am I **paper-trading** before real capital?

---

## §3 — The end-to-end lifecycle map

One glance. Each stage has a tool, a gate to pass before the next, and a governing document.

| Stage | What | Primary tool(s) | Gate to advance | Governs |
|---|---|---|---|---|
| A | Idea & pre-registration | idea log / `revision_hypotheses.md` | Written down *before* testing | §4, REVISION_POLICY |
| B | Universe & data | `filter_investable_universe.py`, `validate_data_csv.py` | Clean data, ≥95% coverage, no look-ahead | §5, DATA_WORKFLOW |
| C | Portfolio exploration | `portfolio_exploration_global.py`, `portimization.py` | IS Sharpe ≤ 1.5 alarm respected; sane universe | §6 |
| D | OOS validation | `validate_portfolio_oos.py`, `validate_fixed_portfolio.py` | Degradation acceptable AND stable (necessary, not sufficient) | §7 |
| E | Model selection / backtest / WFOV | `model_selection_workflow.py`, `run_backtest_optimized.py`, `wfov_runner.py` | Profile B+, WFOV pass, survives 2× PTC | §8 |
| F | Apples-to-apples | re-run D/E on deployed AND candidate | Candidate beats deployed by documented margin | §9, DIAGNOSIS_PLAN §2 |
| G | Validation gates & trials | `revision_check.py`, trials ledger | DSR ≥ 0.5 at inflated N; under trials/retirement limits | §10, REVISION_POLICY |
| H | Deployment | `deploy_models.py`, `run_revision.sh` | Trials-gate + typed `DEPLOY` confirm | §11 |
| I | Post-deployment & monitoring | `daily_routine.py`, OPERATIONS_MANUAL | Paper-trade first; capital scales slowly | §12, OPERATIONS_MANUAL |

LLM tools (dexter, TradingAgents) sit *beside* this pipeline as a qualitative overlay (§13) — never inside a gate.

---

## §4 — Stage A: Idea and pre-registration

**Why this stage exists.** Pre-registrWation is the cheapest, most powerful anti-data-mining device you have. Writing down what you expect — and what would falsify it — *before* you test prevents you from rationalizing noise into signal after the fact.

**What you do.**
- **Greenfield:** keep an idea log (a dated section in your research notes). One entry per idea: observation, proposed test, what you expect, what would falsify it, and the date.
- **Live:** the entry goes in `docs/revision_hypotheses.md` using the template there. It is **binding** — no test may run before the entry exists, and no promotion to a `revision_proposal.yaml` before the `Deferred until` date (≥ MinTRL/4 from observation).

**The trap (→ §2.8).** "I'll just check it quickly." No. The check *is* the test. Write the entry first.

**The gate.** An idea may advance to Stage B only when its entry exists and (if live) its cooldown has elapsed. If the cooldown has not elapsed, the correct action is to wait — not to test "just to see".

**Checklist.**
- [ ] Research regime identified (greenfield vs live)?
- [ ] Entry written *before* any tooling?
- [ ] Falsification condition stated?
- [ ] (Live) cooldown date computed and respected?

### §4.1 — Triage: from urge to hypothesis (HARD blocker before research)

Most research does not begin with a clean idea. It begins with an **urge** — "the market moved, let me check for a better portfolio", "mine is meh while everything's soaring, I need to update". This subsection converts an urge into something you are *allowed* to test. **You may not run any tool until an urge clears every step below.** This is a blocker, not advice.

#### Step 1 — Is this even greenfield?

> **Default assumption: if the urge touches the portfolio you currently trade, you are in LIVE, not greenfield.**

Greenfield = no real capital is deployed *for this idea*. A genuinely new strategy class, or a universe you do not hold. "Test if a new portfolio beats mine" and "update my portfolio" are about your **live** book — they are live revisions wearing a playground costume, and `REVISION_POLICY` binds (MinTRL, cooldown, trials, pre-registration). Calling them "just playground exploration" is the first lie to catch.

> **"I'll just test it, I won't deploy" does NOT make a live idea greenfield.** The research regime is set by *what the idea is about*, not by what you promise to do with the result. "Does a new HRP beat *mine*?" is a question about your live book → LIVE, full stop. Two reasons the promise is worthless as a safeguard:
>
> 1. **The trial is counted at the TEST, not the deployment.** Every comparison you run against your live portfolio inflates your cumulative N under multiplicity accounting — whether or not you deploy. "Just looking" five times this month silently raises the DSR bar for the *next real* decision. Unregistered "just testing" is not free; it taxes your future.
> 2. **The data-mining loop fires on observation, not on action.** Observing live data → forming a hypothesis → testing it on the same data is the one-step loop (§2.8) regardless of whether you trade. The damage is to your *inference*, which happens at the test.
>
> You cannot pre-commit your future self out of a bias. "I'll only look" is the exact story that preceded the 6–20 unregistered runs. **Heuristic: if the answer could change how you feel about your deployed money, it is a live idea — log it, register it, gate it.** The genuinely greenfield version studies HRP on a universe you do NOT hold ("do I understand HRP's behavior?"), never against your deployed weights.

A **market-regime observation is not your research regime.** "We're risk-on and bullish" or "my book is down 3% vs the S&P MTD" describes the *market* and your *relative performance* — it is an **urge input** to feed into Step 3, not the greenfield/live classification. Note too: a number is necessary but not sufficient. "Down 3% vs S&P MTD" is quantified yet still ~1 month of data = sub-MinTRL noise, so even a metric-form urge usually resolves to "journal and wait" (§9, §12). Quantified ≠ actionable.

#### Step 2 — Name the unit(s): one falsifiable claim = one idea = one trial

The unit of research is the **hypothesis**, never the *session*. A single sitting may contain several ideas; blurring them is how you run twenty trials and call it "one exploration" (your 2026-05-17 failure mode, §2.8).

> **Splitting rule: if it has its own failure condition, it is its own entry, and it counts as its own trial.**

- "A fresh HRP candidate beats deployed on the same OOS window by ≥ the documented margin" → **one** idea.
- "...AND 40 tickers beat 30 AND max_sharpe beats HRP this regime" → **three** ideas, **three** trials. Log them separately or do not run them.

#### Step 3 — The falsifiability blocker (must pass ALL FOUR)

Apply to each named idea. If it fails any one, you do **not** research it — you journal it and stop.

- [ ] **(1) Metric, not mood.** Stated with a number and a comparison ("below the 25th percentile of its backtest distribution", "OOS Sharpe ≥ X"), not a feeling ("meh", "soaring", "behind", "desperate urge").
- [ ] **(2) Pre-stated failure condition.** You can write, *before* testing, the result that would make you abandon it. If you cannot imagine being wrong, it is not a hypothesis.
- [ ] **(3) Out-of-sample, same window.** The claim is about data the candidate did not see, compared apples-to-apples against the incumbent — not "looks good in-sample".
- [ ] **(4) Survives multiplicity.** It would still hold if you had tried it once, not twenty times (DSR at inflated N).

**Mood-words are the tell.** "meh", "soaring", "I feel behind", "desperate urge" mean you are at Step 0, not Step 1. The correct move is to either *convert the feeling into a metric* (then re-run the four checks) or **journal it and do nothing**.

#### Step 4 — Pre-register, then respect the cooldown

Each idea that passes Step 3 gets an entry — `revision_hypotheses.md` (live) or your idea log (greenfield) — written *before* any tool runs. If live and the `Deferred until` date has not arrived, you may **build the candidate as research but you may NOT act on the comparison**. One month of new data is sub-MinTRL noise; acting on it is the §2.8 trap.

#### §4.1 worked examples (generic, to illustrate the judgment)

These illustrate the four-step blocker above; they are not additional steps.

**Example A — calendar-driven re-test ("market moved a month; does a new HRP beat mine?").**
- Research regime: **LIVE** (it is your deployed book). Not greenfield.
- Unit: one idea (one comparison, one margin).
- Falsifiability: PASSES if framed as *"a freshly generated HRP candidate beats the deployed portfolio on the same OOS window by ≥ the documented margin, surviving DSR at inflated N."* Metric ✓, failure condition ✓ (candidate fails to beat by the margin → abandon), OOS same-window ✓, multiplicity ✓.
- Verdict: legitimate hypothesis, but **cooldown-gated**. You may generate the candidate as research; you may **not** deploy before the deferred date. One month of data does not move MinTRL. → Pre-register, build if you wish, wait to act.

**Example B — the "meh while everything's soaring" urge.**
- Research regime: **LIVE**, and the input is an *emotion*.
- Falsifiability: **FAILS at check (1)** as stated — "meh" and "soaring" are moods, not metrics. There is no failure condition you can write.
- To rescue it, convert: *"My portfolio's MTD return is below the 25th percentile of its backtest monthly distribution AND a candidate beats it on the same OOS window."* Now it has a metric and a failure condition → re-run the four checks. (Note: even rescued, if you are below MinTRL the live Sharpe comparison is statistically meaningless — §9, §12 — so it likely still resolves to "journal and wait".)
- Verdict if not converted: **journal the feeling, do nothing.** This is the exact urge that produced the unregistered runs and the 2026-05-17 "mediocre performance" misjudgment. Doing nothing here is a *successful* outcome.

#### §4.1 one-line triage you can say out loud

> "Greenfield or live? How many distinct failure conditions am I really testing? Is each one a metric with a pre-stated way to be wrong, out of sample, that survives multiplicity? If any answer is fuzzy — journal, don't run."

---

## §5 — Stage B: Universe construction and data hygiene

**Why this stage exists.** Every downstream result inherits the quality of your universe and data. Survivorship bias, look-ahead, and stale prices are silent killers — they inflate backtests and evaporate live.

**What you do.**

1. **Filter the investable universe** down to what you can actually trade at
   your budget and liquidity:
   ```bash
   python algos/backtest_code/filter_investable_universe.py \
       --from-store --start 2020-01-01 --end 2026-01-01 \
       --budget 50000 --auto-weight \
       --min-avg-dollar-volume-30d 5000000 \
       --output data/investable_tickers.txt
   ```

   Key flags: `--budget` (required), `--min-weight` OR `--auto-weight` (required), `--min-avg-dollar-volume-30d` (default 5e6), `--lot-sizes`, `--exclude-exchanges`, `--exchange-rates`, `--output`/`-o`. The output file feeds `portfolio_exploration_global.py --tickers-file`.

2. **Download data** for 3, 5, and 7-year lookbacks (Step 0): longer windows capture more regimes and resist overfitting.

3. **Validate the data every time you download** (do not skip this):
   ```bash
   python scripts/validate_data_csv.py \
       --csv data/financial_data_combined_prices_2023-01-27_2026-01-21_1d.csv \
       --start 2023-01-27 --end 2026-01-21
   ```

**The traps.**
- **Survivorship bias:** a universe of "tickers that exist today" hides the ones that died. Check asset coverage (§7) and treat < 95% coverage as a warning, < 60% as invalid.
- **Look-ahead:** never let test-period information leak into the training window. Use the purge + embargo discipline (§8).
- **Stale data:** a frozen pipeline (you saw this 2026-05-22) silently poisons baselines. Confirm the last row's date before trusting anything.

**The gate.** Clean validation, ≥ 95% asset coverage on the test window, and data freshness confirmed. Otherwise fix the data before exploring.

**Checklist.**
- [ ] Universe filtered to tradeable names at budget + liquidity?
- [ ] Data downloaded at 3/5/7-yr lookbacks?
- [ ] `validate_data_csv.py` clean?
- [ ] Coverage ≥ 95%? Freshness confirmed?

---

## §6 — Stage C: Portfolio exploration

**Why this stage exists.** This is where you distill a large universe into a candidate portfolio. It is also where overfitting is born — the more you search, the more you risk fitting noise. Treat the IS Sharpe as a suspect, not a trophy.

**What you do.**

Run the 3-stage exploration:

```bash
python algos/backtest_code/portfolio_exploration_global.py \
    --from-store --start 2020-01-01 --end 2026-01-01 \
    --tickers-file data/investable_tickers.txt \
    --min-sharpe 0.5 --stage2-target 40 \
    --output-weights data/candidate_weights.json \
    --output-weights-type hrp --seed 42
```

The stages (from the code):
- **Stage 1 — multi-criteria screening** (`--stage1-top-n`, `--min-sharpe`, `--min-annual-return`, `--min-trading-days`, `--max-correlation`): cut the universe to the top N by quality.
- **Stage 2 — direct selection** (`--stage2-target`): pick the target count by composite score.
- **Stage 3 — global optimization** (`--max-weight`, `--risk-free-rate`): produce `max_sharpe`, `min_volatility`, and `hrp` weights + efficient frontier. `--output-weights-type` chooses which to export (`hrp` default — see why in §7).

For a lower-dimensional, production-share-aware optimization (efficient frontier, target volatility, integer shares via MIQP):

```bash
python algos/backtest_code/portimization.py \
    --from-store --start 2020-01-01 --end 2026-01-01 \
    --target_volatility 0.2 --mode miqp --budget 50000 \
    --export_weights target_vol --seed 42
```

**The traps (→ §2.2, §2.6).**
- **IS Sharpe > 1.5 alarm.** On daily long-only equity, anything above ~1.5 is suspect; above ~2.0 is almost certainly overfit. A high IS Sharpe is a *warning*, not a win.
- **Universe sanity gates** (from DIAGNOSIS_PLAN §5): cap single-name beta, dedupe sector/region concentration, require ≥ 95% coverage. A portfolio that is secretly one bet is not diversified.
- **Objective-mining:** do not try ten objective functions and keep the one that looks best. That is data-mining with extra steps.

**The gate.** A candidate with a *defensible* IS Sharpe (not maxed), a sane universe, and a fixed seed. Advance to OOS validation (§7) — never deploy on IS results (→ §2.3).

**Checklist.**
- [ ] IS Sharpe ≤ ~1.5 (or alarm acknowledged and justified)?
- [ ] Universe sanity gates passed (beta, sector/region, coverage)?
- [ ] Seed fixed; run reproducible?
- [ ] Did NOT pick the objective post-hoc?

---

## §7 — Stage D: Out-of-sample validation (your 6-step, hardened)

**Why this stage exists.** IS performance is the easiest thing to fake. OOS validation is the first honest test: does the portfolio (or the *process* that built it) survive on data it never saw?

### §7.1 — Which validator? (know the difference)

| | `validate_portfolio_oos.py` | `validate_fixed_portfolio.py` |
|---|---|---|
| Tests | The optimization **strategy** (re-optimizes each window) | **Specific weights** (no re-optimization) |
| Answers | "Should I re-optimize periodically?" | "Can I buy-and-hold these weights?" |
| Output | IS→OOS degradation | Consistency over time (CV) |

### §7.2 — Strategy validation

```bash
python algos/backtest_code/validate_portfolio_oos.py \
    --from-store --start 2020-01-01 --end 2026-01-01 \
    --portfolio both --mode expanding \
    --train-years 2 --test-months 12 --embargo-pct 0.02 --seed 42
```

Modes: `expanding` (default; training grows — mimics real life; use for deployment decisions), `rolling` (fixed-size sliding window — use for regime-dependence detection), `monte_carlo` (50–200 random windows — high *apparent* power; **screening only**, see the trap below). `--portfolio both` compares max_sharpe vs hrp; **HRP frequently wins OOS despite lower IS** — that is why it is the default export type.

### §7.3 — Fixed-weight stability

```bash
python algos/backtest_code/validate_fixed_portfolio.py \
    --weights "XYZ.MI:0.033,BIL:0.126,..." \
    --from-store --start 2020-01-01 --end 2026-01-01 \
    --test-period-months 12
```

Or extract weights from an exploration log with `--log <log> --portfolio hrp`.

### §7.4 — Reading the results (necessary, not sufficient)

Degradation = (IS Sharpe − OOS Sharpe) / IS Sharpe:

| Degradation | Verdict | Action |
|---|---|---|
| < 10% | Excellent | Strong candidate |
| 10–20% | Acceptable | Candidate, monitor |
| 20–30% | Caution | Review before proceeding |
| > 30% | Reject | Do not proceed |

Stability: OOS Sharpe std < 0.3 (very stable) … > 1.0 (breaks in some regimes). Fixed-weight CV < 0.15 (very stable) … > 0.50 (reject).

**The traps (→ §2.1, §2.4).**
- **Non-overlapping windows are your real n.** Monte Carlo's 200 iterations do *not* give you 200 independent observations. Count the non-overlapping windows and apply the humility of that small n.
- **Degradation thresholds are necessary, not sufficient.** Passing them (greenfield screening) does **not** authorize a live deployment. Live deployment additionally requires DSR/PBO/MinTRL/trials (§10).
- **0% degradation is suspicious** — it can indicate data leakage.

**The gate.** Greenfield: degradation < 20% AND stable → advance to model selection (§8). Live: this is a screen only; you still face §9 and §10.

**Checklist.**
- [ ] Used the right validator (strategy vs weights)?
- [ ] Counted non-overlapping windows?
- [ ] Degradation < 20% AND OOS std < 0.5 (or CV < 0.30)?
- [ ] Compared `--portfolio both` (HRP vs max_sharpe)?
- [ ] Understood this is a screen, not a deploy authorization?

---

## §8 — Stage E: Model selection, backtesting, and WFOV

**Why this stage exists.** The portfolio decides *what* to hold; the per-ticker ML models decide *when* to lean in. This is the stage most prone to multiple-testing inflation, so the discipline is strictest here.

**What you do.**

1. **Per-ticker model selection** (orchestrates WFOV, ranks models):
   ```bash
   python scripts/model_selection_workflow.py \
       --ticker SPY --preset comprehensive --profile B \
       --validation-mode walk_forward_expanding --seed 42
   ```

   `--profile` is the ranking standard: **A = max returns (exploration only), B = risk-adjusted (your deploy minimum), C = institutional**. `--preset quick|comprehensive` selects the model set; `--validation-mode monte_carlo|walk_forward_expanding|walk_forward_rolling`.

2. **Single backtest** (lower-level, for one model/ticker):
   ```bash
   python algos/backtest_code/run_backtest_optimized.py \
       --model_name svm_optimized --ticker SPY \
       --lookback_days 1260 --embargo_pct 0.02 --ptc 0.00035
   ```

   The 2% embargo (López de Prado) prevents train/test leakage. **Re-run at 2–3× PTC** to test the margin of safety.

3. **WFOV** (the core walk-forward / Monte Carlo validation with the full statistical suite — DSR, PBO, Newey-West, bootstrap CIs):
   ```bash
   python -m algos.wfov.wfov_runner \
       --mode walk_forward_expanding --model_name svm_optimized \
       --ticker SPY --initial_train_days 1260 --test_days 252 \
       --ptc 0.00035 --seed 42
   ```

   Modes: `monte_carlo` (requires `--iterations`), `walk_forward_expanding` (`--initial_train_days`), `walk_forward_rolling` (`--window_size`). The ranker emits DEPLOY / REVIEW / REJECT tiers.

**The traps (→ §2.2, §2.5, §2.6, §2.7).**
- **Excess Sharpe, not absolute.** A model must beat buy-and-hold. `excess Sharpe > 0` is a hard gate.
- **Profile B minimum.** Never deploy on Profile A.
- **Seed stability.** Run the selection across multiple seeds. If a different model wins each time, you are looking at noise.
- **Fake diversity.** 13 classifiers on the same 36 indicators is one experiment, not thirteen. Count it as such for multiple-testing.
- **2× transaction costs.** If it dies at 10 bps, it has no margin.

**The gate.** WFOV pass at Profile B+, excess Sharpe > 0, survives 2× PTC, stable across seeds, failure rate < 5%. Reproducibility: seeds fixed, scalers auto-saved (so live matches backtest).

**Checklist.**
- [ ] Profile B or C (not A)?
- [ ] Walk-forward passed (not just Monte Carlo)?
- [ ] Excess Sharpe > 0?
- [ ] Survives 2× PTC?
- [ ] Same winner across seeds?
- [ ] Embargo applied; scalers saved; seeds pinned?

---

## §9 — Stage F: Apples-to-apples comparison (mandatory before any swap)

**Why this stage exists.** This is the single highest-leverage step in the entire lifecycle, and the one you skipped before 2026-05-17. Before replacing anything live, you must prove the candidate beats the *currently deployed* portfolio on the **same** out-of-sample window. The answer often resolves the whole question in ~20 minutes.

**What you do.** Validate BOTH the deployed weights and the candidate weights over the identical OOS window (use `validate_fixed_portfolio.py` for each, or re-run the WFOV comparison), then apply the decision tree from `PORTFOLIO_REVISION_DIAGNOSIS_PLAN.md` §2:

- Deployed Sharpe beats all candidates by ≥ 0.3 → **deployed wins outright; no deployment; re-test later.**
- Candidate clearly beats deployed by a documented margin → proceed to §10.
- Draw → keep the incumbent (changing costs a trial for no edge).

**The trap (relative vs absolute — diagnosis_outcome §3).** A favorable OOS window (e.g. the deployed's 2.88 Sharpe over 5.5 months) is sufficient evidence for a *relative* claim ("deployed beats candidate") but **not** for an *absolute* claim ("durable skill"). At 5.5 months the standard error on Sharpe is roughly ±1.0. A good window never justifies accelerating capital (§12).

**The gate.** Candidate beats deployed by the documented margin on the same window. Otherwise: do nothing.

**Checklist.**
- [ ] Deployed AND candidate validated on the SAME window?
- [ ] Decision tree applied?
- [ ] Resisted the urge to read an absolute claim into a relative one?

---

## §10 — Stage G: Validation gates and trials accounting

**Why this stage exists.** This is the contract that keeps you honest across *many* revisions. Each test you run inflates your cumulative trial count; the deploy gate must account for all of them, or you will eventually deploy the luckiest of fifty tries.

**What you do (live revisions).**

1. Pre-register the hypothesis (§4), let the cooldown elapse.
2. Draft a `revision_proposal.yaml` (layer + `source_wfov_run` + description).
3. Run the DSR check:
   ```bash
   python scripts/revision_check.py path/to/revision_proposal.yaml
   ```

   It computes the **Deflated Sharpe Ratio at the current cumulative N** from the WFOV run. **Threshold = 0.5.** Exit 0 = PASS, 1 = REJECT, 2 = error.

**The gates and limits (from REVISION_POLICY).**
- **DSR ≥ 0.5** at inflated N to deploy.
- **PBO** (probability of backtest overfitting) low.
- **MinTRL** (~27 months; cooldown ~6.7 months) — no Sharpe-based deployment judgment is statistically meaningful before it elapses.
- **Trials budget:** cumulative trials must stay under **50** (a retirement criterion). Know what counts as a trial: a new WFOV proposing a deployable model, a new universe, new weights, an ensemble change, retraining, or a signal-threshold change. Bug fixes and operational hot-fixes are **not** trials (but must be tagged as such).
- **Retirement** if ANY: DSR at cumulative N < 0; cumulative trials > 50; three Red triggers in 12 months; live Sharpe materially negative over a full post-revision MinTRL window.

**The trap (→ §2.7).** "The framework caught overfitting." Only if DSR uses the correct trial count and you count fake diversity (§2.6) honestly. Ask: "Would this survive if I'd tested one model, not twelve?"

**The gate.** PASS from `revision_check.py` AND under all trials/retirement limits AND MinTRL respected.

**Checklist.**
- [ ] `revision_check.py` returned PASS (DSR ≥ 0.5 at inflated N)?
- [ ] Cumulative trials < 50 after this one?
- [ ] MinTRL elapsed (or this is greenfield)?
- [ ] No retirement criterion tripped?

---

## §11 — Stage H: Deployment

**Why this stage exists.** Deployment is the one irreversible-with-real-money step. It is gated on purpose.

### §11.1 — Greenfield / initial deploy

```bash
python deploy_models.py --portfolio data/candidate_weights.json --dry-run
python deploy_models.py --portfolio data/candidate_weights.json
```

`deploy_models.py` enforces a **trials-ledger gate**: it requires an accepted `weights` trial in `algos/wfov/trials_ledger.db` whose source references your portfolio file, else it exits with code 3. (`--bypass-trials-check` exists but is audited — logs a timestamp + git hash; use only for a documented operational hot-fix.) It then rewrites `execution/config.py` (`TARGET_ALLOCATION`, `ASSET_SPECIFIC_CONFIGS`, `SYMBOLS`) and deploys model/scaler files.

After config is written, hand off to **OPERATIONS_MANUAL §2** (Day-0 setup: preflight, reduce capital to 5%, baseline, MinTRL, dashboard) — and **you** install the crontab (it is your job, not an agent's).

### §11.2 — Live revision (the gated chain)

```bash
bash scripts/run_revision.sh path/to/revision_proposal.yaml
```

Ordered gates (aborts on any failure): retirement check → DSR check (`revision_check.py`) → WFOV gate (only for `retrain`/`architecture`) → preflight (`preflight_check.py --nav`) → **typed human confirmation (you must type `DEPLOY`)** → deploy → register accepted trial (`--commit`) → manual sign-off reminders.

**The trap.** Never bypass `run_revision.sh` for a live strategy change. "I'll just edit config.py directly" skips every gate and burns no trial on the ledger — which corrupts your entire trials accounting.

**The gate.** Typed `DEPLOY` only after every prior gate passed. If any gate fails, the deploy aborts and you do nothing.

**Checklist.**
- [ ] (Greenfield) `--dry-run` reviewed before the real run?
- [ ] (Live) used `run_revision.sh`, never a manual config edit?
- [ ] Trials ledger updated (accepted trial recorded)?
- [ ] Handed off to OPERATIONS_MANUAL §2; you (not an agent) installed cron?

---

## §12 — Stage I: Post-deployment and long-term monitoring

**Why this stage exists.** Deployment is the start of the hard part. Most of the discipline lives *after* you go live, and it is governed by `OPERATIONS_MANUAL.md`, not this manual.

**What you do.**
- **Paper-trade first (→ §2.3).** A WFOV pass authorizes paper-trading, not real capital. Run 1–3 months of paper/observation before committing.
- **Run `python execution/daily_routine.py` every morning.** It is your one-command daily check (sentinels, last-cron health, missed-run/staleness detector, tier). It surfaces problems; it does not fix them — that is your judgment call.
- **Follow the OPERATIONS_MANUAL cadence:** daily (§3), weekly (§4), monthly (§5: re-run baselines, hit-rate trends), quarterly (§6: MinTRL recompute, hypotheses audit), annual (§7: postmortem, retire/scale decision).
- **Scale capital slowly:** start at 5% of NAV, add 1pp per *clean* month, cap at 50%; any Red trigger resets you to 5%. Speed of capital growth is not a virtue; staying alive is.
- **The 4-week watch:** after any accepted revision, monitor closely; a Red trigger in that window is an auto-rollback signal.

**The trap.** A good early window tempts you to scale faster. Do not. Below MinTRL, your live Sharpe is statistically meaningless (→ §9).

**The gate (back into the lifecycle).** A *real* trigger (per `REVISION_POLICY` tiers), observed and aged past cooldown, is the only thing that sends you back to Stage A for a live revision. Noise does not.

**Checklist.**
- [ ] Paper-traded before real capital?
- [ ] `daily_routine.py` run each morning; anomalies logged (not acted on)?
- [ ] Capital scaling rule honored (5% → +1pp/clean month → 50%)?
- [ ] No revision attempted on sub-MinTRL noise?

---

## §13 — Qualitative overlay: dexter and TradingAgents

**Why this section exists.** You have two LLM research tools. They are genuinely useful for *thinking* — and genuinely dangerous if you let them near a deploy gate. This section fixes their role.

**The hard rule.** These tools are a **qualitative overlay**. They are NEVER a statistical gate, NEVER override WFOV/DSR/PBO/MinTRL, and NEVER directly trigger a trade. Treat their output as research notes with provenance — an input to *your* judgment, not a decision.

### §13.1 — dexter (private single-agent analyst tool)

Interactive single-agent analyst (TypeScript/Bun). Best for:
- **Idea sourcing & universe due-diligence** — its stock screener, filings reader (10-K/10-Q/8-K), fundamentals, ratios, insider/institutional activity.
- **Portfolio-review deliverables** — the DCF skill (intrinsic value) and write-memo skill (a PM-ready HTML investment memo: bear/base/bull, variant view, falsifiable thesis).
- **Provenance** — JSONL scratchpads record which data backed each claim.

Invoke: `bun start` (interactive), `/model` to switch provider. Needs `FINANCIAL_DATASETS_API_KEY` + an LLM key.

Use it at: **Stage A** (idea sourcing), **Stage B** (universe DD), and post-deployment **portfolio review** (memos on held names).

### §13.2 — TradingAgents (open-source: TauricResearch/TradingAgents)

Multi-agent LLM framework (Python/LangGraph). Produces a structured buy/sell/hold *decision* per ticker/date via an analyst → researcher (bull vs bear) → trader → risk → PM debate, tracking realized alpha vs SPY.

Invoke: `tradingagents` (interactive) or `TradingAgentsGraph().propagate(ticker, date)` (scriptable over a basket). Needs an LLM key + `ALPHA_VANTAGE_API_KEY`.

Use it at: **Stage A / portfolio review** — a multi-perspective narrative context on a name. Note it is explicitly non-deterministic and research-only; it does not connect to a broker.

**The trap.** "The agents said buy, and the DCF says undervalued, so I'll overweight it." No. That is a qualitative opinion. It may inform which idea you *pre-register* (§4); it never substitutes for §7–§10.

**Checklist.**
- [ ] Used as input to judgment, not as a gate?
- [ ] No WFOV/DSR/MinTRL gate overridden by an LLM opinion?
- [ ] No trade triggered directly from LLM output?

---

## §14 — Master checklists (start to finish)

### Pre-hoc (before a research session)
- [ ] Research regime identified: greenfield or live (§1)?
- [ ] Idea pre-registered with a falsification condition (§4)?
- [ ] (Live) cooldown elapsed; MinTRL respected (§10)?
- [ ] I know which §2 bias this idea is most exposed to.

### In-flight (during research)
- [ ] Data validated, coverage ≥ 95%, fresh (§5)?
- [ ] IS Sharpe ≤ ~1.5 alarm respected; universe sane (§6)?
- [ ] OOS validated; non-overlapping windows counted (§7)?
- [ ] Profile B+, excess Sharpe > 0, survives 2× PTC, seed-stable (§8)?
- [ ] Apples-to-apples vs deployed done before any swap (§9)?

### Post-hoc (before and after deploying)
- [ ] `revision_check.py` PASS; under trials/retirement limits (§10)?
- [ ] (Live) deployed via `run_revision.sh` with typed `DEPLOY` (§11)?
- [ ] Trials ledger updated (§10/§11)?
- [ ] Paper-traded; capital at 5% start; daily routine running (§12)?
- [ ] Result reproducible (seed, scaler, pinned window)?

---

## §15 — Never-dos (consolidated)

- **Never test before pre-registering** the hypothesis (§4, §2.8).
- **Never deploy on in-sample Sharpe** alone (§6, §2.3).
- **Never treat a high IS Sharpe as a win** — > 1.5 is an alarm (§6, §2.2).
- **Never use Profile A for deployment** — B is the minimum (§8, §2.5).
- **Never trust Monte Carlo p-values as your sample size** — count non-overlapping windows (§7, §2.1).
- **Never assume single-level transaction costs** — screen at 2–3× (§8, §2.4).
- **Never count 13 classifiers on the same features as 13 experiments** (§8, §2.6).
- **Never swap a live portfolio without an apples-to-apples comparison** (§9).
- **Never revise a live strategy on sub-MinTRL noise** (§10, §12).
- **Never edit `config.py` directly for a live change** — use `run_revision.sh` (§11).
- **Never bypass the trials gate** except a documented, audited operational hot-fix (§10, §11).
- **Never retrain on a schedule** — "schedule retraining is data-mining with extra steps" (OPERATIONS_MANUAL §7-A4).
- **Never let an LLM tool act as a gate or trigger a trade** (§13).
- **Never date-mine, seed-mine, universe-mine, or objective-mine** to find a passing result (DIAGNOSIS_PLAN §7).
- **Never accelerate capital scaling** on a favorable sub-MinTRL window (§12).

---

## §16 — Glossary and cross-references

Terms specific to this manual (for shared terms — Sentinel, Tier, Trial, DSR, MinTRL, PBO, HRP, Kill-switch — see `OPERATIONS_MANUAL.md` §11).

- **Research regime** — greenfield vs live; the ONLY sense of "regime" used in this manual and the only thing meant by "identify the regime". Greenfield = no capital deployed for this idea (explore freely); live = capital at risk (REVISION_POLICY binds). Determines which rulebook governs. See §1.1.
- **Market regime** — the state of the market (bull/bear, risk-on/risk-off, high/low volatility). A *property of the data*, surfaced by tools like `validate_portfolio_oos.py --mode rolling` ("regime-dependence detection"). **Do NOT confuse it with the research regime, and never treat it — by itself — as a reason to revise a live strategy** (that is market-timing on sub-MinTRL noise). Orthogonal to research regime: greenfield-in-a-bull and live-in-a-bear are both possible.
- **Degradation** — (IS Sharpe − OOS Sharpe) / IS Sharpe. A screening metric (§7); necessary but not sufficient for a live deploy.
- **Coefficient of variation (CV)** — std/mean of OOS Sharpe across periods; a fixed-weight stability measure (§7).
- **Excess (alpha) Sharpe** — model Sharpe minus buy-and-hold Sharpe; the honest metric (§8, §2.2).
- **Deflated Sharpe Ratio (DSR)** — Sharpe adjusted for the number of trials attempted; the live deploy gate at ≥ 0.5 (§10).
- **Embargo / purging** — leave a gap (here 2%) between train and test to prevent leakage (López de Prado) (§8).
- **Efficient frontier / target volatility / MIQP** — `portimization.py` outputs; integer-share, budget-aware optimization (§6).
- **Apples-to-apples comparison** — deployed vs candidate on the same OOS window; mandatory before any swap (§9).
- **Non-overlapping windows** — your true sample size; the antidote to the more-iterations illusion (§7, §2.1).
- **Profiles A/B/C** — ranking standards; B is the deployment minimum (§8).

**Cross-references:** `OPERATIONS_MANUAL.md` (run cadence, break-glass), `REVISION_POLICY.md` (binding gates, retirement), `revision_hypotheses.md` (pre-registration log), `PORTFOLIO_REVISION_DIAGNOSIS_PLAN.md` (diagnosis playbook), `WORKFLOW_GUIDE.md` / `MODEL_SELECTION_GUIDE.md` / `DATA_WORKFLOW.md` / `WFOV_V2_SUMMARY.md` (tool-level detail).

---

*End of Research Manual. This document teaches the discipline; you practice it. Agents help you debug and fix — they do not operate your pipeline for you.*
