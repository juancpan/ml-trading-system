# TODO / Roadmap

Honest, unordered roadmap. This is a curated public subset (see
[DISCLAIMER.md](DISCLAIMER.md)); items marked *(private)* exist in the
private system and are candidates for porting.

## CI & packaging

- [ ] GitHub Actions CI: `pytest --collect-only`, run the unit tests that
      don't need broker connectivity, and `python -m compileall` on push.
- [ ] `pyproject.toml` packaging so `algos`/`execution` import cleanly
      without `sys.path` bootstrapping.
- [ ] Replace the remaining `sys.path.insert` patterns with proper package
      imports.

## Research methodology

- [ ] Purged/embargoed k-fold cross-validation (López de Prado) alongside
      walk-forward, for shorter histories.
- [ ] CPCV (combinatorial purged cross-validation) mode in
      `algos/wfov/`.
- [ ] Meta-labeling layer on top of the primary signal models.
- [ ] Publish a small sample parquet bundle so WFOV can be exercised
      without a data vendor.

## Docs & demos

- [ ] A `docs/` guide for the trials ledger and pre-registration workflow
      *(private, needs sanitization)*.
- [ ] Quarterly-review walkthrough *(private, heavily sanitized)*.
- [ ] Demo notebook: data → features → one model → WFOV, end to end on
      yfinance data.
- [ ] Architecture diagram rendered as image (current: ASCII in README).

## Execution layer

- [ ] Integration-test harness against an IBKR **paper** account
      (gate 0→3 flow from `run_region.sh`).
- [ ] Unit-test the Telegram alert path with a mock server instead of
      relying only on credential scrubbing.
- [ ] The position/portfolio state caches (`positions.pkl`,
      `account_values.pkl`) still assume a single-writer discipline;
      add file-locking for multi-process safety.

## Known incompleteness in this public subset

- [ ] Some `tests/` reference ops scripts that are not included
      (e.g. `test_sunday_maintenance.py`, `test_action_matrix.py`); they
      collect but are expected to fail at runtime here.
- [ ] `portfolio_oversight/` (monitoring dashboard) and the crypto module
      are documented in README but not shipped.
- [ ] A few execution modules write state files into their own directory
      at runtime; a central runtime-state directory would be cleaner.
