# Disclaimers

**Read this before using anything in this repository.**

## 1. Educational and research purposes only

This codebase is published as a **portfolio showcase of engineering and
research methodology**. It is NOT investment advice, NOT a recommendation
to trade any instrument, and NOT an offer to manage money. Nothing here
should be construed as financial, investment, legal, or tax advice.

## 2. This repository is an incomplete, curated subset

This public repo is a **sanitized subset of a larger private system**. Be
aware of what is *not* here:

- **No market data.** No price files ship with the repo. You must source
  your own data (the code downloads from Yahoo Finance where possible, or
  from a broker/data vendor you are licensed to use).
- **No trained model weights.** All `.pkl` / `.keras` / `.h5` artifacts are
  excluded. You must train your own models (the training pipelines are
  included).
- **No crypto-trading module.** A Bybit crypto module exists in the private
  system; it is not part of this public repo.
- **No monitoring dashboard.** The portfolio-oversight dashboard is not
  included.
- **Some modules reference components that are not included** (ops shell
  scripts, private runbooks). Code may mention paths or files that do not
  exist in this subset. Where a subsystem is outlined in docs but absent in
  code, treat the docs as architecture notes.
- **Example configuration.** `execution/config.py` ships with a *placeholder
  example portfolio* (liquid ETFs), not any real allocation. All tickers,
  weights, and thresholds in it are illustrative.

## 3. No performance claims

**No returns, Sharpe ratios, or track-record figures are claimed anywhere in
this repository.** Any historical figures mentioned in prose describe the
development process, not an offerable track record. Past performance — of
any strategy, model, or backtest — does not guarantee future results.
Backtests are simulations and routinely overstate live results (costs,
slippage, financing, capacity, and regime change).

## 4. Live trading involves substantial risk

The execution layer talks to Interactive Brokers via their API. If you wire
it to a live account, you do so entirely at your own risk. Trading leveraged
instruments can result in losses exceeding your investment. The included
risk controls (kill switch, circuit breakers, preflight gates) reduce but do
not eliminate risk, and contain no guarantee of correctness. This project is
not affiliated with, endorsed by, or supported by Interactive Brokers.

## 5. Data-source terms

Users are responsible for complying with the terms of service of any data
vendor (Yahoo Finance, FRED, Interactive Brokers, etc.) they configure the
code to use.

## 6. Software "AS IS"

Provided under the MIT License, WITHOUT WARRANTY OF ANY KIND. See
[LICENSE](LICENSE).
