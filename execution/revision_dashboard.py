"""Revision Health dashboard (Phase 1.4 of the Revision Protocol).

Generates a self-contained ``revision_dashboard.html`` summarising:

* Latest kill-switch status (sentinel presence + last decision).
* Equity curve vs. backtest percentile bands (Phase 2.1 fills the bands;
  for now we show the curve alone).
* Five attribution numbers, last 30 days.
* Per-model 20-day rolling hit-rate.
* Shadow check divergences in the last week.
* Trials budget consumed (placeholder until Phase 3).
* Active sentinels.

Intentionally a single-file HTML output — no Flask, no Jinja, no
JavaScript dependencies beyond plain HTML + a tiny inline canvas-free
chart rendered as an SVG. Good enough for a personal operator dashboard;
much simpler to maintain than a full web app.

Refresh by re-running the script (cron, every 5 minutes during trading
hours).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sqlite3
from datetime import datetime, timezone, date as Date
from pathlib import Path
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent

DEFAULT_OUT = _THIS_DIR / "revision_dashboard.html"
DEFAULT_ATTR_DB = _THIS_DIR / "attribution.db"
DEFAULT_EQUITY = _THIS_DIR / "equity_history.parquet"
DEFAULT_SHADOW = _THIS_DIR / "shadow_history.parquet"
HARD_KILL = _THIS_DIR / "KILL_SWITCH_ACTIVE"
SOFT_HALT = _THIS_DIR / "SOFT_HALT_ACTIVE"
DAILY_MOVE = _THIS_DIR / "DAILY_MOVE_ACTIVE"


_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h2 { margin-top: 1.8em; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
  th { background: #f4f4f4; }
  tr:nth-child(even) td { background: #fafafa; }
  .ok { color: #2a7; font-weight: 600; }
  .warn { color: #d80; font-weight: 600; }
  .crit { color: #c33; font-weight: 700; }
  .muted { color: #888; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 0.85em; font-weight: 600; }
  .badge-ok { background: #d3f5e0; color: #1a7039; }
  .badge-warn { background: #fff1c1; color: #8a5a00; }
  .badge-crit { background: #ffd2d2; color: #a01818; }
  .num { font-variant-numeric: tabular-nums; }
  .svgwrap { max-width: 900px; overflow-x: auto; }
  pre { background: #f7f7f7; padding: 10px; overflow-x: auto;
        font-size: 0.85em; border-radius: 4px; }
</style>
"""


def _read_sentinel(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"reason": "(unparseable sentinel)", "details": {}}


def _sentinel_section() -> str:
    rows: list[str] = []
    for label, p in [("HARD KILL", HARD_KILL), ("SOFT HALT", SOFT_HALT),
                     ("DAILY MOVE", DAILY_MOVE)]:
        payload = _read_sentinel(p)
        if payload is None:
            rows.append(f"<tr><td>{label}</td><td><span class='badge badge-ok'>clear</span></td>"
                        "<td class='muted'>—</td><td class='muted'>—</td></tr>")
        else:
            sev = "badge-crit" if label == "HARD KILL" else "badge-warn"
            ts = payload.get("written_at", "?")
            reason = html.escape(str(payload.get("reason", "?")))
            rows.append(
                f"<tr><td>{label}</td>"
                f"<td><span class='badge {sev}'>ACTIVE</span></td>"
                f"<td class='muted'>{ts}</td>"
                f"<td>{reason}</td></tr>"
            )
    return (
        "<h2>Kill-switch sentinels</h2>"
        "<table><thead><tr><th>Tier</th><th>Status</th><th>Written</th><th>Reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _equity_svg(path: Optional[Path] = None, *, days: int = 30) -> str:
    if path is None:
        path = DEFAULT_EQUITY
    if not path.exists():
        return "<p class='muted'>No equity_history.parquet yet.</p>"
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return f"<p class='warn'>Could not read equity_history: {exc}</p>"
    if df.empty:
        return "<p class='muted'>equity_history is empty.</p>"

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff].sort_values("timestamp")
    if df.empty:
        return "<p class='muted'>No recent equity points.</p>"

    # Aggregate to one point per day per region (last value).
    df["date"] = df["timestamp"].dt.date
    agg = df.groupby(["date", "region"], as_index=False)["nav_usd"].last()
    agg = agg.sort_values("date")
    if agg.empty:
        return "<p class='muted'>No data.</p>"

    nav = agg["nav_usd"].astype(float).tolist()
    dates = [d.isoformat() for d in agg["date"].tolist()]
    n = len(nav)
    w, h_chart = 800, 200
    pad = 30
    if n < 2 or min(nav) == max(nav):
        return f"<p class='muted'>Only {n} equity points; need at least 2 to plot.</p>"
    xs = [pad + i * (w - 2 * pad) / (n - 1) for i in range(n)]
    ymin, ymax = min(nav), max(nav)
    ys = [pad + (h_chart - 2 * pad) * (1 - (v - ymin) / (ymax - ymin)) for v in nav]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    svg = (
        f"<svg width='{w}' height='{h_chart}' style='border:1px solid #ddd; background:#fff'>"
        f"<polyline fill='none' stroke='#2a6' stroke-width='1.5' points='{pts}' />"
        f"<text x='{pad}' y='{pad-10}' font-size='10' fill='#888'>"
        f"NAV {min(nav):,.0f} → {max(nav):,.0f}, last 30d</text>"
        "</svg>"
    )
    return f"<div class='svgwrap'>{svg}</div>"


def _attribution_table(db_path: Optional[Path] = None, *, days: int = 30) -> str:
    if db_path is None:
        db_path = DEFAULT_ATTR_DB
    if not db_path.exists():
        return "<p class='muted'>attribution.db not present (have you run attribution.py?).</p>"
    # True calendar-date window ("last N days"), not a row LIMIT. A row LIMIT
    # would silently show the most recent N *rows* regardless of how old they
    # are, which masks staleness (the original bug masked a 3-week-old DB).
    cutoff = (datetime.now(timezone.utc).date() - pd.Timedelta(days=days)).isoformat()
    with sqlite3.connect(str(db_path)) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM attribution_daily WHERE date >= ? ORDER BY date DESC",
                conn, params=(cutoff,),
            )
        except Exception as exc:
            return f"<p class='warn'>Could not read attribution_daily: {exc}</p>"
    if df.empty:
        return (
            f"<p class='warn'>No attribution rows in the last {days} days "
            f"(since {cutoff}). Is attribution.py running? Check the cron logs.</p>"
        )

    cols = [
        ("date", "date"),
        ("region", "region"),
        ("total_pnl", "region_nav_delta"),
        ("execution_drag", "execution_drag"),
        ("signal_contribution", "signal_contribution"),
        ("weighting_contribution", "weighting_contribution"),
        ("sizing_contribution", "sizing_contribution"),
    ]
    pending_cols = {"signal_contribution", "weighting_contribution", "sizing_contribution"}
    header = "".join(f"<th>{label}</th>" for _, label in cols)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c, _label in cols:
            v = r[c]
            if pd.isna(v):
                if c in pending_cols:
                    tds.append("<td class='muted'>Phase 1.4 pending</td>")
                else:
                    tds.append("<td class='muted'>—</td>")
            elif isinstance(v, float):
                sign_class = "" if v == 0 else ("ok" if v > 0 else "crit")
                tds.append(f"<td class='num {sign_class}'>{v:+.2f}</td>")
            else:
                tds.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (
        "<p class='warn'>Attribution is Phase 1.4-partial: region_nav_delta is "
        "a region-window NAV delta, not TWS daily P&amp;L. Counterfactual "
        "columns are pending until decision_price and per-ticker return "
        "plumbing lands.</p>"
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _hit_rate_table(db_path: Optional[Path] = None, *, latest_only: bool = True) -> str:
    if db_path is None:
        db_path = DEFAULT_ATTR_DB
    if not db_path.exists():
        return "<p class='muted'>attribution.db not present.</p>"
    with sqlite3.connect(str(db_path)) as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM model_hit_rate ORDER BY date DESC", conn)
        except Exception as exc:
            return f"<p class='warn'>Could not read model_hit_rate: {exc}</p>"
    if df.empty:
        return "<p class='muted'>No hit-rate rows yet.</p>"

    # Staleness banner: if the newest hit-rate row is more than a few days old,
    # the producer (attribution.py) has likely stopped running. Make it loud.
    banner = ""
    try:
        latest = max(pd.to_datetime(df["date"]).dt.date)
        age = (datetime.now(timezone.utc).date() - latest).days
        if age > 4:
            banner = (
                f"<p class='crit'>STALE: newest hit-rate row is {latest} "
                f"({age} days old). attribution.py is probably not running.</p>"
            )
    except Exception:
        pass

    if latest_only:
        df = df.sort_values(["ticker", "model_type", "date"], ascending=[True, True, False])
        df = df.drop_duplicates(subset=["ticker", "model_type", "window_days"])

    has_n = "total" in df.columns
    has_unresolved = "unresolved" in df.columns
    rows = []
    for _, r in df.iterrows():
        hits = int(r["hits"])
        misses = int(r["misses"])
        n = int(r["total"]) if (has_n and pd.notna(r.get("total"))) else hits + misses
        hr = float(r["hit_rate"]) if pd.notna(r["hit_rate"]) else None
        cls = ""
        if hr is not None:
            if hr >= 0.55:
                cls = "ok"
            elif hr <= 0.45:
                cls = "crit"
            else:
                cls = "warn"
        # Build the rate cell separately. Doing this inline as a ternary inside
        # the implicitly-concatenated f-strings below silently truncates the row
        # (precedence makes the whole row the ternary's true-branch), so the
        # closing </td>, the date cell and </tr> get dropped, or the entire row
        # collapses to "—" when hr is None.
        if hr is not None:
            hr_text = f"{hr:.3f}"
        elif n > 0:
            # NaN rate with samples present == below the min-sample threshold.
            hr_text = f"<span class='muted'>n&lt;min</span>"
            cls = "muted"
        else:
            hr_text = "—"
        n_cell = (f"<td class='num'>{n}</td>" if has_n else "")
        unres = int(r["unresolved"]) if (has_unresolved and pd.notna(r.get("unresolved"))) else 0
        unres_note = (f" <span class='muted'>(+{unres} pending)</span>"
                      if unres > 0 else "")
        rows.append(
            f"<tr><td>{html.escape(str(r['ticker']))}</td>"
            f"<td>{html.escape(str(r['model_type']))}</td>"
            f"<td class='num'>{hits}</td>"
            f"<td class='num'>{misses}</td>"
            f"{n_cell}"
            f"<td class='num {cls}'>{hr_text}</td>"
            f"<td class='muted'>{html.escape(str(r['date']))}{unres_note}</td></tr>"
        )
    n_header = "<th>N</th>" if has_n else ""
    return (
        banner
        + "<p class='muted'>Hit-rate suppressed (shown as n&lt;min) below "
        "the minimum sample size; a model with only 1-2 scored signals is not "
        "statistically meaningful.</p>"
        + "<table><thead><tr><th>Ticker</th><th>Model</th><th>Hits</th>"
        f"<th>Misses</th>{n_header}<th>Hit rate</th><th>As of</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _shadow_section(path: Optional[Path] = None, *, days: int = 7) -> str:
    if path is None:
        path = DEFAULT_SHADOW
    if not path.exists():
        return "<p class='muted'>shadow_history.parquet not present.</p>"
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return f"<p class='warn'>Could not read shadow_history: {exc}</p>"
    if df.empty:
        return "<p class='muted'>No shadow rows yet.</p>"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff]
    divergent = df[df["divergence_flag"]] if "divergence_flag" in df.columns else df
    if divergent.empty:
        return "<p class='ok'>No shadow divergences in the last week.</p>"
    rows = []
    for _, r in divergent.iterrows():
        rows.append(
            f"<tr><td>{html.escape(str(r['ticker']))}</td>"
            f"<td>{html.escape(str(r['model_type']))}</td>"
            f"<td class='num'>{int(r['live_signal'])}</td>"
            f"<td class='num'>{int(r['shadow_signal'])}</td>"
            f"<td>{html.escape(str(r.get('divergence_reason', '')))}</td>"
            f"<td class='muted'>{r['timestamp'].isoformat()}</td></tr>"
        )
    return (
        f"<p class='warn'>{len(divergent)} divergences in last {days}d</p>"
        "<table><thead><tr><th>Ticker</th><th>Model</th><th>Live</th>"
        "<th>Shadow</th><th>Reason</th><th>When</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render(out_path: Path = DEFAULT_OUT) -> Path:
    parts: list[str] = []
    parts.append("<!DOCTYPE html><html><head>")
    parts.append("<meta charset='utf-8'><title>Revision Health</title>")
    parts.append(_CSS)
    parts.append("</head><body>")
    parts.append(f"<h1>Revision Health Dashboard</h1>")
    parts.append(
        f"<p class='muted'>Generated {datetime.now(timezone.utc).isoformat()} UTC. "
        "Refresh: re-run <code>revision_dashboard.py</code>.</p>"
    )

    parts.append(_sentinel_section())

    parts.append("<h2>Equity curve — last 30d</h2>")
    parts.append(_equity_svg())

    parts.append("<h2>Attribution — last 30d</h2>")
    parts.append(_attribution_table())

    parts.append("<h2>Per-model rolling hit-rate</h2>")
    parts.append(_hit_rate_table())

    parts.append("<h2>Shadow check divergences — last 7d</h2>")
    parts.append(_shadow_section())

    parts.append("<h2>Trials budget</h2>")
    parts.append("<p class='muted'>Phase 3 fills this in.</p>")

    parts.append("</body></html>")

    out_path.write_text("".join(parts))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Revision Health dashboard")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    p = render(args.out)
    print(f"Dashboard written: {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
