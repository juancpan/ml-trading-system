"""
CFTC Commitment of Traders data downloader.

COT reports are published weekly (Friday, 3:30 PM ET) with data as of Tuesday.
Publication lag: 3 business days (Tuesday data -> Friday release).

For USDJPY: Use CME Japanese Yen futures (contract code: 090741).
For EURUSD: Use CME Euro FX futures (contract code: 099741).

Requires: pip install cot-reports
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# COT publication lag: data is as-of Tuesday, published Friday = ~4 calendar days
COT_PUBLICATION_LAG_DAYS = 4

# CME contract codes for major currencies (CFTC legacy report)
COT_CONTRACT_CODES = {
    "JPY": "090741",
    "EUR": "099741",
    "GBP": "096742",
    "AUD": "232741",
    "CAD": "090741",
    "CHF": "092741",
    "NZD": "112741",
    "MXN": "095741",
}


def fetch_cot_net_positioning(
    currency: str = "JPY",
    start: str = "2020-01-01",
    end: str = None,
    cache_dir: str = None,
) -> Optional[pd.Series]:
    """
    Fetch COT net non-commercial positioning for a given currency.

    Net positioning = Non-commercial Long - Non-commercial Short.
    Positive = speculators net long the currency.

    Returns a weekly Series shifted forward by COT_PUBLICATION_LAG_DAYS
    for point-in-time correctness.
    """
    try:
        import cot_reports as cot
    except ImportError:
        logger.warning("cot-reports not installed. Run: pip install cot-reports")
        return None

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "external"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / f"cot_{currency.lower()}_net_pos.parquet"

    # Use cache if fresh (< 7 days old)
    if cache_file.exists():
        import time

        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < 7:
            try:
                df = pd.read_parquet(cache_file)
                series = df["net_positioning"]
                series.name = f"cot_{currency.lower()}_net"
                logger.info(
                    f"COT {currency}: loaded from cache ({len(series)} obs, "
                    f"age {age_days:.1f} days)"
                )
                return series
            except Exception as e:
                logger.warning(f"COT cache read failed: {e}, re-downloading")

    contract_code = COT_CONTRACT_CODES.get(currency.upper())
    if contract_code is None:
        logger.warning(f"No COT contract code for currency: {currency}")
        return None

    try:
        # Download legacy futures-only reports year by year
        all_dfs = []
        start_year = max(2017, pd.Timestamp(start).year - 1) if start else 2017
        end_year = pd.Timestamp.now().year

        for year in range(start_year, end_year + 1):
            try:
                yearly = cot.cot_year(year=year, cot_report_type="legacy_fut")
                if yearly is not None and not yearly.empty:
                    all_dfs.append(yearly)
            except Exception as e:
                logger.debug(f"COT year {year} download failed: {e}")
                continue

        if not all_dfs:
            logger.warning(f"No COT data downloaded for years {start_year}-{end_year}")
            return None

        df = pd.concat(all_dfs, ignore_index=True)

        # Filter for the specific contract
        mask = df["CFTC Contract Market Code"].astype(str).str.strip() == contract_code
        cot_data = df[mask].copy()

        if cot_data.empty:
            # Try matching by market name as fallback
            currency_names = {
                "JPY": "JAPANESE YEN",
                "EUR": "EURO FX",
                "GBP": "BRITISH POUND",
                "AUD": "AUSTRALIAN DOLLAR",
                "CHF": "SWISS FRANC",
                "CAD": "CANADIAN DOLLAR",
                "NZD": "NEW ZEALAND DOLLAR",
                "MXN": "MEXICAN PESO",
            }
            name_pattern = currency_names.get(currency.upper(), currency.upper())
            mask = df["Market and Exchange Names"].str.contains(
                name_pattern, case=False, na=False
            )
            cot_data = df[mask].copy()

        if cot_data.empty:
            logger.warning(f"No COT data found for {currency} (code {contract_code})")
            return None

        # Parse dates and compute net positioning
        date_col = None
        for candidate in [
            "As of Date in Form YYYY-MM-DD",
            "As_of_Date_In_Form_YYYY-MM-DD",
            "Report_Date_as_YYYY-MM-DD",
        ]:
            if candidate in cot_data.columns:
                date_col = candidate
                break

        if date_col is None:
            # Try first column that looks like a date
            for col in cot_data.columns:
                if "date" in col.lower():
                    date_col = col
                    break

        if date_col is None:
            logger.warning("Could not find date column in COT data")
            return None

        cot_data["date"] = pd.to_datetime(cot_data[date_col])
        cot_data = cot_data.sort_values("date").set_index("date")

        # Find the long/short columns
        long_col = None
        short_col = None
        for col in cot_data.columns:
            col_lower = col.lower()
            if (
                "noncommercial" in col_lower
                and "long" in col_lower
                and "spread" not in col_lower
            ):
                long_col = col
            elif (
                "noncommercial" in col_lower
                and "short" in col_lower
                and "spread" not in col_lower
            ):
                short_col = col

        if long_col is None or short_col is None:
            logger.warning(
                f"Could not find non-commercial long/short columns. "
                f"Available: {[c for c in cot_data.columns if 'commercial' in c.lower()]}"
            )
            return None

        cot_data["net_positioning"] = pd.to_numeric(
            cot_data[long_col], errors="coerce"
        ) - pd.to_numeric(cot_data[short_col], errors="coerce")

        result = cot_data[["net_positioning"]].dropna().copy()

        # Shift forward by publication lag (point-in-time)
        result.index = result.index + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS)

        # Cache
        try:
            result.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Could not cache COT data: {e}")

        series = result["net_positioning"]
        series.name = f"cot_{currency.lower()}_net"

        logger.info(
            f"COT {currency}: {len(series)} weekly observations, "
            f"{series.index[0].strftime('%Y-%m-%d')} to "
            f"{series.index[-1].strftime('%Y-%m-%d')}"
        )
        return series

    except Exception as e:
        logger.error(f"COT download failed for {currency}: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def is_cot_ticker(ticker: str) -> bool:
    """Check if ticker uses COT: prefix."""
    return ticker.upper().startswith("COT:")
