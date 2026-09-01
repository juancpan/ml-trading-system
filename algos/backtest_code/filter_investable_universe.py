"""
Investable Universe Filter - Pre-optimization lot size feasibility check

Filters out tickers whose minimum lot cost exceeds the minimum position value,
ensuring portfolio optimization can allocate to all remaining tickers.

Workflow Position:
    validate_portfolio_oos.py -> filter_investable_universe.py -> portfolio_exploration_global.py

Key Formula:
    min_position_value = budget * min_weight_pct
    price_usd = price_native / exchange_rate  (for USDJPY, USDHKD, etc.)
    price_usd = price_native * exchange_rate  (for GBPUSD)
    min_lot_cost_usd = price_usd * lot_size

    Keep if: min_lot_cost_usd <= min_position_value
    Drop if: min_lot_cost_usd > min_position_value

Usage:
    # Basic usage (US stocks only)
    python filter_investable_universe.py \\
        --csv data.csv --budget 50000 --min-weight 0.05

    # Multi-currency with custom lot sizes
    python filter_investable_universe.py \\
        --csv data.csv --budget 50000 --min-weight 0.05 \\
        --exchange-rates USDJPY=158 GBPUSD=1.27 USDHKD=7.8 \\
        --lot-sizes 1277.HK=2000

    # Auto-calculate min weight (equal weight assumption)
    python filter_investable_universe.py \\
        --csv data.csv --budget 30000 --auto-weight

    # Automatic mode (no prompts, for scripting)
    python filter_investable_universe.py \\
        --csv data.csv --budget 50000 --min-weight 0.05 --auto

Exchange Rate Format:
    USDJPY=158    -> 1 USD = 158 JPY  (divide JPY price by rate to get USD)
    GBPUSD=1.27   -> 1 GBP = 1.27 USD (multiply GBP price by rate to get USD)
    USDHKD=7.8    -> 1 USD = 7.8 HKD  (divide HKD price by rate to get USD)

Author: Algorithmic Trading System
Date: 2026-01
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "execution"))
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf

from exchange_manager import ExchangeManager
from portimization import get_latest_prices_from_csv, BASE_LOG_DIR

# Generate timestamp for outputs
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LSE_CURRENCY_OVERRIDES_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "market_data"
    / "lse_currency_overrides.json"
)

# Reverse mapping: Column names in CSV → yfinance tickers
# Built from yfinance_downloader_v5.py tickers_map
COLUMN_TO_YFINANCE_MAP = {
    # Japanese stocks (.T suffix)
    "Marubeni": "8002.T",
    "ITOCHU": "8001.T",
    "MITSUBISHI": "8058.T",
    "MITSUI": "8031.T",
    "SUMITOMO": "8053.T",
    "Shin_Etsu_Chemical": "4063.T",
    "Komatsu": "6301.T",
    "Daiichi_Kosan": "6367.T",
    "Renesas_Electronics": "6723.T",
    "Sony_Group": "6758.T",
    "Keyence": "6861.T",
    "Fanuc": "6954.T",
    "Murata_Manufacturing": "6981.T",
    "Tokyo_Electron": "8035.T",
    "Fast_Retailing": "9983.T",
    # LSE stocks (.L suffix) - mostly GBX; non-GBX tickers use overrides
    "3I": "III.L",
    "BARCLAYS": "BARC.L",
    "Rolls_Royce": "RR.L",
    "Glencore": "GLEN.L",
    "Rio_Tinto": "RIO.L",
    "London_Stock_Exchange_Group": "LSEG.L",
    "AstraZeneca": "AZN.L",
    "Unilever": "ULVR.L",
    "Anglo American": "AAL.L",
    "Vodafone": "VOD.L",
    "Aviva": "AV.L",
    "Standard_Chartered": "STAN.L",
    "BP": "BP.L",
    "HSBC_LON": "HSBA.L",
    "BHP": "BHP.L",
    # Hong Kong stocks (.HK suffix)
    # Column names from yfinance_downloader_v5.py -> yfinance tickers
    "TENCENT": "0700.HK",
    "PETROCHINA": "0857.HK",
    "ICBC": "1398.HK",
    "中国移动": "0941.HK",
    "SHENHUA": "1088.HK",
    "CNOOC": "0883.HK",
    "Zijin_Mining": "2899.HK",
    "民生银行": "1988.HK",
    "力量发展": "1277.HK",
    "中国银行": "3988.HK",
    "中国中车": "1766.HK",
    "中远海控": "1919.HK",
    "交通银行": "3328.HK",
    "新华保险": "1336.HK",
    "CGN_Mining": "1164.HK",
    "ABC_HK": "1288.HK",
    # Additional HK stocks with different column names
    "HSBC_HK": "0005.HK",  # HSBC Hong Kong
    "农行_HK": "1288.HK",  # Agricultural Bank of China HK
    "ALIBABA": "9988.HK",  # Alibaba HK
    "MEITUAN": "3690.HK",  # Meituan
    "中广核电力": "1816.HK",  # CGN Power
    "光大银行": "6818.HK",  # Everbright Bank
    "邮储银行": "1658.HK",  # Postal Savings Bank
    "中国铁塔": "0788.HK",  # China Tower
    "中金公司": "3908.HK",  # CICC
    "Autohome": "2518.HK",  # Autohome
    "Guotai": "2611.HK",  # Guotai Junan
    "Fuyao_Boli": "3606.HK",  # Fuyao Glass
    "CGN_Power": "1816.HK",  # CGN Power (alternative name)
    # German stocks (.DE suffix)
    "BMW": "BMW.DE",
    "BASF": "BAS.DE",
    "Bayer": "BAYN.DE",
    "Deutsche_Bank": "DBK.DE",
    "Allianz": "ALV.DE",
    "Adidas": "ADS.DE",
    "Continental_AG": "CON.DE",
    "Infineon_Technologies": "IFX.DE",
    "Siemens": "SIE.DE",
    "SAP": "SAP.DE",
    "Mercedes_Benz": "MBG.DE",
    "Volkswagen": "VOW3.DE",
    # French stocks (.PA suffix)
    "Airbus": "AIR.PA",
    "BNP_Paribas": "BNP.PA",
    "L_Oreal": "OR.PA",
    "LVMH": "MC.PA",
    "Hermes": "RMS.PA",
    "AXA_PA": "CS.PA",
    "Safran": "SAF.PA",
    "Sanofi": "SAN.PA",
    "Societe_Generale": "GLE.PA",
    "TotalEnergies": "TTE.PA",
    "Veolia_Environnement": "VIE.PA",
    "Schneider_Electric": "SU.PA",
    "Bureau_Veritas": "4BV.F",
    # Dutch stocks (.AS suffix)
    "Shell": "SHELL.AS",
    "Aegon": "AGN.AS",
    # Swiss stocks (.SW suffix)
    "Nestle": "NESN.SW",
    "Givaudan": "GIVN.SW",
    "Roche": "ROG.SW",
    "Zurich_Insurance": "ZURN.SW",
    # Singapore stocks (.SI suffix)
    "DBS_Group_Holdings": "D05.SI",
    "Singapore_Telecom": "Z74.SI",
    # Canadian stocks (.TO suffix)
    "Fairfax_Financial": "FFH.TO",
    "CCO": "CCO.TO",
    # Australian stocks (.AX suffix)
    "ASX_Limited": "ASX.AX",
    "Paladin_Energy": "PDN.AX",
    "Silex_Systems": "SLX.AX",
    # Other international
    "Chunghwa_Telecom": "2412.TW",
    "Maersk": "MAERSK-B.CO",
    "CEZ_Group": "CEZ.F",
    "Fortum": "FORTUM.HE",
    "Aristocrat_Leisure": "AC8.F",
    "Industria de Diseño Textil": "ITX.MC",
    "Bank_Central_Asia": "BBCA.JK",
    "Bank_Rakyat_Indonesia": "BBRI.JK",
    "Astra_International": "ASII.JK",
    "Airports_of_Thailand": "AOT.AAA",
    "Hoa_Phap_Group": "HPG.VN",
    "Tata_Steel_India": "TTST.IL",
    "Reliance_Industries_India": "RIGD.IL",
    # US stocks with different names
    "3M_Company": "MMM",
    "TOYOTA": "TM",
    "HSBC_US": "HSBC",
    "Ituran": "ITRN",
    "Jefferies": "JEF",
    "Jefferies_Financial_Group": "JEF",
    "Goldman_Sachs": "GS",
    "Novartis": "NVS",
    "enCore": "EU",
    "Lumen_Technologies": "LUMN",
    "Netflix": "NFLX",
}


def normalize_hk_ticker(ticker: str) -> str:
    """
    Normalize HK stock ticker to Yahoo Finance 4-digit format.

    Different data providers use different formats:
    - HKEX/TradingView: 5 (or 5.HK) - no padding
    - Yahoo Finance: 0005.HK - 4-digit with leading zeros

    This function converts any HK ticker format to Yahoo Finance format.

    Parameters:
    -----------
    ticker : str
        HK ticker in any format (e.g., '5', '5.HK', '0005', '0005.HK', '00005')

    Returns:
    --------
    str
        Yahoo Finance format (e.g., '0005.HK')

    Examples:
    ---------
    >>> normalize_hk_ticker('5.HK')
    '0005.HK'
    >>> normalize_hk_ticker('0005.HK')
    '0005.HK'
    >>> normalize_hk_ticker('00005')
    '0005.HK'
    >>> normalize_hk_ticker('700.HK')
    '0700.HK'
    >>> normalize_hk_ticker('1277.HK')
    '1277.HK'
    """
    # Remove .HK suffix if present
    if ticker.endswith(".HK"):
        code = ticker[:-3]
    else:
        code = ticker

    # Strip leading zeros and convert to int, then format as 4 digits
    try:
        code_int = int(code)
        # Yahoo Finance uses 4-digit format (0001.HK to 9999.HK)
        return f"{code_int:04d}.HK"
    except ValueError:
        # Not a numeric code, return as-is with .HK suffix
        return ticker if ticker.endswith(".HK") else f"{ticker}.HK"


def resolve_ticker_name(column_name: str) -> str:
    """
    Resolve a CSV column name to its yfinance ticker format.

    If the column is already in yfinance format (has exchange suffix or is a known US ticker),
    return as-is. Otherwise, look up in the reverse mapping.

    Parameters:
    -----------
    column_name : str
        Column name from CSV (e.g., 'Marubeni', '8002.T', 'NVDA')

    Returns:
    --------
    str
        yfinance ticker format (e.g., '8002.T', 'III.L', 'NVDA')
    """
    # Check if already in yfinance format with exchange suffix
    exchange_suffixes = [
        ".T",
        ".L",
        ".HK",
        ".DE",
        ".PA",
        ".AS",
        ".SW",
        ".SI",
        ".TO",
        ".AX",
        ".TW",
        ".CO",
        ".HE",
        ".F",
        ".MC",
        ".JK",
        ".AAA",
        ".VN",
        ".IL",
        ".NS",
        ".BO",
        ".TA",
        ".OL",
        ".BD",
        ".ST",
        ".SR",
        ".KL",
        ".PR",
        ".AE",
        ".MI",
        ".RO",
        ".WA",
        ".BR",
        ".TL",
        ".VS",
        ".VI",
    ]
    for suffix in exchange_suffixes:
        if column_name.endswith(suffix):
            # Normalize HK tickers to Yahoo Finance 4-digit format
            # Handles: 5.HK → 0005.HK, 700.HK → 0700.HK
            if suffix == ".HK":
                return normalize_hk_ticker(column_name)
            return column_name

    # Check reverse mapping
    if column_name in COLUMN_TO_YFINANCE_MAP:
        return COLUMN_TO_YFINANCE_MAP[column_name]

    # Assume it's already a valid US ticker
    return column_name


# Configure logging
LOG_FILE_PATH = os.path.join(
    BASE_LOG_DIR, f"filter_investable_universe_{TIMESTAMP}.log"
)


def setup_logging():
    """Configure logging to both console and file."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_lse_currency_overrides(
    override_file: Optional[str] = None,
) -> Dict[str, str]:
    """Load LSE per-ticker currency overrides for non-GBX listings."""
    file_path = Path(override_file) if override_file else LSE_CURRENCY_OVERRIDES_PATH

    if not file_path.exists():
        logging.info(f"LSE currency override file not found: {file_path}")
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load LSE currency overrides from {file_path}: {e}")
        return {}

    overrides = {}
    for ticker, currency in payload.items():
        if ticker.startswith("_"):
            continue
        if not isinstance(ticker, str) or not isinstance(currency, str):
            continue

        normalized_ticker = ticker.upper().strip()
        normalized_currency = currency.upper().strip()
        if normalized_ticker.endswith(".L"):
            overrides[normalized_ticker] = normalized_currency

    logging.info(f"Loaded {len(overrides)} LSE currency overrides from {file_path}")
    return overrides


def get_ticker_currency(
    ticker: str,
    exchange_mgr,
    lse_currency_overrides: Optional[Dict[str, str]] = None,
) -> str:
    """Resolve ticker currency with per-symbol LSE overrides."""
    normalized_ticker = ticker.upper().strip()
    if normalized_ticker.endswith(".L") and lse_currency_overrides:
        override_currency = lse_currency_overrides.get(normalized_ticker)
        if override_currency:
            return override_currency

    return exchange_mgr.get_currency(ticker)


def is_lse_pence_pricing(
    ticker: str, lse_currency_overrides: Optional[Dict[str, str]] = None
) -> bool:
    """Return True when an LSE ticker should be interpreted as GBX pence."""
    normalized_ticker = ticker.upper().strip()
    return normalized_ticker.endswith(".L") and normalized_ticker not in (
        lse_currency_overrides or {}
    )


def fetch_hkex_lot_sizes(allow_online_fetch: bool = True) -> Dict[str, int]:
    """
    Fetch lot sizes from HKEX official Excel file with 7-day caching.

    Downloads from: https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx

    Returns:
    --------
    Dict[str, int]
        Mapping of yfinance ticker (e.g., '1277.HK') to board lot size
    """
    HKEX_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
    CACHE_FILE = os.path.join(BASE_LOG_DIR, "hkex_lot_sizes_cache.json")

    # Check cache first (valid for 7 days)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            cache_time = datetime.fromisoformat(cache["timestamp"])
            if datetime.now() - cache_time < timedelta(days=7):
                logging.info(
                    f"  Using cached HKEX lot sizes ({len(cache['lot_sizes'])} stocks)"
                )
                return cache["lot_sizes"]
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # Cache corrupted, re-download

    if not allow_online_fetch:
        logging.info("  Online HKEX fetch disabled; using cache only")
        return {}

    # Download and parse
    logging.info("  Fetching HKEX lot sizes from official source...")
    try:
        # HKEX Excel structure:
        # Row 0: "List of Securities" (title)
        # Row 1: "Updated as at..." (date)
        # Row 2: Column headers ("Stock Code", "Board Lot", etc.)
        # Row 3+: Data
        df = pd.read_excel(HKEX_URL, header=2)

        # Column names are: 'Stock Code', 'Board Lot', etc.
        lot_col = "Board Lot"
        code_col = "Stock Code"

        if lot_col not in df.columns or code_col not in df.columns:
            logging.warning(
                f"  Could not find required columns. Available: {df.columns.tolist()}"
            )
            return {}

        # Clean Board Lot column (remove commas, convert to int)
        def parse_lot(x):
            try:
                return int(re.sub(r"[,\s]", "", str(x)))
            except (ValueError, TypeError):
                return 0

        # Convert to yfinance format (00005 → 0005.HK, 01277 → 1277.HK)
        lot_sizes = {}
        for _, row in df.iterrows():
            try:
                code = str(row[code_col]).strip()
                lot = parse_lot(row[lot_col])
                if code and lot > 0:
                    # Convert HKEX 5-digit format to yfinance 4-digit format
                    # HKEX uses 00005, 00001, 00700; Yahoo Finance uses 0005.HK, 0001.HK, 0700.HK
                    yf_code = normalize_hk_ticker(code)
                    lot_sizes[yf_code] = lot
            except (ValueError, TypeError):
                continue

        # Save to cache
        with open(CACHE_FILE, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "lot_sizes": lot_sizes}, f
            )

        logging.info(f"  Fetched {len(lot_sizes)} HKEX lot sizes")
        return lot_sizes

    except Exception as e:
        logging.warning(f"  Failed to fetch HKEX lot sizes: {e}")
        # Try to use stale cache if available
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    cache = json.load(f)
                logging.info(f"  Using stale cache ({len(cache['lot_sizes'])} stocks)")
                return cache["lot_sizes"]
            except:
                pass
        return {}


def fetch_exchange_rates(required_rates: List[str]) -> Dict[str, float]:
    """
    Fetch current exchange rates from yfinance.

    Parameters:
    -----------
    required_rates : List[str]
        List of rate keys needed, e.g., ['USDJPY', 'GBPUSD', 'USDHKD']

    Returns:
    --------
    Dict[str, float]
        Exchange rates, e.g., {'USDJPY': 157.5, 'GBPUSD': 1.27}
    """
    if not required_rates:
        return {}

    rates = {}
    failed = []

    # Map our rate keys to yfinance symbols
    # USDJPY -> USDJPY=X (how many JPY per 1 USD)
    # GBPUSD -> GBPUSD=X (how many USD per 1 GBP)
    yf_symbols = []
    rate_key_to_symbol = {}

    for rate_key in required_rates:
        yf_symbol = f"{rate_key}=X"
        yf_symbols.append(yf_symbol)
        rate_key_to_symbol[rate_key] = yf_symbol

    logging.info(f"Fetching exchange rates: {required_rates}")

    try:
        # Download rates with 5d period to handle weekends/holidays
        data = yf.download(yf_symbols, period="5d", progress=False, auto_adjust=True)

        if data.empty:
            logging.warning("  No exchange rate data returned from yfinance")
            return {}

        # yfinance may return MultiIndex columns even for single tickers
        # Check if columns are MultiIndex
        is_multiindex = isinstance(data.columns, pd.MultiIndex)

        for rate_key, yf_symbol in rate_key_to_symbol.items():
            try:
                # Extract the Close price series depending on column structure
                if is_multiindex:
                    # MultiIndex columns: ('Close', 'USDHKD=X')
                    if ("Close", yf_symbol) in data.columns:
                        rate_series = data[("Close", yf_symbol)].dropna()
                    else:
                        failed.append(rate_key)
                        continue
                else:
                    # Simple columns: 'Close' or just the symbol
                    if "Close" in data.columns:
                        rate_series = data["Close"].dropna()
                    elif yf_symbol in data.columns:
                        rate_series = data[yf_symbol].dropna()
                    else:
                        failed.append(rate_key)
                        continue

                # Get last non-NaN value (handles weekends/holidays)
                if not rate_series.empty:
                    # Ensure we get a scalar value, not a Series
                    rate_value = rate_series.iloc[-1]
                    if hasattr(rate_value, "item"):
                        rate = rate_value.item()  # Convert numpy scalar
                    elif hasattr(rate_value, "values"):
                        rate = float(rate_value.values[0])  # Handle Series
                    else:
                        rate = float(rate_value)

                    if rate > 0:
                        rates[rate_key] = float(rate)
                        logging.info(f"  {rate_key} = {rate:.4f}")
                    else:
                        failed.append(rate_key)
                else:
                    failed.append(rate_key)
            except Exception as e:
                logging.debug(f"  Failed to get {rate_key}: {e}")
                failed.append(rate_key)

        if failed:
            logging.warning(f"  Could not fetch rates for: {failed}")

        return rates

    except Exception as e:
        logging.warning(f"  Failed to fetch exchange rates: {e}")
        return {}


def fetch_exchange_rates_from_store(
    required_rates: List[str], end_date: Optional[str] = None
) -> Dict[str, float]:
    """Fetch exchange rates from local parquet store.

    Parameters:
    -----------
    required_rates : List[str]
        Rate keys like ['USDJPY', 'GBPUSD'].
    end_date : str, optional
        As-of end date (YYYY-MM-DD). Defaults to today.

    Returns:
    --------
    Dict[str, float]
        Rates found in store.
    """
    if not required_rates:
        return {}

    # Import lazily so this script can still run without store module.
    try:
        from pathlib import Path as _Path

        _project_root = str(_Path(__file__).resolve().parent.parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from algos.common.market_data_store import MarketDataStore
    except Exception:
        return {}

    rates = {}
    store = MarketDataStore()
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp(end_date) - timedelta(days=15)).strftime("%Y-%m-%d")

    logging.info(f"Checking exchange rates from parquet store: {required_rates}")

    def _fetch_last_close(rate_ticker: str) -> Optional[float]:
        """Fetch last close from store for rate ticker key."""
        candidates = [rate_ticker, f"{rate_ticker}=X"]
        for ticker in candidates:
            try:
                df = store.get_ohlcv(
                    ticker, start=start_date, end=end_date, use_adj_close=True
                )
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
                if close_series.empty:
                    continue
                rate = float(close_series.iloc[-1])
                if rate > 0:
                    return rate
            except Exception:
                continue
        return None

    for rate_key in required_rates:
        direct_rate = _fetch_last_close(rate_key)
        if direct_rate is not None:
            rates[rate_key] = direct_rate
            logging.info(f"  {rate_key} = {direct_rate:.4f} (source: store)")
            continue

        # Try inverse quote if direct pair does not exist in store.
        # Example: CADUSD missing but USDCAD exists => CADUSD = 1 / USDCAD.
        if len(rate_key) == 6 and rate_key.isalpha():
            inverse_key = f"{rate_key[3:]}{rate_key[:3]}"
            inverse_rate = _fetch_last_close(inverse_key)
            if inverse_rate is not None and inverse_rate > 0:
                converted = 1.0 / inverse_rate
                rates[rate_key] = converted
                logging.info(
                    f"  {rate_key} = {converted:.6f} (source: store inverse {inverse_key})"
                )
                continue

        logging.debug(f"  Missing in store: {rate_key}")

    return rates


def convert_price_to_usd(
    ticker: str,
    price_native: float,
    exchange_rates: Dict[str, float],
    exchange_mgr,
    lse_currency_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[float, str]:
    """
    Convert native currency price to USD with proper handling of:
    - LSE stocks (GBX pence → GBP → USD)
    - USD/Foreign format exchange rates

    Parameters:
    -----------
    ticker : str
        Ticker symbol (e.g., 'NVDA', '8002.T', 'III.L')
    price_native : float
        Price in native currency (from CSV/yfinance)
    exchange_rates : Dict[str, float]
        Exchange rates in format {'USDJPY': 158, 'GBPUSD': 1.27, 'USDHKD': 7.8}
    exchange_mgr : ExchangeManager
        For currency lookup

    Returns:
    --------
    Tuple[float, str]
        (price_usd, rate_key_used)
    """
    currency = get_ticker_currency(ticker, exchange_mgr, lse_currency_overrides)

    # LSE stocks: yfinance returns GBX (pence), convert to GBP first
    if is_lse_pence_pricing(ticker, lse_currency_overrides):
        price_native = price_native / 100  # GBX → GBP
        currency = "GBP"
        logging.debug(f"  {ticker}: Converted GBX to GBP: {price_native:.4f}")

    # Tel Aviv stocks: yfinance returns ILA (Agorot), convert to ILS first
    # 1 ILS = 100 Agorot (similar to 1 GBP = 100 pence)
    if ticker.endswith(".TA"):
        price_native = price_native / 100  # Agorot → ILS
        logging.debug(f"  {ticker}: Converted Agorot to ILS: {price_native:.4f}")

    # USD stocks - no conversion needed
    if currency == "USD":
        return price_native, "USD"

    # Determine rate key based on currency
    # Major currencies quoted as XXXUSD (how many USD per 1 XXX): multiply to get USD
    # Minor currencies quoted as USDXXX (how many XXX per 1 USD): divide to get USD

    # Currencies using XXXUSD format (multiply: native * rate = USD)
    xxxusd_currencies = {"GBP", "EUR", "AUD", "CAD", "CHF", "NOK", "NZD", "SGD"}

    if currency in xxxusd_currencies:
        rate_key = f"{currency}USD"
        rate = exchange_rates.get(rate_key)
        if rate is None:
            raise ValueError(f"Missing exchange rate: {rate_key}")
        price_usd = price_native * rate  # XXX * XXXUSD = USD
    else:
        # Most emerging/minor currencies: USDXXX format (JPY, HKD, INR, ILS, HUF, etc.)
        rate_key = f"USD{currency}"
        rate = exchange_rates.get(rate_key)
        if rate is None:
            raise ValueError(f"Missing exchange rate: {rate_key}")
        price_usd = price_native / rate  # XXX / USDXXX = USD

    return price_usd, rate_key


def get_required_exchange_rates(
    tickers: List[str],
    exchange_mgr,
    lse_currency_overrides: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Get list of required exchange rate keys for the given tickers.

    Returns:
    --------
    List[str]
        e.g., ['USDJPY', 'GBPUSD', 'USDHKD', 'CADUSD', 'CHFUSD']
    """
    # Currencies using XXXUSD format (consistent with convert_price_to_usd)
    xxxusd_currencies = {"GBP", "EUR", "AUD", "CAD", "CHF", "NOK", "NZD", "SGD"}

    required = set()
    for ticker in tickers:
        currency = get_ticker_currency(ticker, exchange_mgr, lse_currency_overrides)
        if currency == "USD":
            continue
        elif currency in xxxusd_currencies:
            required.add(f"{currency}USD")
        else:
            required.add(f"USD{currency}")
    return sorted(required)


def parse_exchange_rates(rate_strings: List[str]) -> Dict[str, float]:
    """
    Parse exchange rate strings from CLI in USD/Foreign format.

    Parameters:
    -----------
    rate_strings : List[str]
        Format: ['USDJPY=158', 'GBPUSD=1.27', 'USDHKD=7.8']

    Returns:
    --------
    Dict[str, float]
        {'USDJPY': 158.0, 'GBPUSD': 1.27, 'USDHKD': 7.8}
    """
    rates = {}

    for rate_str in rate_strings:
        if "=" not in rate_str:
            raise ValueError(
                f"Invalid exchange rate format: '{rate_str}'. Expected PAIR=RATE (e.g., USDJPY=158)"
            )
        pair, rate = rate_str.split("=", 1)
        rates[pair.upper()] = float(rate)

    return rates


def parse_lot_sizes(lot_size_strings: List[str]) -> Dict[str, int]:
    """
    Parse lot size override strings from CLI.

    Parameters:
    -----------
    lot_size_strings : List[str]
        Format: ['1277.HK=2000', '8002.T=100']

    Returns:
    --------
    Dict[str, int]
        {'1277.HK': 2000, '8002.T': 100}
    """
    lot_sizes = {}

    for lot_str in lot_size_strings:
        if "=" not in lot_str:
            raise ValueError(
                f"Invalid lot size format: '{lot_str}'. Expected TICKER=SIZE"
            )
        ticker, size = lot_str.split("=", 1)
        lot_sizes[ticker] = int(size)

    return lot_sizes


def fetch_average_volume_30d(
    ticker: str, allow_online_fetch: bool = True
) -> Tuple[Optional[float], str]:
    """Fetch 30-day average daily volume for one ticker.

    Tries parquet store first, then falls back to yfinance.

    Returns:
    --------
    Tuple[Optional[float], str]
        (avg_volume_30d, source) where source is 'store', 'yfinance', or 'none'.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    # Try parquet store first
    try:
        from pathlib import Path as _Path

        _project_root = str(_Path(__file__).resolve().parent.parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from algos.common.market_data_store import MarketDataStore

        store = MarketDataStore()
        df = store.get_ohlcv_raw(ticker, start=start_date, end=end_date)
        if df is not None and not df.empty and "volume" in df.columns:
            vol = pd.to_numeric(df["volume"], errors="coerce").dropna()
            vol = vol[vol > 0]
            if not vol.empty:
                window = vol.tail(30)
                if not window.empty:
                    return float(window.mean()), "store"
    except Exception:
        pass

    if not allow_online_fetch:
        return None, "none"

    # Fallback to yfinance
    try:
        yf_df = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )

        if yf_df is None or yf_df.empty:
            return None, "none"

        if isinstance(yf_df.columns, pd.MultiIndex):
            if "Volume" in yf_df.columns.get_level_values(0):
                vol = yf_df["Volume"]
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, -1]
            else:
                return None, "none"
        else:
            if "Volume" not in yf_df.columns:
                return None, "none"
            vol = yf_df["Volume"]

        vol = pd.to_numeric(vol, errors="coerce").dropna()
        vol = vol[vol > 0]
        if vol.empty:
            return None, "none"

        return float(vol.tail(30).mean()), "yfinance"

    except Exception:
        return None, "none"


def filter_investable_universe(
    csv_path: str,
    budget: float,
    min_weight_pct: float,
    exchange_rates: Dict[str, float],
    lot_size_overrides: Optional[Dict[str, int]] = None,
    tickers: Optional[List[str]] = None,
    exclude_exchanges: Optional[List[str]] = None,
    min_avg_volume_30d: Optional[float] = None,
    min_avg_dollar_volume_30d: Optional[float] = None,
    exclude_missing_volume: bool = False,
    allow_online_fetch: bool = True,
    rate_end_date: Optional[str] = None,
    lse_currency_overrides_file: Optional[str] = None,
) -> Tuple[List[str], List[Dict], Dict[str, Dict]]:
    """
    Filter tickers whose minimum lot cost exceeds minimum position value.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file with price data (Date column + ticker columns)
    budget : float
        Total portfolio budget in USD
    min_weight_pct : float
        Minimum weight percentage per position (e.g., 0.05 for 5%)
    exchange_rates : Dict[str, float]
        Exchange rates in format {'USDJPY': 158, 'GBPUSD': 1.27, 'USDHKD': 7.8}
    lot_size_overrides : Dict[str, int], optional
        Per-ticker lot size overrides
    tickers : List[str], optional
        Specific tickers to filter. If None, uses all columns from CSV.
    exclude_exchanges : List[str], optional
        Exchange suffixes to exclude (e.g., ['.JK', '.VN', '.AAA'])
    min_avg_volume_30d : float, optional
        Minimum 30-day average daily volume threshold. If provided, tickers below
        this threshold are excluded as illiquid.
    min_avg_dollar_volume_30d : float, optional
        Minimum 30-day average daily dollar volume threshold in USD.
        Computed as price_usd * avg_volume_30d * lot_size.
    exclude_missing_volume : bool
        If True, exclude tickers when 30-day volume cannot be determined.
    allow_online_fetch : bool
        If False, disable all network fallback (yfinance/HKEX downloads).
    rate_end_date : str, optional
        End date used for store-based FX rate lookup.

    Returns:
    --------
    Tuple[List[str], List[Dict], Dict[str, Dict]]
        (included_tickers, excluded_tickers_with_details, all_ticker_details)

        excluded_tickers_with_details contains dicts with:
        {
            'ticker': str,
            'price_native': float,
            'currency': str,
            'lot_size': int,
            'min_lot_cost_usd': float,
            'min_position_value': float,
            'excess_pct': float,
            'reason': str
        }
    """
    exchange_mgr = ExchangeManager()
    lse_currency_overrides = load_lse_currency_overrides(lse_currency_overrides_file)
    lot_size_overrides = lot_size_overrides or {}
    exclude_exchanges = exclude_exchanges or []

    # Normalize exchange suffixes (ensure they start with '.')
    exclude_exchanges = [s if s.startswith(".") else f".{s}" for s in exclude_exchanges]

    min_position_value = budget * min_weight_pct

    # Get column names from CSV/parquet if not provided
    if tickers is None:
        if str(csv_path).endswith(".parquet"):
            df = pd.read_parquet(csv_path)
            if df.index.name != "Date":
                df.index.name = "Date"
        else:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        tickers = list(df.columns)

    if not tickers:
        raise ValueError("No tickers found in CSV file")

    # Resolve column names to yfinance tickers
    # This handles cases like 'Marubeni' → '8002.T', '3I' → 'III.L'
    column_to_ticker = {}
    resolved_tickers = []
    exchange_excluded = []  # Track tickers excluded by exchange filter

    for col in tickers:
        resolved = resolve_ticker_name(col)
        column_to_ticker[col] = resolved

        # Check if ticker should be excluded by exchange
        if any(resolved.endswith(suffix) for suffix in exclude_exchanges):
            exchange_excluded.append((col, resolved))
            continue

        resolved_tickers.append(resolved)
        if resolved != col:
            logging.info(f"  Resolved: {col} → {resolved}")

    # Log exchange exclusions
    if exchange_excluded:
        logging.info(
            f"  Excluded {len(exchange_excluded)} tickers by exchange filter: {exclude_exchanges}"
        )
        for col, resolved in exchange_excluded[:5]:  # Show first 5
            logging.info(f"    {col} → {resolved}")
        if len(exchange_excluded) > 5:
            logging.info(f"    ... and {len(exchange_excluded) - 5} more")

    # Remove exchange-excluded tickers from processing list
    tickers = [
        col
        for col in tickers
        if column_to_ticker[col] not in [r for _, r in exchange_excluded]
    ]

    # Get latest prices (using original column names)
    prices_dict = get_latest_prices_from_csv(csv_path, tickers)

    # Fetch HKEX lot sizes for HK stocks (using resolved tickers)
    hkex_lot_sizes = {}
    hk_tickers = [t for t in resolved_tickers if t.endswith(".HK")]
    if hk_tickers:
        logging.info("Fetching HKEX lot sizes...")
        hkex_lot_sizes = fetch_hkex_lot_sizes(allow_online_fetch=allow_online_fetch)

    # Check required exchange rates and auto-fetch missing ones
    required_rates = get_required_exchange_rates(
        resolved_tickers,
        exchange_mgr,
        lse_currency_overrides=lse_currency_overrides,
    )
    missing = [rate for rate in required_rates if rate not in exchange_rates]

    # First try local parquet store for missing rates.
    if missing:
        store_rates = fetch_exchange_rates_from_store(missing, end_date=rate_end_date)
        exchange_rates.update(store_rates)
        missing = [rate for rate in required_rates if rate not in exchange_rates]

    if missing:
        if allow_online_fetch:
            logging.info(f"Auto-fetching missing exchange rates: {missing}")
            fetched_rates = fetch_exchange_rates(missing)
            exchange_rates.update(fetched_rates)
            missing = [rate for rate in required_rates if rate not in exchange_rates]
        else:
            logging.info(
                f"Online exchange-rate fetch disabled; missing rates must be provided manually: {missing}"
            )

        # Check if any rates still missing after store + optional online fetch
        if missing:
            missing_specs = " ".join(f"{rate}=<rate>" for rate in missing)
            raise ValueError(
                f"Could not fetch exchange rates: {missing}.\n"
                f"Add manually with: --exchange-rates {missing_specs}\n"
                f"Format: USDJPY=158 GBPUSD=1.27 USDHKD=7.8"
            )

    # Fetch liquidity metrics if requested
    avg_volume_map = {}
    volume_source_map = {}
    if min_avg_volume_30d is not None or min_avg_dollar_volume_30d is not None:
        logging.info(
            f"Checking 30-day liquidity metrics: "
            f"avg_volume>={min_avg_volume_30d if min_avg_volume_30d is not None else 'N/A'}, "
            f"avg_dollar_volume_usd>={min_avg_dollar_volume_30d if min_avg_dollar_volume_30d is not None else 'N/A'}"
        )
        if min_avg_dollar_volume_30d is not None and min_avg_volume_30d is not None:
            logging.info(
                "Both volume filters provided: using --min-avg-dollar-volume-30d as primary threshold"
            )
        for ticker in resolved_tickers:
            avg_vol, vol_source = fetch_average_volume_30d(
                ticker, allow_online_fetch=allow_online_fetch
            )
            avg_volume_map[ticker] = avg_vol
            volume_source_map[ticker] = vol_source

    included = []
    excluded = []
    all_details = {}

    for column_name in tickers:
        # Resolve column name to yfinance ticker
        ticker = column_to_ticker[column_name]
        price_native = prices_dict.get(column_name)

        # Handle invalid prices
        if price_native is None or price_native <= 0:
            detail = {
                "ticker": column_name,
                "yf_ticker": ticker,
                "price_native": price_native,
                "currency": "N/A",
                "lot_size": 0,
                "min_lot_cost_usd": 0,
                "min_position_value": min_position_value,
                "excess_pct": 0,
                "reason": f"Invalid price: {price_native}",
                "included": False,
            }
            excluded.append(detail)
            all_details[column_name] = detail
            continue

        # Get currency (using resolved ticker)
        currency = get_ticker_currency(ticker, exchange_mgr, lse_currency_overrides)
        lse_pence_pricing = is_lse_pence_pricing(ticker, lse_currency_overrides)

        # Get lot size (priority: override > HKEX > exchange_manager default)
        # Check overrides with both column name and resolved ticker
        if column_name in lot_size_overrides:
            lot_size = lot_size_overrides[column_name]
        elif ticker in lot_size_overrides:
            lot_size = lot_size_overrides[ticker]
        elif ticker.endswith(".HK"):
            # Normalize HK ticker to Yahoo Finance 4-digit format for lookup
            # Handles discrepancy: user may have 5.HK but HKEX data is keyed as 0005.HK
            normalized_hk = normalize_hk_ticker(ticker)
            if normalized_hk in hkex_lot_sizes:
                lot_size = hkex_lot_sizes[normalized_hk]
            else:
                lot_size = exchange_mgr.get_lot_size(ticker)
        else:
            lot_size = exchange_mgr.get_lot_size(ticker)

        # Convert price to USD using proper conversion logic (using resolved ticker)
        try:
            price_usd, rate_key = convert_price_to_usd(
                ticker,
                price_native,
                exchange_rates,
                exchange_mgr,
                lse_currency_overrides=lse_currency_overrides,
            )
        except ValueError as e:
            detail = {
                "ticker": column_name,
                "yf_ticker": ticker,
                "price_native": price_native,
                "currency": currency,
                "lot_size": lot_size,
                "min_lot_cost_usd": 0,
                "min_position_value": min_position_value,
                "excess_pct": 0,
                "reason": str(e),
                "included": False,
            }
            excluded.append(detail)
            all_details[column_name] = detail
            continue

        # Calculate min lot cost in USD
        min_lot_cost_usd = price_usd * lot_size

        detail = {
            "ticker": column_name,
            "yf_ticker": ticker,
            "price_native": price_native,
            "price_usd": price_usd,
            "currency": currency,
            "lse_pence_pricing": lse_pence_pricing,
            "rate_key": rate_key,
            "lot_size": lot_size,
            "min_lot_cost_usd": min_lot_cost_usd,
            "min_position_value": min_position_value,
            "excess_pct": (min_lot_cost_usd - min_position_value)
            / min_position_value
            * 100
            if min_position_value > 0
            else 0,
            "avg_volume_30d": avg_volume_map.get(ticker),
            "avg_dollar_volume_30d_usd": None,
            "volume_source": volume_source_map.get(ticker, "none"),
        }

        # Compute dollar-volume metric (USD notional traded per day)
        # using market lot granularity for venues with lot trading.
        avg_vol = detail.get("avg_volume_30d")
        if avg_vol is not None:
            detail["avg_dollar_volume_30d_usd"] = price_usd * avg_vol * lot_size

        # Apply liquidity filter first (if enabled).
        # Priority: dollar-volume threshold overrides plain volume threshold.
        if min_avg_dollar_volume_30d is not None:
            avg_dollar_vol = detail.get("avg_dollar_volume_30d_usd")
            if avg_dollar_vol is None:
                if exclude_missing_volume:
                    detail["included"] = False
                    detail["reason"] = "Missing 30D dollar volume"
                    excluded.append(detail)
                    all_details[column_name] = detail
                    continue
            elif avg_dollar_vol < min_avg_dollar_volume_30d:
                detail["included"] = False
                detail["reason"] = (
                    f"Illiquid: 30D avg dollar vol ${avg_dollar_vol:,.0f} < ${min_avg_dollar_volume_30d:,.0f}"
                )
                excluded.append(detail)
                all_details[column_name] = detail
                continue
        elif min_avg_volume_30d is not None:
            if avg_vol is None:
                if exclude_missing_volume:
                    detail["included"] = False
                    detail["reason"] = "Missing 30D volume"
                    excluded.append(detail)
                    all_details[column_name] = detail
                    continue
            elif avg_vol < min_avg_volume_30d:
                detail["included"] = False
                detail["reason"] = (
                    f"Illiquid: 30D avg vol {avg_vol:,.0f} < {min_avg_volume_30d:,.0f}"
                )
                excluded.append(detail)
                all_details[column_name] = detail
                continue

        # Apply filter
        if min_lot_cost_usd <= min_position_value:
            detail["included"] = True
            detail["reason"] = "OK"
            # Mark marginal cases
            if min_lot_cost_usd > min_position_value * 0.8:
                detail["reason"] = "OK (marginal)"
            included.append(column_name)  # Use original column name for output
        else:
            detail["included"] = False
            detail["reason"] = (
                f"Min lot ${min_lot_cost_usd:,.2f} > threshold ${min_position_value:,.2f}"
            )
            excluded.append(detail)

        all_details[column_name] = detail

    return included, excluded, all_details


def display_filter_results(
    included: List[str],
    excluded: List[Dict],
    all_details: Dict[str, Dict],
    budget: float,
    min_weight_pct: float,
    exchange_rates: Dict[str, float],
) -> None:
    """Display formatted filter results."""
    min_position_value = budget * min_weight_pct
    total_tickers = len(included) + len(excluded)

    logging.info("=" * 80)
    logging.info(" INVESTABLE UNIVERSE FILTER")
    logging.info("=" * 80)
    logging.info("")
    logging.info("Configuration:")
    logging.info(f"  Budget:              ${budget:,.2f}")
    logging.info(f"  Min Weight:          {min_weight_pct * 100:.2f}%")
    logging.info(f"  Min Position Value:  ${min_position_value:,.2f}")
    logging.info("")

    # Show exchange rates in new format
    if exchange_rates:
        logging.info("Exchange Rates:")
        for pair, rate in sorted(exchange_rates.items()):
            # Format rate appropriately based on pair
            if pair.startswith("USD"):  # USDJPY, USDHKD
                logging.info(f"  {pair} = {rate:.2f} (1 USD = {rate:.2f} {pair[3:]})")
            else:  # GBPUSD, EURUSD, AUDUSD
                logging.info(f"  {pair} = {rate:.4f} (1 {pair[:3]} = {rate:.4f} USD)")
        logging.info("")

    logging.info("=" * 80)
    logging.info(" FILTER RESULTS")
    logging.info("=" * 80)
    logging.info("")

    # Show included tickers
    logging.info(f"INCLUDED ({len(included)} tickers):")
    if included:
        logging.info(
            f"  {'Ticker':<20} {'yf_ticker':<12} {'Price':>12} {'Currency':<8} {'Lot':>6} {'Min Lot Cost':>14} {'Status':<15}"
        )
        logging.info(
            f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 8} {'-' * 6} {'-' * 14} {'-' * 15}"
        )

        # Sort by min lot cost descending (show marginal cases first)
        sorted_included = sorted(
            included, key=lambda t: all_details[t]["min_lot_cost_usd"], reverse=True
        )

        for ticker in sorted_included:
            d = all_details[ticker]
            yf_ticker = d.get("yf_ticker", ticker)

            # Format price with currency symbol
            if d["currency"] == "JPY":
                price_str = f"¥{d['price_native']:,.0f}"
            elif d["currency"] == "GBP":
                # LSE GBX listings are displayed in pence.
                if d.get("lse_pence_pricing", False):
                    price_str = f"{d['price_native']:,.0f}p"  # pence
                else:
                    price_str = f"£{d['price_native']:,.2f}"
            elif d["currency"] == "HKD":
                price_str = f"HK${d['price_native']:,.2f}"
            elif d["currency"] == "INR":
                price_str = f"₹{d['price_native']:,.2f}"
            elif d["currency"] == "EUR":
                price_str = f"€{d['price_native']:,.2f}"
            elif d["currency"] == "AUD":
                price_str = f"A${d['price_native']:,.2f}"
            elif d["currency"] == "CAD":
                price_str = f"C${d['price_native']:,.2f}"
            elif d["currency"] == "CHF":
                price_str = f"CHF{d['price_native']:,.2f}"
            elif d["currency"] == "NOK":
                price_str = f"kr{d['price_native']:,.2f}"
            elif d["currency"] == "ILS":
                # Tel Aviv stocks: yfinance returns Agorot (1/100 ILS), show as Agorot
                if yf_ticker.endswith(".TA"):
                    price_str = f"{d['price_native']:,.0f}ag"  # Agorot
                else:
                    price_str = f"₪{d['price_native']:,.2f}"
            elif d["currency"] == "HUF":
                price_str = f"Ft{d['price_native']:,.0f}"
            elif d["currency"] == "SGD":
                price_str = f"S${d['price_native']:,.2f}"
            elif d["currency"] == "SEK":
                price_str = f"kr{d['price_native']:,.2f}"
            elif d["currency"] == "SAR":
                price_str = f"SR{d['price_native']:,.2f}"
            elif d["currency"] == "MYR":
                price_str = f"RM{d['price_native']:,.2f}"
            elif d["currency"] == "CZK":
                price_str = f"Kč{d['price_native']:,.0f}"
            elif d["currency"] == "AED":
                price_str = f"AED{d['price_native']:,.2f}"
            elif d["currency"] == "DKK":
                price_str = f"kr{d['price_native']:,.2f}"
            elif d["currency"] == "RON":
                price_str = f"lei{d['price_native']:,.2f}"
            elif d["currency"] == "PLN":
                price_str = f"zł{d['price_native']:,.2f}"
            else:
                price_str = f"${d['price_native']:,.2f}"

            # Show yf_ticker only if different from column name
            yf_display = yf_ticker if yf_ticker != ticker else ""
            logging.info(
                f"  {ticker:<20} {yf_display:<12} {price_str:>12} {d['currency']:<8} {d['lot_size']:>6} "
                f"${d['min_lot_cost_usd']:>12,.2f} {d['reason']:<15}"
            )
    else:
        logging.info("  (none)")

    logging.info("")

    # Show excluded tickers
    logging.info(f"EXCLUDED ({len(excluded)} tickers):")
    if excluded:
        logging.info(
            f"  {'Ticker':<20} {'yf_ticker':<12} {'Min Lot Cost':>14} {'Threshold':>12} {'Excess':>10} {'Reason':<30}"
        )
        logging.info(
            f"  {'-' * 20} {'-' * 12} {'-' * 14} {'-' * 12} {'-' * 10} {'-' * 30}"
        )

        for d in sorted(excluded, key=lambda x: x.get("excess_pct", 0), reverse=True):
            excess_str = (
                f"+{d['excess_pct']:.1f}%" if d.get("excess_pct", 0) > 0 else "N/A"
            )
            min_lot_str = (
                f"${d['min_lot_cost_usd']:,.2f}"
                if d.get("min_lot_cost_usd", 0) > 0
                else "N/A"
            )
            yf_ticker = d.get("yf_ticker", d["ticker"])
            yf_display = yf_ticker if yf_ticker != d["ticker"] else ""
            logging.info(
                f"  {d['ticker']:<20} {yf_display:<12} {min_lot_str:>14} ${min_position_value:>10,.2f} {excess_str:>10} {d['reason']:<30}"
            )
    else:
        logging.info("  (none)")

    logging.info("")
    logging.info("=" * 80)
    logging.info(" SUMMARY")
    logging.info("=" * 80)
    logging.info("")
    logging.info(f"  Total tickers:  {total_tickers}")
    logging.info(
        f"  Included:       {len(included)} ({len(included) / total_tickers * 100:.1f}%)"
    )
    logging.info(
        f"  Excluded:       {len(excluded)} ({len(excluded) / total_tickers * 100:.1f}%)"
    )
    logging.info("")

    # Warnings
    if len(included) < 3:
        logging.warning(
            "  WARNING: Fewer than 3 tickers included. Consider increasing budget or decreasing min weight."
        )
    if len(excluded) > len(included):
        logging.warning(
            "  WARNING: More tickers excluded than included. Budget may be too small for this universe."
        )

    # Recommendations
    if excluded:
        max_excluded_cost = max(d.get("min_lot_cost_usd", 0) for d in excluded)
        if max_excluded_cost > 0:
            required_budget = max_excluded_cost / min_weight_pct
            logging.info(
                f"  To include all tickers: increase budget to ${required_budget:,.0f}"
            )
            logging.info(
                f"  Or: decrease min weight to {max_excluded_cost / budget * 100:.2f}%"
            )

    logging.info("")


def save_filtered_tickers(included: List[str], output_path: str) -> None:
    """Save filtered ticker list to file."""
    with open(output_path, "w") as f:
        for ticker in sorted(included):
            f.write(f"{ticker}\n")
    logging.info(f"Filtered tickers written to: {output_path}")


def prompt_user_confirmation(
    included: List[str], excluded: List[Dict], auto_mode: bool = False
) -> bool:
    """Interactive confirmation for semi-automated mode."""
    if auto_mode:
        return True

    print("")
    response = (
        input(f"Proceed with {len(included)} included tickers? [Y/n]: ").strip().lower()
    )
    return response in ("", "y", "yes")


def main():
    parser = argparse.ArgumentParser(
        description="Filter investable universe based on lot size constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - exchange rates auto-fetched from yfinance
  python filter_investable_universe.py \\
      --csv ../../data/prices.csv \\
      --budget 50000 \\
      --min-weight 0.05

  # Override specific exchange rates (others auto-fetched)
  python filter_investable_universe.py \\
      --csv ../../data/prices.csv \\
      --budget 50000 \\
      --min-weight 0.05 \\
      --exchange-rates USDJPY=160

  # With custom lot sizes and output file
  python filter_investable_universe.py \\
      --csv ../../data/prices.csv \\
      --budget 50000 \\
      --min-weight 0.05 \\
      --lot-sizes 1277.HK=2000 1288.HK=1000 \\
      --output filtered_tickers.txt

  # Auto-calculate min weight (equal weight for N tickers)
  python filter_investable_universe.py \\
      --csv ../../data/prices.csv \\
      --budget 30000 \\
      --auto-weight

  # Automatic mode (no prompts, for scripting)
  python filter_investable_universe.py \\
      --csv ../../data/prices.csv \\
      --budget 50000 \\
      --min-weight 0.05 \\
      --auto

Exchange Rate Format:
  USDJPY=158    -> 1 USD = 158 JPY  (divide JPY price by rate to get USD)
  GBPUSD=1.27   -> 1 GBP = 1.27 USD (multiply GBP price by rate to get USD)
  USDHKD=7.8    -> 1 USD = 7.8 HKD  (divide HKD price by rate to get USD)
  EURUSD=1.08   -> 1 EUR = 1.08 USD (multiply EUR price by rate to get USD)

Workflow Integration:
  # Step 1: Validate data quality
  python scripts/validate_data_csv.py --csv data.csv --start 2022-01-01

  # Step 2: Filter investable universe (THIS SCRIPT)
  python algos/backtest_code/filter_investable_universe.py \\
      --csv data.csv --budget 50000 --min-weight 0.05 \\
      --exchange-rates USDJPY=158 GBPUSD=1.27 --output filtered_tickers.txt

  # Step 3: Run portfolio optimization
  python algos/backtest_code/portfolio_exploration_global.py \\
      --csv data.csv --start 2022-01-01 --end 2025-01-01 \\
      --tickers-file filtered_tickers.txt
        """,
    )

    # Data source (one of --csv or --from-store is required)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV or parquet file with price data",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="Read data from the parquet market data store (no CSV needed).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date for --from-store export (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for --from-store export (YYYY-MM-DD)",
    )

    # Required arguments
    parser.add_argument(
        "--budget", type=float, required=True, help="Total portfolio budget in USD"
    )

    # Weight specification (mutually exclusive)
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument(
        "--min-weight",
        type=float,
        help="Minimum weight percentage (e.g., 0.05 for 5%%)",
    )
    weight_group.add_argument(
        "--auto-weight",
        action="store_true",
        help="Auto-calculate min weight as 0.5/n (half of equal weight)",
    )

    # Multi-currency support
    parser.add_argument(
        "--lse-currency-overrides",
        type=str,
        default=str(LSE_CURRENCY_OVERRIDES_PATH),
        help=(
            "Path to JSON file with per-ticker LSE currency overrides "
            "for non-GBX listings"
        ),
    )
    parser.add_argument(
        "--exchange-rates",
        type=str,
        nargs="*",
        default=[],
        help="Override exchange rates (auto-fetched if not provided): PAIR=RATE (e.g., USDJPY=158)",
    )

    # Lot size overrides
    parser.add_argument(
        "--lot-sizes",
        type=str,
        nargs="*",
        default=[],
        help="Lot size overrides: TICKER=SIZE (e.g., 1277.HK=2000)",
    )

    # Ticker filtering
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=None,
        help="Specific tickers to filter (default: all from CSV)",
    )
    parser.add_argument(
        "--exclude-exchanges",
        type=str,
        nargs="*",
        default=[],
        help="Exclude tickers from these exchanges by suffix (e.g., .JK .VN .AAA .TW)",
    )
    parser.add_argument(
        "--min-avg-volume-30d",
        type=float,
        default=None,
        help="Minimum 30-day average daily volume threshold for liquidity filtering",
    )
    parser.add_argument(
        "--min-avg-dollar-volume-30d",
        type=float,
        default=5000000,
        help=(
            "Minimum 30-day average daily dollar volume in USD "
            "(price_usd * avg_volume_30d * lot_size). Default: 5000000"
        ),
    )
    parser.add_argument(
        "--exclude-missing-volume",
        action="store_true",
        help="Exclude tickers when 30-day average volume cannot be determined",
    )
    parser.add_argument(
        "--allow-online-fetch",
        action="store_true",
        help="Allow online fallback fetches (yfinance/HKEX) even in --from-store mode",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file for filtered ticker list",
    )

    # Automation
    parser.add_argument(
        "--auto", action="store_true", help="Automatic mode (no user prompts)"
    )

    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Resolve data source: --from-store or --csv
    if args.from_store:
        import tempfile
        from pathlib import Path as _Path

        _project_root = str(_Path(__file__).resolve().parent.parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        try:
            from algos.common.market_data_store import MarketDataStore
            from algos.common.update_market_data import load_ticker_universe

            store = MarketDataStore()
            tickers_map = load_ticker_universe()
            if not tickers_map:
                logging.error("No tickers found in ticker_universe.json")
                sys.exit(1)

            # Determine date range (default: last 5 years)
            from datetime import datetime as _dt, timedelta as _td

            end_date = args.end or _dt.now().strftime("%Y-%m-%d")
            start_date = args.start or (_dt.now() - _td(days=1825)).strftime("%Y-%m-%d")

            tmp = tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, prefix="filter_data_"
            )
            args.csv = tmp.name
            tmp.close()

            export_df = store.export_portfolio_csv(
                tickers_map,
                start_date,
                end_date,
                args.csv,
                min_coverage=0.8,
            )
            if export_df.empty:
                logging.error(
                    "Parquet store returned no data. "
                    "Run 'python -m algos.common.update_market_data --init' first."
                )
                sys.exit(1)
            logging.info(f"Exported {export_df.shape} from parquet store")
        except ImportError as e:
            logging.error(f"MarketDataStore not available: {e}. Use --csv instead.")
            sys.exit(1)
    elif args.csv is None:
        parser.error("Either --csv or --from-store is required.")

    # Validate CSV exists
    if not os.path.exists(args.csv):
        logging.error(f"CSV file not found: {args.csv}")
        sys.exit(1)

    # Parse exchange rates
    try:
        exchange_rates = parse_exchange_rates(args.exchange_rates)
    except ValueError as e:
        logging.error(f"Error parsing exchange rates: {e}")
        sys.exit(1)

    # Parse lot sizes
    try:
        lot_size_overrides = parse_lot_sizes(args.lot_sizes)
    except ValueError as e:
        logging.error(f"Error parsing lot sizes: {e}")
        sys.exit(1)

    # Determine min weight
    if args.auto_weight:
        # Auto-calculate based on target portfolio size, not total universe size.
        # The optimizer typically selects 20-40 tickers from the universe.
        # We want to check: "if this ticker gets the minimum weight in a
        # ~40-stock optimized portfolio, can I afford at least 1 lot?"
        #
        # Formula: 0.5 / target_portfolio_size  (half of equal weight)
        # This gives a conservative floor -- if even the smallest position
        # in the optimized portfolio can't buy 1 lot, the ticker is excluded.
        target_portfolio_size = 40  # Typical stage2_target from portimization.py
        min_weight_pct = 0.5 / target_portfolio_size  # 1.25% for 40 tickers
        logging.info(
            f"Auto-calculated min weight: {min_weight_pct * 100:.2f}% "
            f"(0.5/{target_portfolio_size} target portfolio tickers, "
            f"threshold: ${args.budget * min_weight_pct:,.0f})"
        )
    else:
        min_weight_pct = args.min_weight

    # Run filter
    allow_online_fetch = args.allow_online_fetch or (not args.from_store)
    if args.from_store and not allow_online_fetch:
        logging.info(
            "--from-store active: online fetch disabled for rates/volume/lot-size fallbacks"
        )

    try:
        included, excluded, all_details = filter_investable_universe(
            csv_path=args.csv,
            budget=args.budget,
            min_weight_pct=min_weight_pct,
            exchange_rates=exchange_rates,
            lot_size_overrides=lot_size_overrides,
            tickers=args.tickers,
            exclude_exchanges=args.exclude_exchanges,
            min_avg_volume_30d=args.min_avg_volume_30d,
            min_avg_dollar_volume_30d=args.min_avg_dollar_volume_30d,
            exclude_missing_volume=args.exclude_missing_volume,
            allow_online_fetch=allow_online_fetch,
            rate_end_date=args.end,
            lse_currency_overrides_file=args.lse_currency_overrides,
        )
    except ValueError as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

    # Display results
    display_filter_results(
        included, excluded, all_details, args.budget, min_weight_pct, exchange_rates
    )

    # Prompt for confirmation (unless auto mode)
    if not prompt_user_confirmation(included, excluded, args.auto):
        logging.info("Aborted by user.")
        sys.exit(0)

    # Save output
    if args.output:
        save_filtered_tickers(included, args.output)
    else:
        # Default output path
        default_output = os.path.join(BASE_LOG_DIR, f"filtered_tickers_{TIMESTAMP}.txt")
        save_filtered_tickers(included, default_output)

    logging.info(f"Log file: {LOG_FILE_PATH}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
