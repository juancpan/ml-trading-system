"""
Feature Engineering Engine for ML trading models.

Centralized, configurable feature computation shared identically between
backtesting and live trading. Uses pandas-ta-classic for technical indicators.

Architecture:
    FeatureRegistry  - maps indicator names to compute functions
    FeatureConfig    - loads/validates YAML config, resolves per-model overrides
    FeatureEngine    - computes indicators from OHLCV + external data
"""

import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import yaml

# pandas-ta-classic for technical indicators
try:
    import pandas_ta_classic as ta

    _HAS_PANDAS_TA = True
except ImportError:
    try:
        import pandas_ta as ta

        _HAS_PANDAS_TA = True
    except ImportError:
        _HAS_PANDAS_TA = False
        print(
            "Warning: pandas-ta-classic not installed. Feature engineering will be limited. "
            "Install with: pip install pandas-ta-classic"
        )


def _safe_rsi(gain: pd.Series, loss: pd.Series) -> pd.Series:
    """Compute RSI from gain/loss avoiding NaN when loss or gain is zero.

    Standard RSI formula: RSI = 100 - 100/(1+RS) where RS = avg_gain/avg_loss.
    Edge cases:
        loss==0, gain>0  -> RSI = 100 (fully overbought, no losses)
        gain==0, loss>0  -> RSI = 0   (fully oversold, no gains)
        gain==0, loss==0 -> RSI = 50  (neutral, no movement)
    """
    rs = gain / loss  # inf when loss=0 and gain>0; nan when 0/0
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where rs is inf (loss=0, gain>0): RSI should be 100
    # Where rs is nan (both 0): RSI should be 50
    # Where rs is 0 (gain=0, loss>0): formula gives 0 correctly
    mask_inf = np.isinf(rs)
    mask_nan = rs.isna() & gain.notna() & loss.notna()  # true NaN from 0/0, not warmup
    rsi = rsi.copy()
    rsi[mask_inf] = 100.0
    rsi[mask_nan] = 50.0
    return rsi


# ---------------------------------------------------------------------------
# Feature Registry — maps indicator names to compute functions
# ---------------------------------------------------------------------------


class FeatureRegistry:
    """
    Registry of indicator compute functions.

    Each registered function has signature:
        (df: pd.DataFrame, **params) -> pd.DataFrame

    The returned DataFrame has one or more columns named with the indicator prefix.
    The input df must contain standardized lowercase OHLCV columns:
        open, high, low, close, volume, returns
    """

    def __init__(self):
        self._registry: dict[str, callable] = {}
        self._warmup: dict[str, int] = {}  # indicator -> max warmup periods needed
        self._register_builtins()

    def register(self, name: str, func: callable, warmup: int = 0) -> None:
        """Register an indicator compute function."""
        self._registry[name] = func
        self._warmup[name] = warmup

    def get(self, name: str) -> callable:
        """Get a registered indicator function."""
        if name not in self._registry:
            raise KeyError(
                f"Unknown indicator '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def get_warmup(self, name: str) -> int:
        """Get the warmup period for an indicator."""
        return self._warmup.get(name, 0)

    def list_indicators(self) -> list[str]:
        """List all available indicator names."""
        return sorted(self._registry.keys())

    def _register_builtins(self) -> None:
        """Register all built-in indicators."""

        # --- Lagged returns (existing behavior) ---
        def compute_lagged_returns(
            df: pd.DataFrame, lags: list[int] = None, **kwargs
        ) -> pd.DataFrame:
            if lags is None:
                lags = [1, 2, 3, 4, 5]
            result = pd.DataFrame(index=df.index)
            for lag in lags:
                result[f"lag_{lag}"] = df["returns"].shift(lag)
            return result

        self.register("lagged_returns", compute_lagged_returns, warmup=5)

        # --- Trend indicators ---
        def compute_sma(df: pd.DataFrame, period: int = 20, **kwargs) -> pd.DataFrame:
            close = df["close"] if "close" in df.columns else df["price"]
            sma = close.rolling(window=period, min_periods=period).mean()
            # Normalize as % deviation from close (makes feature stationary)
            pct_dev = (close - sma) / sma
            return pd.DataFrame({f"sma_{period}_dev": pct_dev}, index=df.index)

        def compute_ema(df: pd.DataFrame, period: int = 20, **kwargs) -> pd.DataFrame:
            close = df["close"] if "close" in df.columns else df["price"]
            ema = close.ewm(span=period, adjust=False, min_periods=period).mean()
            pct_dev = (close - ema) / ema
            return pd.DataFrame({f"ema_{period}_dev": pct_dev}, index=df.index)

        self.register("sma", compute_sma, warmup=50)
        self.register("ema", compute_ema, warmup=20)

        # --- Momentum indicators ---
        def compute_rsi(df: pd.DataFrame, period: int = 14, **kwargs) -> pd.DataFrame:
            close = df["close"] if "close" in df.columns else df["price"]
            if _HAS_PANDAS_TA:
                rsi = ta.rsi(close, length=period)
            else:
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
                rsi = _safe_rsi(gain, loss)
            # RSI is bounded [0, 100]; rescale to [-1, 1] centered on 50
            rsi_scaled = (rsi - 50.0) / 50.0
            return pd.DataFrame({f"rsi_{period}": rsi_scaled}, index=df.index)

        def compute_macd(
            df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs
        ) -> pd.DataFrame:
            close = df["close"] if "close" in df.columns else df["price"]
            if _HAS_PANDAS_TA:
                macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
                if macd_df is not None and not macd_df.empty:
                    result = pd.DataFrame(index=df.index)
                    cols = macd_df.columns.tolist()
                    # pandas-ta returns: MACD_{fast}_{slow}_{signal}, MACDh_..., MACDs_...
                    # Normalize MACD by close price to make it scale-independent
                    for i, col in enumerate(cols):
                        if i == 0:
                            result[f"macd_{fast}_{slow}"] = macd_df[col] / close
                        elif i == 1:
                            result[f"macd_hist_{fast}_{slow}"] = macd_df[col] / close
                        elif i == 2:
                            result[f"macd_signal_{fast}_{slow}"] = macd_df[col] / close
                    return result
            # Fallback
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = (ema_fast - ema_slow) / close
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return pd.DataFrame(
                {
                    f"macd_{fast}_{slow}": macd_line,
                    f"macd_hist_{fast}_{slow}": histogram,
                    f"macd_signal_{fast}_{slow}": signal_line,
                },
                index=df.index,
            )

        def compute_roc(df: pd.DataFrame, period: int = 10, **kwargs) -> pd.DataFrame:
            close = df["close"] if "close" in df.columns else df["price"]
            roc = close.pct_change(periods=period)
            return pd.DataFrame({f"roc_{period}": roc}, index=df.index)

        def compute_stoch(
            df: pd.DataFrame, k: int = 14, d: int = 3, **kwargs
        ) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close"]):
                return pd.DataFrame(index=df.index)
            if _HAS_PANDAS_TA:
                stoch_df = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d)
                if stoch_df is not None and not stoch_df.empty:
                    result = pd.DataFrame(index=df.index)
                    cols = stoch_df.columns.tolist()
                    for i, col in enumerate(cols):
                        if i == 0:
                            result[f"stoch_k_{k}"] = (stoch_df[col] - 50.0) / 50.0
                        elif i == 1:
                            result[f"stoch_d_{k}"] = (stoch_df[col] - 50.0) / 50.0
                    return result
            # Fallback
            low_min = df["low"].rolling(window=k).min()
            high_max = df["high"].rolling(window=k).max()
            k_val = (
                100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
            )
            d_val = k_val.rolling(window=d).mean()
            return pd.DataFrame(
                {
                    f"stoch_k_{k}": (k_val - 50.0) / 50.0,
                    f"stoch_d_{k}": (d_val - 50.0) / 50.0,
                },
                index=df.index,
            )

        def compute_williams_r(
            df: pd.DataFrame, period: int = 14, **kwargs
        ) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close"]):
                return pd.DataFrame(index=df.index)
            if _HAS_PANDAS_TA:
                wr = ta.willr(df["high"], df["low"], df["close"], length=period)
                if wr is not None:
                    return pd.DataFrame({f"willr_{period}": wr / 100.0}, index=df.index)
            high_max = df["high"].rolling(window=period).max()
            low_min = df["low"].rolling(window=period).min()
            wr = (
                (high_max - df["close"])
                / (high_max - low_min).replace(0, np.nan)
                * -100
            )
            return pd.DataFrame({f"willr_{period}": wr / 100.0}, index=df.index)

        def compute_cci(df: pd.DataFrame, period: int = 20, **kwargs) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close"]):
                return pd.DataFrame(index=df.index)
            if _HAS_PANDAS_TA:
                cci = ta.cci(df["high"], df["low"], df["close"], length=period)
                if cci is not None:
                    return pd.DataFrame({f"cci_{period}": cci / 200.0}, index=df.index)
            tp = (df["high"] + df["low"] + df["close"]) / 3
            sma_tp = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True
            )
            cci = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            return pd.DataFrame({f"cci_{period}": cci / 200.0}, index=df.index)

        def compute_mfi(df: pd.DataFrame, period: int = 14, **kwargs) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
                return pd.DataFrame(index=df.index)
            if _HAS_PANDAS_TA:
                mfi = ta.mfi(
                    df["high"], df["low"], df["close"], df["volume"], length=period
                )
                if mfi is not None:
                    return pd.DataFrame(
                        {f"mfi_{period}": (mfi - 50.0) / 50.0}, index=df.index
                    )
            return pd.DataFrame(index=df.index)

        self.register("rsi", compute_rsi, warmup=14)
        self.register("macd", compute_macd, warmup=35)
        self.register("roc", compute_roc, warmup=10)
        self.register("stoch", compute_stoch, warmup=14)
        self.register("williams_r", compute_williams_r, warmup=14)
        self.register("cci", compute_cci, warmup=20)
        self.register("mfi", compute_mfi, warmup=14)

        # --- Volatility indicators ---
        def compute_bbands(
            df: pd.DataFrame, period: int = 20, std: float = 2.0, **kwargs
        ) -> pd.DataFrame:
            close = df["close"] if "close" in df.columns else df["price"]
            if _HAS_PANDAS_TA:
                bb_df = ta.bbands(close, length=period, std=std)
                if bb_df is not None and not bb_df.empty:
                    result = pd.DataFrame(index=df.index)
                    cols = bb_df.columns.tolist()
                    # pandas-ta returns: BBL, BBM, BBU, BBB, BBP
                    for col in cols:
                        col_lower = col.lower()
                        if "bbb" in col_lower:
                            result[f"bb_width_{period}"] = bb_df[col] / 100.0
                        elif "bbp" in col_lower:
                            result[f"bb_pct_{period}"] = bb_df[col] - 0.5  # Center on 0
                    return result if not result.empty else pd.DataFrame(index=df.index)
            # Fallback
            sma = close.rolling(window=period).mean()
            std_val = close.rolling(window=period).std()
            upper = sma + std * std_val
            lower = sma - std * std_val
            width = (upper - lower) / sma
            pct = (close - lower) / (upper - lower).replace(0, np.nan)
            return pd.DataFrame(
                {f"bb_width_{period}": width, f"bb_pct_{period}": pct - 0.5},
                index=df.index,
            )

        def compute_atr(df: pd.DataFrame, period: int = 14, **kwargs) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close"]):
                return pd.DataFrame(index=df.index)
            if _HAS_PANDAS_TA:
                atr = ta.atr(df["high"], df["low"], df["close"], length=period)
                if atr is not None:
                    # Normalize by close price
                    close = df["close"]
                    return pd.DataFrame({f"atr_{period}": atr / close}, index=df.index)
            # Fallback
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift(1)).abs()
            low_close = (df["low"] - df["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return pd.DataFrame({f"atr_{period}": atr / df["close"]}, index=df.index)

        def compute_rolling_std(
            df: pd.DataFrame, period: int = 10, **kwargs
        ) -> pd.DataFrame:
            vol = df["returns"].rolling(window=period, min_periods=period).std()
            return pd.DataFrame({f"rvol_{period}": vol}, index=df.index)

        def compute_keltner(
            df: pd.DataFrame, period: int = 20, multiplier: float = 2.0, **kwargs
        ) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close"]):
                return pd.DataFrame(index=df.index)
            close = df["close"]
            ema = close.ewm(span=period, adjust=False).mean()
            # ATR for keltner
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift(1)).abs()
            low_close = (df["low"] - df["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            upper = ema + multiplier * atr
            lower = ema - multiplier * atr
            # Features: position within channel, channel width
            pct = (close - lower) / (upper - lower).replace(0, np.nan)
            width = (upper - lower) / ema
            return pd.DataFrame(
                {f"keltner_pct_{period}": pct - 0.5, f"keltner_width_{period}": width},
                index=df.index,
            )

        self.register("bbands", compute_bbands, warmup=20)
        self.register("atr", compute_atr, warmup=14)
        self.register("rolling_std", compute_rolling_std, warmup=30)
        self.register("keltner", compute_keltner, warmup=20)

        # --- Volume indicators ---
        def compute_vwap(df: pd.DataFrame, period: int = 20, **kwargs) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
                return pd.DataFrame(index=df.index)
            tp = (df["high"] + df["low"] + df["close"]) / 3
            rolling_tp_vol = (
                (tp * df["volume"]).rolling(window=period, min_periods=period).sum()
            )
            rolling_vol = df["volume"].rolling(window=period, min_periods=period).sum()
            vwap = rolling_tp_vol / rolling_vol.replace(0, np.nan)
            pct_dev = (df["close"] - vwap) / vwap
            return pd.DataFrame({"vwap_dev": pct_dev}, index=df.index)

        def compute_obv(df: pd.DataFrame, period: int = 20, **kwargs) -> pd.DataFrame:
            if "volume" not in df.columns:
                return pd.DataFrame(index=df.index)
            close = df["close"] if "close" in df.columns else df["price"]
            direction = np.sign(close.diff())
            obv = (
                (direction * df["volume"])
                .rolling(window=period, min_periods=period)
                .sum()
            )
            obv_roc = obv.pct_change(periods=5)
            return pd.DataFrame({"obv_roc": obv_roc}, index=df.index)

        def compute_volume_ratio(
            df: pd.DataFrame, period: int = 20, **kwargs
        ) -> pd.DataFrame:
            if "volume" not in df.columns:
                return pd.DataFrame(index=df.index)
            avg_vol = df["volume"].rolling(window=period).mean()
            ratio = df["volume"] / avg_vol.replace(0, np.nan)
            # Log ratio to handle extreme volume spikes
            log_ratio = np.log1p(ratio.clip(lower=1e-8) - 1)
            return pd.DataFrame({f"vol_ratio_{period}": log_ratio}, index=df.index)

        self.register("vwap", compute_vwap, warmup=20)
        self.register("obv", compute_obv, warmup=25)
        self.register("volume_ratio", compute_volume_ratio, warmup=20)

        # --- Microstructure / derived features ---
        def compute_high_low_range(
            df: pd.DataFrame, period: int = 10, **kwargs
        ) -> pd.DataFrame:
            if not all(c in df.columns for c in ["high", "low", "close"]):
                return pd.DataFrame(index=df.index)
            daily_range = (df["high"] - df["low"]) / df["close"]
            avg_range = daily_range.rolling(window=period).mean()
            return pd.DataFrame(
                {f"hl_range": daily_range, f"hl_range_avg_{period}": avg_range},
                index=df.index,
            )

        def compute_close_to_open_gap(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
            if not all(c in df.columns for c in ["open", "close"]):
                return pd.DataFrame(index=df.index)
            gap = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
            return pd.DataFrame({f"gap": gap}, index=df.index)

        def compute_return_zscore(
            df: pd.DataFrame, period: int = 20, **kwargs
        ) -> pd.DataFrame:
            mean = df["returns"].rolling(window=period).mean()
            std = df["returns"].rolling(window=period).std()
            zscore = (df["returns"] - mean) / std.replace(0, np.nan)
            return pd.DataFrame({f"ret_zscore_{period}": zscore}, index=df.index)

        self.register("high_low_range", compute_high_low_range, warmup=10)
        self.register("close_to_open_gap", compute_close_to_open_gap, warmup=1)
        self.register("return_zscore", compute_return_zscore, warmup=20)

        # --- Research-backed alpha indicators ---
        # Sources: Quantpedia, 100k-backtest study (Starks 2025),
        # SetupAlpha mean-reversion series, SSRN Hurst exponent papers.

        def compute_rsi2(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
            """Connors RSI-2: ultra-short 2-period RSI.
            One of the most well-documented mean-reversion signals.
            75%+ win rate in backtested equity strategies (Connors & Alvarez).
            Normalized to [-1, 1] range."""
            close = df["close"] if "close" in df.columns else None
            if close is None:
                return pd.DataFrame(index=df.index)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(2).mean()
            rsi2 = _safe_rsi(gain, loss)
            return pd.DataFrame({"rsi2": (rsi2 - 50.0) / 50.0}, index=df.index)

        def compute_connors_rsi(
            df: pd.DataFrame,
            rsi_period: int = 3,
            streak_period: int = 2,
            pctrank_period: int = 100,
            **kwargs,
        ) -> pd.DataFrame:
            """ConnorsRSI: composite of RSI(3) + streak RSI + percentile-rank
            of daily return.  Proven for short-term mean-reversion timing.
            Ref: Connors & Alvarez, 'Short Term Trading Strategies That Work'.
            Output normalized to [0, 1]."""
            close = df["close"] if "close" in df.columns else None
            if close is None:
                return pd.DataFrame(index=df.index)

            # Component 1: RSI of close (short period)
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
            rsi_val = _safe_rsi(gain, loss)

            # Component 2: streak length (consecutive up/down days)
            direction = np.sign(delta)
            streak = pd.Series(0.0, index=close.index)
            for i in range(1, len(direction)):
                if (
                    direction.iloc[i] == direction.iloc[i - 1]
                    and direction.iloc[i] != 0
                ):
                    streak.iloc[i] = streak.iloc[i - 1] + direction.iloc[i]
                else:
                    streak.iloc[i] = direction.iloc[i]
            # RSI of the streak
            s_delta = streak.diff()
            s_gain = s_delta.where(s_delta > 0, 0.0).rolling(streak_period).mean()
            s_loss = (-s_delta.where(s_delta < 0, 0.0)).rolling(streak_period).mean()
            streak_rsi = _safe_rsi(s_gain, s_loss)

            # Component 3: percentile rank of today's return
            ret = close.pct_change()
            pct_rank = (
                ret.rolling(pctrank_period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
                * 100
            )

            # Composite: equal-weighted average, normalized to [0, 1]
            crsi = (rsi_val + streak_rsi + pct_rank) / 3.0
            return pd.DataFrame({"connors_rsi": crsi / 100.0}, index=df.index)

        def compute_hurst(
            df: pd.DataFrame, period: int = 100, **kwargs
        ) -> pd.DataFrame:
            """Hurst exponent (rolling) via rescaled-range (R/S) analysis.
            H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.
            Ref: SSRN #3824032 (Sidhu et al.), proven as ML feature for regime detection."""
            returns = df["returns"].values
            n = len(returns)
            hurst_vals = np.full(n, np.nan)
            for i in range(period, n):
                ts = returns[i - period : i]
                mean_ts = np.mean(ts)
                deviations = np.cumsum(ts - mean_ts)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(ts, ddof=1)
                if s > 1e-12:
                    hurst_vals[i] = np.log(r / s) / np.log(period)
            return pd.DataFrame({"hurst": hurst_vals}, index=df.index)

        def compute_dma(df: pd.DataFrame, period: int = 10, **kwargs) -> pd.DataFrame:
            """Distance from Moving Average (DMA): percentage distance from
            short-term SMA.  Z-scored stretch from mean is one of the top
            mean-reversion features (SetupAlpha, 100k-backtest study).
            Positive = above MA, negative = below."""
            close = df["close"] if "close" in df.columns else None
            if close is None:
                return pd.DataFrame(index=df.index)
            sma = close.rolling(period).mean()
            dma = (close - sma) / sma
            return pd.DataFrame({f"dma_{period}": dma}, index=df.index)

        self.register("rsi2", compute_rsi2, warmup=5)
        self.register("connors_rsi", compute_connors_rsi, warmup=105)
        self.register("hurst", compute_hurst, warmup=100)
        self.register("dma", compute_dma, warmup=10)

        # --- Calendar features ---
        def compute_calendar(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
            idx = df.index
            result = pd.DataFrame(index=idx)
            # Day of week: 0=Monday, 4=Friday; encode as sin/cos for cyclicity
            dow = idx.dayofweek
            result["cal_dow_sin"] = np.sin(2 * np.pi * dow / 5)
            result["cal_dow_cos"] = np.cos(2 * np.pi * dow / 5)
            # Month: encode as sin/cos
            month = idx.month
            result["cal_month_sin"] = np.sin(2 * np.pi * month / 12)
            result["cal_month_cos"] = np.cos(2 * np.pi * month / 12)
            return result

        self.register("calendar", compute_calendar, warmup=0)

        # --- External data features ---
        def compute_external_close(
            df: pd.DataFrame,
            feature_name: str = None,
            external_data: dict = None,
            **kwargs,
        ) -> pd.DataFrame:
            """Inject an external series (e.g., VIX close) as a feature."""
            if external_data is None or feature_name is None:
                return pd.DataFrame(index=df.index)
            if feature_name not in external_data:
                return pd.DataFrame(index=df.index)
            series = external_data[feature_name]
            # Align to primary index and forward-fill
            aligned = series.reindex(df.index, method="ffill")
            return pd.DataFrame({feature_name: aligned}, index=df.index)

        def compute_external_returns(
            df: pd.DataFrame,
            feature_name: str = None,
            external_data: dict = None,
            **kwargs,
        ) -> pd.DataFrame:
            """Inject log returns of an external series as a feature."""
            if external_data is None or feature_name is None:
                return pd.DataFrame(index=df.index)
            if feature_name not in external_data:
                return pd.DataFrame(index=df.index)
            series = external_data[feature_name]
            aligned = series.reindex(df.index, method="ffill")
            log_ret = np.log(aligned / aligned.shift(1))
            return pd.DataFrame({f"{feature_name}_ret": log_ret}, index=df.index)

        self.register("external_close", compute_external_close, warmup=0)
        self.register("external_returns", compute_external_returns, warmup=1)

        # --- Cross-asset computed indicators ---
        def compute_vix_term_slope(
            df: pd.DataFrame, external_data: dict = None, **kwargs
        ) -> pd.DataFrame:
            """VIX term structure slope: VIX3M - VIX spot.
            Positive = contango (normal), negative = backwardation (fear)."""
            if external_data is None:
                return pd.DataFrame(index=df.index)
            # Keys match the YAML config name (e.g. "vix_close"), NOT
            # the auto-generated "ext_..." prefix used when name is None.
            vix = external_data.get("vix_close")
            if vix is None:
                vix = external_data.get("ext_vix_close")
            vix3m = external_data.get("vix3m_close")
            if vix3m is None:
                vix3m = external_data.get("ext_vix3m_close")
            if vix is None or vix3m is None:
                return pd.DataFrame(index=df.index)
            vix_a = vix.reindex(df.index, method="ffill")
            vix3m_a = vix3m.reindex(df.index, method="ffill")
            slope = vix3m_a - vix_a
            return pd.DataFrame({"vix_term_slope": slope}, index=df.index)

        def compute_spread(
            df: pd.DataFrame,
            series_a: str = None,
            series_b: str = None,
            external_data: dict = None,
            **kwargs,
        ) -> pd.DataFrame:
            """Spread between two external series (a - b).
            E.g., US 2Y yield - Japan 10Y yield for rate differential."""
            if external_data is None or series_a is None or series_b is None:
                return pd.DataFrame(index=df.index)
            a_data = external_data.get(series_a)
            b_data = external_data.get(series_b)
            if a_data is None or b_data is None:
                return pd.DataFrame(index=df.index)
            a = a_data.reindex(df.index, method="ffill")
            b = b_data.reindex(df.index, method="ffill")
            spread = a - b
            name = (
                f"spread_{series_a.replace('ext_', '')}_{series_b.replace('ext_', '')}"
            )
            return pd.DataFrame({name: spread}, index=df.index)

        def compute_ratio(
            df: pd.DataFrame,
            series_a: str = None,
            series_b: str = None,
            external_data: dict = None,
            **kwargs,
        ) -> pd.DataFrame:
            """Ratio between two external series (a / b).
            E.g., copper/gold ratio as a global growth indicator."""
            if external_data is None or series_a is None or series_b is None:
                return pd.DataFrame(index=df.index)
            a_data = external_data.get(series_a)
            b_data = external_data.get(series_b)
            if a_data is None or b_data is None:
                return pd.DataFrame(index=df.index)
            a = a_data.reindex(df.index, method="ffill")
            b = b_data.reindex(df.index, method="ffill").replace(0, np.nan)
            ratio = a / b
            name = (
                f"ratio_{series_a.replace('ext_', '')}_{series_b.replace('ext_', '')}"
            )
            return pd.DataFrame({name: ratio}, index=df.index)

        self.register("vix_term_slope", compute_vix_term_slope, warmup=0)
        self.register("spread", compute_spread, warmup=0)
        self.register("ratio", compute_ratio, warmup=0)

        # --- COT (Commitment of Traders) indicator ---
        def compute_cot_zscore(
            df: pd.DataFrame,
            external_data: dict = None,
            currency: str = "jpy",
            **kwargs,
        ) -> pd.DataFrame:
            """COT net non-commercial positioning, z-scored for stationarity.
            Weekly data forward-filled to daily frequency."""
            key = f"cot_{currency.lower()}_net"
            if external_data is None or key not in external_data:
                # Try with ext_ prefix
                key = f"ext_cot_{currency.lower()}_net_close"
                if external_data is None or key not in external_data:
                    return pd.DataFrame(index=df.index)
            cot = external_data[key].reindex(df.index, method="ffill")
            # Rolling z-score: 52 weeks ~ 1 year of weekly data
            rolling_mean = cot.rolling(252, min_periods=60).mean()
            rolling_std = cot.rolling(252, min_periods=60).std()
            cot_z = (cot - rolling_mean) / rolling_std.replace(0, np.nan)
            return pd.DataFrame(
                {f"cot_{currency.lower()}_zscore": cot_z}, index=df.index
            )

        self.register("cot_zscore", compute_cot_zscore, warmup=252)


# Global singleton registry
_GLOBAL_REGISTRY = None


def get_registry() -> FeatureRegistry:
    """Get or create the global feature registry singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = FeatureRegistry()
    return _GLOBAL_REGISTRY


# ---------------------------------------------------------------------------
# Feature Config — loads YAML and resolves per-model overrides
# ---------------------------------------------------------------------------


class FeatureConfig:
    """
    Loads feature engineering configuration from YAML.
    Resolves a 4-level override hierarchy:

        Level 1: defaults              (all tickers, all models)
        Level 2: model_overrides       (all tickers, specific model)
        Level 3: ticker_overrides      (specific ticker, all models)
        Level 4: ticker_overrides      (specific ticker:model combo)

    Each override level can: include_defaults, exclude groups, add additional indicators.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        ticker: Optional[str] = None,
        config_dict: Optional[dict] = None,
    ):
        """
        Args:
            config_path: Path to feature_config.yaml (auto-detected if None)
            model_name: Model name for per-model overrides (optional)
            ticker: Ticker symbol for per-ticker overrides (optional)
            config_dict: Direct config dict (overrides config_path)
        """
        self.model_name = model_name
        self.ticker = ticker
        self._raw_config = {}
        self._indicators: list[tuple[str, dict]] = []  # (indicator_name, params)
        self._external_configs: list[dict] = []
        self._warmup_minimum: int = 60
        self._config_hash: str = ""

        if config_dict is not None:
            self._raw_config = config_dict
        elif config_path is not None:
            self._load_from_file(config_path)
        else:
            self._load_from_file(self._find_config_file())

        self._resolve_config()

    def _find_config_file(self) -> Optional[str]:
        """Auto-detect feature_config.yaml location."""
        search_paths = []
        try:
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            search_paths = [
                project_root / "feature_config.yaml",
                current_dir / "feature_config.yaml",
            ]
        except NameError:
            search_paths = [Path("feature_config.yaml")]

        for p in search_paths:
            if p.exists():
                return str(p)
        return None

    def _load_from_file(self, path: Optional[str]) -> None:
        """Load config from YAML file."""
        if path is None:
            return
        try:
            with open(path, "r") as f:
                self._raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load feature config from {path}: {e}")
            self._raw_config = {}

    def _resolve_config(self) -> None:
        """
        Resolve the final indicator list using the 4-level override hierarchy:
            Level 1: defaults              (all tickers, all models)
            Level 2: model_overrides       (all tickers, specific model)
            Level 3: ticker_overrides      (specific ticker, all models)
            Level 4: ticker_overrides      (specific ticker:model combo, e.g. "NVDA:xgb_optimized")
        """
        fe_config = self._raw_config.get("feature_engineering", {})
        defaults = fe_config.get("defaults", {})
        warmup_cfg = self._raw_config.get("warmup_periods", {})
        self._warmup_minimum = warmup_cfg.get("minimum", 60)

        # Collect override layers (applied in order: model → ticker → ticker:model)
        override_layers = []

        # Level 2: per-model override
        model_overrides = fe_config.get("model_overrides", {})
        if self.model_name and self.model_name in model_overrides:
            override_layers.append(("model", model_overrides[self.model_name]))

        # Level 3: per-ticker override
        ticker_overrides = fe_config.get("ticker_overrides", {})
        if self.ticker and self.ticker in ticker_overrides:
            override_layers.append(("ticker", ticker_overrides[self.ticker]))

        # Level 4: per-ticker:model combo override (most specific)
        combo_key = (
            f"{self.ticker}:{self.model_name}"
            if self.ticker and self.model_name
            else None
        )
        if combo_key and combo_key in ticker_overrides:
            override_layers.append(("ticker:model", ticker_overrides[combo_key]))

        # Merge overrides: later layers win for exclude/include, additionals accumulate
        merged_exclude = set()
        merged_additional = {}
        include_defaults = True

        for layer_name, layer_cfg in override_layers:
            if "include_defaults" in layer_cfg:
                include_defaults = layer_cfg["include_defaults"]
            for grp in layer_cfg.get("exclude", []):
                merged_exclude.add(grp)
            layer_add = layer_cfg.get("additional", {})
            if layer_add:
                # Deep-merge additional: per-indicator lists accumulate
                for grp_name, grp_items in layer_add.items():
                    if grp_name not in merged_additional:
                        merged_additional[grp_name] = grp_items
                    elif isinstance(grp_items, dict) and isinstance(
                        merged_additional[grp_name], dict
                    ):
                        # Merge sub-dicts (e.g., momentum: {stoch: [...], cci: [...]})
                        for k, v in grp_items.items():
                            merged_additional[grp_name][k] = v
                    else:
                        merged_additional[grp_name] = grp_items

            # Handle direct external overrides at the override level
            layer_ext = layer_cfg.get("external", {})
            if layer_ext:
                if "external" not in merged_additional:
                    merged_additional["external"] = {}
                if isinstance(merged_additional["external"], dict):
                    merged_additional["external"].update(layer_ext)

        # Build indicator list
        indicators = []

        if include_defaults:
            indicators.extend(self._parse_indicator_group(defaults, merged_exclude))

        if merged_additional:
            indicators.extend(self._parse_indicator_group(merged_additional, set()))

        # Deduplicate by (name, params) tuple
        seen = set()
        unique_indicators = []
        for name, params in indicators:
            key = (name, json.dumps(params, sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique_indicators.append((name, params))

        self._indicators = unique_indicators

        # Extract external data configs from defaults (respecting excludes)
        self._external_configs = []
        external_section = defaults.get("external", {})
        if "external" not in merged_exclude and external_section:
            for ext_name, ext_cfg in external_section.items():
                if isinstance(ext_cfg, dict):
                    self._external_configs.append(
                        {
                            "name": ext_name,
                            "ticker": ext_cfg.get("ticker", ""),
                            "column": ext_cfg.get("column", "close"),
                        }
                    )

        # Also extract from merged additional external
        additional_external = merged_additional.get("external", {})
        if additional_external and isinstance(additional_external, dict):
            for ext_name, ext_cfg in additional_external.items():
                if isinstance(ext_cfg, dict):
                    # Avoid duplicates by name
                    existing_names = {e["name"] for e in self._external_configs}
                    if ext_name not in existing_names:
                        self._external_configs.append(
                            {
                                "name": ext_name,
                                "ticker": ext_cfg.get("ticker", ""),
                                "column": ext_cfg.get("column", "close"),
                            }
                        )

        # Compute config hash for metadata
        config_str = json.dumps(
            {
                "indicators": [(n, p) for n, p in self._indicators],
                "external": self._external_configs,
            },
            sort_keys=True,
        )
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _parse_indicator_group(
        self, group: dict, excluded_groups: set
    ) -> list[tuple[str, dict]]:
        """
        Parse a group of indicator configs into (name, params) tuples.
        Handles both flat and nested group structures:

        Flat:    lagged_returns: {lags: [1,2,3,4,5]}
        Nested:  trend:
                   sma: [{period: 10}, {period: 20}]
                   ema: [{period: 10}]
        """
        registry = get_registry()
        registered_names = set(registry.list_indicators())
        indicators = []

        for group_name, group_items in group.items():
            if group_name in excluded_groups:
                continue
            if group_name == "external":
                continue  # Handled separately

            if group_name in registered_names:
                # Direct indicator reference (e.g., lagged_returns: {...})
                if isinstance(group_items, dict):
                    indicators.append((group_name, group_items))
                elif isinstance(group_items, list):
                    for item in group_items:
                        if isinstance(item, dict):
                            indicators.append((group_name, item))
                        else:
                            indicators.append((group_name, {}))
                else:
                    indicators.append((group_name, {}))
            elif isinstance(group_items, dict):
                # Check if this is a category group (e.g., trend: {sma: [...], ema: [...]})
                # by seeing if any keys are registered indicator names
                has_sub_indicators = any(k in registered_names for k in group_items)
                if has_sub_indicators:
                    # Recurse into the sub-group
                    for sub_name, sub_items in group_items.items():
                        if sub_name in excluded_groups:
                            continue
                        if isinstance(sub_items, list):
                            for item in sub_items:
                                if isinstance(item, dict):
                                    indicators.append((sub_name, item))
                                else:
                                    indicators.append((sub_name, {}))
                        elif isinstance(sub_items, dict):
                            indicators.append((sub_name, sub_items))
                        else:
                            indicators.append((sub_name, {}))
                else:
                    # Treat as indicator with params dict
                    indicators.append((group_name, group_items))
            elif isinstance(group_items, list):
                # List of configs for an unregistered name (treat as indicator)
                for item in group_items:
                    if isinstance(item, dict):
                        indicators.append((group_name, item))
                    else:
                        indicators.append((group_name, {}))
            else:
                indicators.append((group_name, {}))

        return indicators

    @property
    def indicators(self) -> list[tuple[str, dict]]:
        """Return the resolved list of (indicator_name, params) tuples."""
        return self._indicators

    @property
    def external_configs(self) -> list[dict]:
        """Return external data configurations."""
        return self._external_configs

    @property
    def config_hash(self) -> str:
        """Return a hash of the resolved config for metadata."""
        return self._config_hash

    @property
    def warmup_minimum(self) -> int:
        """Return minimum warmup periods."""
        return self._warmup_minimum

    def get_max_warmup(self, registry: Optional[FeatureRegistry] = None) -> int:
        """Calculate the maximum warmup period across all configured indicators."""
        if registry is None:
            registry = get_registry()

        max_warmup = self._warmup_minimum
        for name, params in self._indicators:
            indicator_warmup = registry.get_warmup(name)
            # Some indicators have period-dependent warmup
            period = params.get("period", 0)
            slow = params.get("slow", 0)
            effective_warmup = max(indicator_warmup, period, slow)
            max_warmup = max(max_warmup, effective_warmup)

        return max_warmup

    def to_dict(self) -> dict:
        """Serialize the resolved config to a dict (for saving metadata)."""
        return {
            "model_name": self.model_name,
            "ticker": self.ticker,
            "indicators": [(name, params) for name, params in self._indicators],
            "external": self._external_configs,
            "config_hash": self._config_hash,
            "n_features": len(self._indicators),
            "warmup_minimum": self._warmup_minimum,
        }

    def describe(self) -> str:
        """Return a human-readable summary of this config (for logging)."""
        # Count indicators by category
        from collections import Counter

        categories = Counter()
        for name, _ in self._indicators:
            if name == "lagged_returns":
                categories["lags"] += 1
            elif name in ("sma", "ema"):
                categories["trend"] += 1
            elif name in (
                "rsi",
                "rsi2",
                "macd",
                "roc",
                "stoch",
                "williams_r",
                "cci",
                "mfi",
                "connors_rsi",
            ):
                categories["momentum"] += 1
            elif name in ("bbands", "atr", "rolling_std", "keltner"):
                categories["volatility"] += 1
            elif name in ("vwap", "obv", "volume_ratio"):
                categories["volume"] += 1
            elif name in ("hurst", "dma"):
                categories["alpha"] += 1
            elif name in (
                "calendar",
                "high_low_range",
                "close_to_open_gap",
                "return_zscore",
                "vix_term_slope",
                "ratio",
                "spread",
            ):
                categories["derived"] += 1
            elif name.startswith("external"):
                categories["external"] += 1
            else:
                categories["other"] += 1

        n_ext = len(self._external_configs)
        if n_ext:
            categories["external"] = n_ext

        parts = [f"{v} {k}" for k, v in sorted(categories.items())]
        breakdown = ", ".join(parts)
        return f"{len(self._indicators)} indicators (hash: {self._config_hash}) [{breakdown}]"


# ---------------------------------------------------------------------------
# Feature Engine — the main computation engine
# ---------------------------------------------------------------------------


class FeatureEngine:
    """
    Computes features from OHLCV data + external data.
    Used identically in backtesting and live trading.
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None):
        self.registry = registry or get_registry()

    def compute_features(
        self,
        data: pd.DataFrame,
        feature_config: FeatureConfig,
        external_data: Optional[dict[str, pd.Series]] = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Compute all configured features and append to the DataFrame.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)
                  plus 'price', 'returns', 'direction'
            feature_config: Resolved feature configuration
            external_data: Dict mapping feature_name -> pd.Series for external data

        Returns:
            (augmented_df, feature_column_names) — same pattern as create_lagged_features()
            The augmented_df has NaN rows from warmup dropped.
        """
        if external_data is None:
            external_data = {}

        all_feature_cols = []

        for indicator_name, params in feature_config.indicators:
            try:
                func = self.registry.get(indicator_name)

                # Pass external_data to all indicators.  Only those whose
                # signature accepts it will use it (external_close,
                # external_returns, vix_term_slope, spread, ratio, cot_zscore).
                # Others absorb it harmlessly via **kwargs.
                call_params = dict(params)
                call_params["external_data"] = external_data

                result_df = func(data, **call_params)

                if result_df is not None and not result_df.empty:
                    for col in result_df.columns:
                        if col not in data.columns:
                            data[col] = result_df[col]
                            all_feature_cols.append(col)

            except KeyError:
                print(f"Warning: Unknown indicator '{indicator_name}', skipping")
            except Exception as e:
                print(f"Warning: Failed to compute indicator '{indicator_name}': {e}")

        # Add external data as direct features (for simple external_close configs)
        for ext_cfg in feature_config.external_configs:
            ext_name = ext_cfg["name"]
            if ext_name in external_data and ext_name not in data.columns:
                aligned = external_data[ext_name].reindex(data.index, method="ffill")
                data[ext_name] = aligned
                all_feature_cols.append(ext_name)

        # Sanitize features: replace inf, drop all-NaN columns, then drop NaN rows
        if all_feature_cols:
            _inf_count = np.isinf(data[all_feature_cols]).sum().sum()
            if _inf_count > 0:
                print(
                    f"Warning: Replaced {_inf_count} inf values with NaN in feature columns"
                )
            data[all_feature_cols] = data[all_feature_cols].replace(
                [np.inf, -np.inf], np.nan
            )

            # Defense-in-depth: Drop feature columns that are entirely NaN.
            # This prevents a single bad indicator (e.g., volume features on forex
            # data with all-zero volume) from wiping out the entire DataFrame via
            # dropna(subset=...). Without this, one all-NaN column causes every row
            # to be dropped, leaving an empty DataFrame and a fatal sys.exit(1).
            all_nan_cols = [col for col in all_feature_cols if data[col].isna().all()]
            if all_nan_cols:
                print(
                    f"Warning: Dropping {len(all_nan_cols)} all-NaN feature columns "
                    f"(likely zero-volume or missing data): {all_nan_cols}"
                )
                data = data.drop(columns=all_nan_cols)
                all_feature_cols = [
                    c for c in all_feature_cols if c not in all_nan_cols
                ]

            if all_feature_cols:
                data = data.dropna(subset=all_feature_cols).copy()

        return data, all_feature_cols

    def compute_live_features(
        self,
        data: pd.DataFrame,
        feature_config: FeatureConfig,
        external_data: Optional[dict[str, pd.Series]] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Compute features for live prediction. Returns the LAST row.
        This is THE SAME CODE PATH as backtesting — no divergence.

        Args:
            data: DataFrame with OHLCV data (typically last ~200-300 bars)
            feature_config: Resolved feature configuration
            external_data: External data dict

        Returns:
            (features_array shape (1, n_features), feature_column_names)
        """
        augmented, feature_cols = self.compute_features(
            data, feature_config, external_data
        )

        if augmented.empty or not feature_cols:
            return np.array([]).reshape(1, 0), feature_cols

        # Take the last row
        last_row = augmented[feature_cols].iloc[-1:].values
        return last_row, feature_cols

    def get_feature_names(self, feature_config: FeatureConfig) -> list[str]:
        """
        Predict the feature column names that will be produced by a config.
        Useful for metadata validation without computing actual features.

        Note: This is an approximation — some indicators produce multiple columns
        whose names depend on parameters. For exact names, use compute_features().
        """
        names = []
        for indicator_name, params in feature_config.indicators:
            # Best-effort name prediction
            if indicator_name == "lagged_returns":
                lags = params.get("lags", [1, 2, 3, 4, 5])
                names.extend([f"lag_{lag}" for lag in lags])
            elif indicator_name == "macd":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                names.extend(
                    [
                        f"macd_{fast}_{slow}",
                        f"macd_hist_{fast}_{slow}",
                        f"macd_signal_{fast}_{slow}",
                    ]
                )
            elif indicator_name == "bbands":
                period = params.get("period", 20)
                names.extend([f"bb_width_{period}", f"bb_pct_{period}"])
            elif indicator_name == "stoch":
                k = params.get("k", 14)
                names.extend([f"stoch_k_{k}", f"stoch_d_{k}"])
            elif indicator_name == "keltner":
                period = params.get("period", 20)
                names.extend([f"keltner_pct_{period}", f"keltner_width_{period}"])
            elif indicator_name == "high_low_range":
                period = params.get("period", 10)
                names.extend(["hl_range", f"hl_range_avg_{period}"])
            elif indicator_name == "calendar":
                names.extend(
                    ["cal_dow_sin", "cal_dow_cos", "cal_month_sin", "cal_month_cos"]
                )
            elif indicator_name == "obv":
                names.append("obv_roc")
            elif indicator_name == "vwap":
                names.append("vwap_dev")
            elif indicator_name == "close_to_open_gap":
                names.append("gap")
            elif indicator_name.startswith("external_"):
                pass  # External names are dynamic
            else:
                # Default: indicator_period
                period = params.get("period", "")
                if period:
                    names.append(f"{indicator_name}_{period}")
                else:
                    names.append(indicator_name)

        # Add external feature names
        for ext_cfg in feature_config.external_configs:
            names.append(ext_cfg["name"])

        return names


def save_feature_metadata(
    feature_names: list[str],
    feature_config: FeatureConfig,
    symbol: str,
    model_type: str,
    save_dir: str,
) -> Path:
    """
    Save feature metadata JSON alongside scaler for deployment validation.

    Args:
        feature_names: Ordered list of feature column names
        feature_config: The config used to generate features
        symbol: Trading symbol
        model_type: Model type string
        save_dir: Directory to save metadata

    Returns:
        Path to saved metadata file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "config_hash": feature_config.config_hash,
        "model_name": feature_config.model_name,
        "model_type": model_type,
        "symbol": symbol,
        "config": feature_config.to_dict(),
    }

    meta_path = save_path / f"feature_meta_{model_type}_{symbol}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved feature metadata ({len(feature_names)} features) to {meta_path}")
    return meta_path


def load_feature_metadata(
    symbol: str,
    model_type: str,
    load_dir: str,
) -> Optional[dict]:
    """
    Load feature metadata JSON for validation.

    Returns:
        Metadata dict or None if not found
    """
    load_path = Path(load_dir)
    search_patterns = [
        f"feature_meta_{model_type}_{symbol}.json",
        f"feature_meta_{symbol}.json",
    ]

    for pattern in search_patterns:
        meta_path = load_path / pattern
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load feature metadata from {meta_path}: {e}")

    return None
