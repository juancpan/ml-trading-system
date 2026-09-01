"""
Weekly Gate Engine -- pure-function module for ML-gated portfolio weight adjustment.

Takes HRP portfolio weights and ML signals, returns gated weights with
proportional redistribution. No side effects, no I/O, no external dependencies.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def apply_gates(
    base_weights: Dict[str, float],
    signals: Dict[str, int],
    max_weight: Optional[float] = None,
    min_active_tickers: Optional[int] = None,
) -> Dict[str, float]:
    """Apply ML signal gates to HRP portfolio weights.

    Parameters
    ----------
    base_weights : dict
        {ticker: weight} from HRP optimisation. Weights should sum to ~1.0.
    signals : dict
        {ticker: signal} where signal > 0 means gate open, <= 0 means gate closed.
        Missing tickers default to gate open.
    max_weight : float, optional
        Maximum allowed weight for any single ticker after redistribution.
        Excess above cap goes to cash (not redistributed further).
    min_active_tickers : int, optional
        If the number of active (gate-open) tickers falls below this floor,
        keep original weights for active tickers only (no scaling up).

    Returns
    -------
    dict
        {ticker: weight} for ALL tickers in base_weights. Gated-out tickers = 0.0.
    """
    # Classify tickers: active (signal > 0) vs gated out (signal <= 0)
    # Missing signals default to gate open
    active = {}
    gated_out = {}
    for ticker, weight in base_weights.items():
        signal = signals.get(ticker, 1)  # default: gate open
        if signal > 0:
            active[ticker] = weight
        else:
            gated_out[ticker] = weight

    # Build result with all tickers, gated-out = 0.0
    result = {ticker: 0.0 for ticker in base_weights}

    # All gates closed: return all zeros
    if not active:
        return result

    # Min-active-tickers floor: if too few active, keep original weights (no scale-up)
    if min_active_tickers is not None and len(active) < min_active_tickers:
        for ticker, weight in active.items():
            result[ticker] = weight
        return result

    # Scale active weights proportionally so they sum to 1.0
    active_sum = sum(active.values())
    if active_sum > 0:
        scale = 1.0 / active_sum
        for ticker, weight in active.items():
            result[ticker] = weight * scale

    # Apply concentration cap if specified
    if max_weight is not None:
        result = _cap_gated_weights(result, max_weight)

    return result


def _cap_gated_weights(
    weights: Dict[str, float],
    max_weight: float,
    max_iterations: int = 50,
) -> Dict[str, float]:
    """Iteratively cap weights, sending overflow to cash (not redistributed).

    Unlike HRP's _cap_weights which redistributes excess to below-cap positions,
    this function sends overflow to cash to prevent concentration creep.

    Parameters
    ----------
    weights : dict
        {ticker: weight}. May sum to <= 1.0.
    max_weight : float
        Maximum allowed weight per ticker.
    max_iterations : int
        Safety limit on iterations.

    Returns
    -------
    dict
        Capped weights. Sum may be < 1.0 (difference is implicit cash).
    """
    capped = dict(weights)
    for _ in range(max_iterations):
        any_over = False
        for ticker, w in capped.items():
            if w > max_weight + 1e-12:
                capped[ticker] = max_weight
                any_over = True
        if not any_over:
            break
    return capped


def compute_gated_allocation(
    hrp_weights: Dict[str, float],
    model_signals: Dict[str, int],
    max_weight: float = 0.25,
    min_active_tickers: int = 3,
) -> Dict:
    """High-level wrapper: apply gates and return a summary allocation dict.

    Parameters
    ----------
    hrp_weights : dict
        {ticker: weight} from HRP optimisation.
    model_signals : dict
        {ticker: signal} from ML model predictions.
    max_weight : float
        Maximum allowed weight per ticker (default 0.25).
    min_active_tickers : int
        Minimum number of active tickers before we fall back to no-scale mode.

    Returns
    -------
    dict
        Keys: gated_weights, active_tickers, gated_out_tickers,
              cash_weight, n_active, n_gated_out
    """
    gated_weights = apply_gates(
        hrp_weights,
        model_signals,
        max_weight=max_weight,
        min_active_tickers=min_active_tickers,
    )

    active_tickers = [t for t, w in gated_weights.items() if w > 0]
    gated_out_tickers = [t for t, w in gated_weights.items() if w == 0.0]
    invested_weight = sum(gated_weights.values())
    cash_weight = max(0.0, 1.0 - invested_weight)

    result = {
        "gated_weights": gated_weights,
        "active_tickers": active_tickers,
        "gated_out_tickers": gated_out_tickers,
        "cash_weight": cash_weight,
        "n_active": len(active_tickers),
        "n_gated_out": len(gated_out_tickers),
    }

    logger.info(
        "Gated allocation: %d active, %d gated out, %.1f%% cash | active=%s | gated=%s",
        result["n_active"],
        result["n_gated_out"],
        cash_weight * 100,
        active_tickers,
        gated_out_tickers,
    )

    return result
