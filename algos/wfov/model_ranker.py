"""
Automated model ranking and categorization system for WFOV results.

Categorizes models into tiers based on statistical significance and performance.
Provides recommendations while maintaining flexibility for manual overrides.

Author: jcp
Date: 2025-12-03
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import sys

from algos.wfov.statistical_tests import apply_benjamini_hochberg_fdr


class ModelRanker:
    """
    Automated model ranking and categorization based on WFOV results.

    Supports 3 trading profiles:
    - A (MAXIMUM RETURNS): Screen garbage, deploy highest Sharpe, tolerate regime risk
    - B (RISK-ADJUSTED): Balance returns vs risk, moderate regime tolerance
    - C (INSTITUTIONAL): Conservative, strict requirements, low regime tolerance

    Categorizes models into tiers:
    - TIER 1 (DEPLOY): High performance, statistically significant
    - TIER 2 (REVIEW): Good performance but with caveats
    - TIER 3 (REJECT): Likely noise or severely flawed

    Provides flexibility for manual overrides based on trader insights.
    """

    # Profile-based threshold presets
    PROFILES = {
        "A": {  # Maximum Returns (Aggressive)
            "name": "MAXIMUM RETURNS (Aggressive)",
            "p_value_threshold": 0.15,
            "min_sharpe_threshold": 0.25,
            "regime_dependency_threshold": 5.0,
            "deflated_sharpe_threshold": 0.2,
            "max_drawdown_threshold": 0.35,  # Tolerate high DD
            "min_hit_ratio": 0.45,
            "variance_penalty_threshold": 0.7,  # High tolerance for variance
            "description": "Deploy highest Sharpe among statistically significant models. Tolerate regime risk (you can monitor and switch fast).",
        },
        "B": {  # Risk-Adjusted (Balanced)
            "name": "RISK-ADJUSTED (Balanced)",
            "p_value_threshold": 0.05,
            "min_sharpe_threshold": 0.4,
            "regime_dependency_threshold": 2.5,
            "deflated_sharpe_threshold": 0.4,
            "max_drawdown_threshold": 0.25,
            "min_hit_ratio": 0.50,
            "variance_penalty_threshold": 0.5,
            "description": "Balance returns and risk. Moderate tolerance for regime dependency. Prefer consistent performers.",
        },
        "C": {  # Institutional (Conservative)
            "name": "INSTITUTIONAL (Conservative)",
            "p_value_threshold": 0.01,
            "min_sharpe_threshold": 0.5,
            "regime_dependency_threshold": 1.5,
            "deflated_sharpe_threshold": 0.5,
            "max_drawdown_threshold": 0.20,
            "min_hit_ratio": 0.52,
            "variance_penalty_threshold": 0.3,
            "description": "Strict requirements for institutional deployment. Low tolerance for regime dependency or high variance.",
        },
    }

    def __init__(
        self,
        profile: str = "A",
        p_value_threshold: Optional[float] = None,
        min_sharpe_threshold: Optional[float] = None,
        regime_dependency_threshold: Optional[float] = None,
        deflated_sharpe_threshold: Optional[float] = None,
        max_drawdown_threshold: Optional[float] = None,
        min_hit_ratio: Optional[float] = None,
    ):
        """
        Initialize model ranker with profile or custom thresholds.

        Args:
            profile: Trading profile 'A' (maximum returns), 'B' (risk-adjusted), 'C' (institutional)
            p_value_threshold: Override profile's p-value threshold
            min_sharpe_threshold: Override profile's min Sharpe
            regime_dependency_threshold: Override profile's regime tolerance
            deflated_sharpe_threshold: Override profile's deflated Sharpe threshold
            max_drawdown_threshold: Override profile's max drawdown
            min_hit_ratio: Override profile's min hit ratio

        Example:
            >>> ranker = ModelRanker(profile='A')  # Maximum returns
            >>> ranker = ModelRanker(profile='B')  # Risk-adjusted
            >>> ranker = ModelRanker(profile='C')  # Institutional
            >>> ranker = ModelRanker(profile='A', p_value_threshold=0.20)  # Custom override
        """
        if profile not in self.PROFILES:
            raise ValueError(f"Invalid profile: {profile}. Must be 'A', 'B', or 'C'")

        # Load profile defaults
        profile_config = self.PROFILES[profile]
        self.profile = profile
        self.profile_name = profile_config["name"]
        self.profile_description = profile_config["description"]

        # Use profile defaults or overrides
        self.p_value_threshold = (
            p_value_threshold or profile_config["p_value_threshold"]
        )
        self.min_sharpe_threshold = (
            min_sharpe_threshold or profile_config["min_sharpe_threshold"]
        )
        self.regime_dependency_threshold = (
            regime_dependency_threshold or profile_config["regime_dependency_threshold"]
        )
        self.deflated_sharpe_threshold = (
            deflated_sharpe_threshold or profile_config["deflated_sharpe_threshold"]
        )
        self.max_drawdown_threshold = (
            max_drawdown_threshold or profile_config["max_drawdown_threshold"]
        )
        self.min_hit_ratio = min_hit_ratio or profile_config["min_hit_ratio"]
        self.variance_penalty_threshold = profile_config["variance_penalty_threshold"]

    def categorize_model(self, summary: Dict) -> Dict:
        """
        Categorize a single model based on WFOV results.

        Args:
            summary: WFOV summary JSON dict

        Returns:
            Dict with:
            {
                'tier': 1|2|3,
                'tier_name': 'DEPLOY'|'REVIEW'|'REJECT',
                'recommendation': str,
                'flags': List[str],  # Warning flags
                'metrics': Dict,  # Key metrics extracted
                'score': float  # Overall score for ranking
            }
        """
        # Extract key metrics
        metadata = summary.get("metadata", {})
        perf = summary.get("performance_metrics", {})
        stat_rig = summary.get("statistical_rigor", {})
        regime_ana = summary.get("regime_analysis", {})

        model_name = metadata.get("model_name", "unknown")
        ticker = metadata.get("ticker", "unknown")
        iterations_failed = metadata.get("iterations_failed", 0)
        iterations_requested = metadata.get("iterations_requested", 1)

        # Core performance metrics
        mean_sharpe = perf.get("sharpe_ratio", {}).get("mean", np.nan)
        sharpe_std = perf.get("sharpe_ratio", {}).get("std", np.nan)
        hit_ratio = perf.get("hit_ratio", {}).get("mean", np.nan)
        max_dd = perf.get("max_drawdown", {}).get("mean", np.nan)

        # Buy-and-hold benchmark metrics
        mean_return = perf.get("annual_return", {}).get("mean", np.nan)
        mean_volatility = perf.get("annual_volatility", {}).get("mean", np.nan)
        bh_sharpe = perf.get("bh_sharpe_ratio", {}).get("mean", np.nan)
        bh_return = perf.get("bh_annual_return", {}).get("mean", np.nan)
        excess_sharpe = perf.get("excess_sharpe", {}).get("mean", np.nan)
        excess_return_val = perf.get("excess_return", {}).get("mean", np.nan)
        info_ratio = perf.get("information_ratio", {}).get("mean", np.nan)

        # Sortino ratio removed: cannot be computed correctly from per-iteration
        # Sharpe values alone. Would require per-bar downside returns across all
        # iterations, which are not stored in the summary JSON.
        sortino_ratio = np.nan
        calmar_ratio = (
            mean_return / abs(max_dd)
            if (
                not np.isnan(mean_return)
                and not np.isnan(max_dd)
                and abs(max_dd) > 1e-9
            )
            else np.nan
        )

        # Statistical rigor metrics
        sharpe_sig = stat_rig.get("sharpe_significance", {})
        p_value = sharpe_sig.get("p_value_two_tailed", 1.0)

        # Validate p-value (handle NaN, None, invalid range ONLY)
        # IMPORTANT: p_value = 0.0 is VALID (extremely significant, p < machine epsilon)
        # Only reject if p_value is None, NaN, or outside [0, 1] range
        if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
            p_value = 1.0  # Conservative: assume not significant
        elif p_value < 0.0 or p_value > 1.0:
            # Invalid p-value (must be in [0, 1] range, but 0.0 is valid!)
            p_value = 1.0

        ci_lower, ci_upper = sharpe_sig.get("confidence_interval", (np.nan, np.nan))

        deflated = stat_rig.get("deflated_sharpe", {})
        deflated_sharpe = deflated.get("deflated_sharpe", np.nan)

        # Regime analysis
        regime_dep = regime_ana.get("regime_dependency", {})
        is_regime_dependent = regime_dep.get("is_regime_dependent", False)
        perf_ratio = regime_dep.get("performance_ratio", 1.0)
        best_regime = regime_dep.get("best_regime", "unknown")
        worst_regime = regime_dep.get("worst_regime", "unknown")

        # Initialize flags
        flags = []

        # MC screening gate (if dual validation was performed)
        mc_screened_out = summary.get("_mc_screened_out", False)

        # Categorization logic
        tier = None
        tier_name = None
        recommendation = None

        # MC SCREENING GATE: reject if MC win_rate or failure_rate are poor
        if mc_screened_out:
            mc_wr = summary.get("_mc_win_rate", 0.0)
            mc_fr = summary.get("_mc_failure_rate", 0.0)
            tier = 3
            tier_name = "REJECT"
            reasons = []
            if mc_wr < 0.60:
                reasons.append(f"win_rate {mc_wr:.1%} < 60%")
            if mc_fr > 0.10:
                reasons.append(f"failure_rate {mc_fr:.1%} > 10%")
            recommendation = f"⛔ MC SCREENING FAIL: {'; '.join(reasons)}"
            flags.append("mc_screening_fail")

        # TIER 3: REJECT - Clear red flags
        elif p_value > 0.20:
            tier = 3
            tier_name = "REJECT"
            recommendation = (
                f"⛔ HIGH NOISE RISK: p-value {p_value:.3f} > 0.20 (likely random)"
            )
            flags.append("high_p_value")

        elif mean_sharpe < 0:
            tier = 3
            tier_name = "REJECT"
            recommendation = (
                f"⛔ NEGATIVE SHARPE: {mean_sharpe:.3f} (loses money on average)"
            )
            flags.append("negative_sharpe")

        elif hit_ratio < self.min_hit_ratio:
            tier = 3
            tier_name = "REJECT"
            recommendation = f"⛔ LOW WIN RATE: {hit_ratio:.1%} < {self.min_hit_ratio:.1%} (poor directional accuracy)"
            flags.append("low_win_rate")

        elif self.profile == "C" and abs(max_dd) > self.max_drawdown_threshold:
            # Hard reject on drawdown ONLY for Profile C (institutional)
            tier = 3
            tier_name = "REJECT"
            recommendation = f"⛔ EXCESSIVE DRAWDOWN: {max_dd:.1%} > {self.max_drawdown_threshold:.1%} (institutional limit)"
            flags.append("excessive_drawdown")

        # TIER 1: DEPLOY - Criteria depend on profile
        elif self._is_tier1(
            mean_sharpe,
            p_value,
            perf_ratio,
            abs(max_dd) if not np.isnan(max_dd) else 0,
            excess_sharpe=excess_sharpe,
        ):
            tier = 1
            tier_name = "DEPLOY"
            recommendation = f"🚀 STRONG PERFORMER: Sharpe {mean_sharpe:.2f}, significant (p={p_value:.4f})"

            # Profile-specific notes
            if self.profile == "A":
                # Profile A: Flag but don't disqualify
                if (
                    is_regime_dependent
                    and perf_ratio > self.regime_dependency_threshold
                ):
                    flags.append(f"regime_dependent_{best_regime}")
                    recommendation += f"\n   ⚠️  Monitor: {perf_ratio:.1f}x better in {best_regime} markets (set alerts for regime shifts)"

            elif self.profile == "B":
                # Profile B: More serious concern
                if (
                    is_regime_dependent
                    and perf_ratio > self.regime_dependency_threshold
                ):
                    flags.append(f"regime_dependent_{best_regime}")
                    recommendation += f"\n   ⚠️  Risk: {perf_ratio:.1f}x better in {best_regime} (consider diversification)"

            elif self.profile == "C":
                # Profile C: Should rarely happen due to strict thresholds
                if (
                    is_regime_dependent
                    and perf_ratio > self.regime_dependency_threshold
                ):
                    flags.append(f"regime_dependent_{best_regime}")
                    recommendation += f"\n   🚨 CRITICAL: {perf_ratio:.1f}x regime dependency (requires justification)"

            # Deflated Sharpe note (all profiles)
            if not np.isnan(deflated_sharpe) and deflated_sharpe < mean_sharpe * 0.7:
                flags.append("deflation_concern")
                if self.profile == "C":
                    recommendation += f"\n   📊 Conservative Estimate: Deflated Sharpe {deflated_sharpe:.2f} (expect this in live trading)"
                else:
                    recommendation += f"\n   ℹ️  Note: Deflated Sharpe {deflated_sharpe:.2f} (multiple testing adjustment)"

        # TIER 2: REVIEW - Moderate performance or profile-specific concerns
        elif (
            mean_sharpe > self.min_sharpe_threshold and p_value < self.p_value_threshold
        ):
            tier = 2
            tier_name = "REVIEW"
            recommendation = f"📊 MODERATE PERFORMER: Sharpe {mean_sharpe:.2f}, significance (p={p_value:.4f})"

            # Identify specific concerns
            if is_regime_dependent and perf_ratio > self.regime_dependency_threshold:
                flags.append(f"regime_dependent_{best_regime}")
                if self.profile == "A":
                    recommendation += f"\n   ⚠️  Regime Risk: {perf_ratio:.1f}x in {best_regime} (deploy if you can monitor daily)"
                else:
                    recommendation += f"\n   ⚠️  Regime Risk: {perf_ratio:.1f}x in {best_regime} (may fail in {worst_regime})"

            if sharpe_std > self.variance_penalty_threshold:
                flags.append("high_variance")
                recommendation += (
                    f"\n   ⚠️  High Variance: Sharpe std {sharpe_std:.2f} (inconsistent)"
                )

            if (
                not np.isnan(deflated_sharpe)
                and deflated_sharpe < self.deflated_sharpe_threshold
            ):
                flags.append("weak_deflated_sharpe")
                recommendation += f"\n   ⚠️  Weak Deflated Sharpe: {deflated_sharpe:.2f} < {self.deflated_sharpe_threshold}"

            if abs(max_dd) > self.max_drawdown_threshold:
                flags.append("high_drawdown")
                recommendation += f"\n   ⚠️  High Drawdown: {max_dd:.1%} > {self.max_drawdown_threshold:.1%}"

        else:
            # Default to TIER 2 if unclear
            tier = 2
            tier_name = "REVIEW"
            recommendation = f"🔍 MARGINAL: Sharpe {mean_sharpe:.2f}, p={p_value:.3f}"
            flags.append("marginal_significance")

        # Calculate composite score for ranking (profile-dependent)
        if self.profile == "A":  # Maximum Returns - prioritize observed Sharpe
            score = mean_sharpe  # Base score = observed Sharpe

            # Bonus for statistical significance (small)
            if p_value < 0.05:
                score += 0.2
            elif p_value < 0.10:
                score += 0.1

            # Small penalty for regime dependency (tolerate up to 5x)
            if is_regime_dependent and perf_ratio > 5.0:
                score -= 0.05 * (perf_ratio / 10.0)

            # Small penalty for high variance (tolerate up to 0.7)
            if sharpe_std > self.variance_penalty_threshold:
                score -= 0.05

            # Penalty for high failure rate (survivorship risk)
            failure_rate = iterations_failed / max(iterations_requested, 1)
            if failure_rate > 0.05:
                score -= failure_rate * 0.5

        elif self.profile == "B":  # Risk-Adjusted - balance Sharpe and deflated Sharpe
            # Use average of observed and deflated Sharpe
            if not np.isnan(deflated_sharpe) and deflated_sharpe > 0:
                score = (mean_sharpe + deflated_sharpe) / 2
            else:
                score = mean_sharpe * 0.8  # Discount if no deflated Sharpe

            # Moderate bonus for significance
            if p_value < 0.05:
                score += 0.3
            elif p_value < 0.10:
                score += 0.15

            # Moderate penalty for regime dependency (tolerate up to 2.5x)
            if is_regime_dependent and perf_ratio > self.regime_dependency_threshold:
                score -= 0.15 * (perf_ratio / 5.0)

            # Moderate penalty for high variance
            if sharpe_std > self.variance_penalty_threshold:
                score -= 0.15

            # Penalty for high drawdown
            if abs(max_dd) > self.max_drawdown_threshold:
                score -= 0.2

            # Penalty for high failure rate (survivorship risk)
            failure_rate = iterations_failed / max(iterations_requested, 1)
            if failure_rate > 0.05:
                score -= failure_rate * 0.5

        elif self.profile == "C":  # Institutional - prioritize deflated Sharpe
            # Use deflated Sharpe as primary metric
            if not np.isnan(deflated_sharpe) and deflated_sharpe > 0:
                score = deflated_sharpe
            else:
                score = mean_sharpe * 0.5  # Heavy discount if no deflated Sharpe

            # Large bonus for high significance
            if p_value < 0.01:
                score += 0.5
            elif p_value < 0.05:
                score += 0.2

            # Heavy penalty for regime dependency (low tolerance)
            if is_regime_dependent and perf_ratio > self.regime_dependency_threshold:
                score -= 0.3 * (perf_ratio / 3.0)

            # Heavy penalty for high variance
            if sharpe_std > self.variance_penalty_threshold:
                score -= 0.3

            # Heavy penalty for high drawdown
            if abs(max_dd) > self.max_drawdown_threshold:
                score -= 0.4

            # Penalty for high failure rate (survivorship risk)
            failure_rate = iterations_failed / max(iterations_requested, 1)
            if failure_rate > 0.05:
                score -= failure_rate * 0.5

        else:
            # Fallback: use observed Sharpe
            score = mean_sharpe

        # Ensure non-negative
        score = max(score, 0.0)

        return {
            "tier": tier,
            "tier_name": tier_name,
            "recommendation": recommendation,
            "flags": flags,
            "score": float(score),
            "metrics": {
                "model_name": model_name,
                "ticker": ticker,
                "mean_sharpe": float(mean_sharpe)
                if not np.isnan(mean_sharpe)
                else None,
                "deflated_sharpe": float(deflated_sharpe)
                if not np.isnan(deflated_sharpe)
                else None,
                "p_value": float(p_value),
                "ci_95": (
                    float(ci_lower) if not np.isnan(ci_lower) else None,
                    float(ci_upper) if not np.isnan(ci_upper) else None,
                ),
                "hit_ratio": float(hit_ratio) if not np.isnan(hit_ratio) else None,
                "sharpe_std": float(sharpe_std) if not np.isnan(sharpe_std) else None,
                "max_drawdown": float(max_dd) if not np.isnan(max_dd) else None,
                "mean_return": float(mean_return) if not np.isnan(mean_return) else 0.0,
                "mean_volatility": float(mean_volatility)
                if not np.isnan(mean_volatility)
                else 0.0,
                "sortino_ratio": float(sortino_ratio)
                if not np.isnan(sortino_ratio)
                else 0.0,
                "calmar_ratio": float(calmar_ratio)
                if not np.isnan(calmar_ratio)
                else 0.0,
                "bh_sharpe": float(bh_sharpe) if not np.isnan(bh_sharpe) else None,
                "bh_return": float(bh_return) if not np.isnan(bh_return) else None,
                "excess_sharpe": float(excess_sharpe)
                if not np.isnan(excess_sharpe)
                else None,
                "excess_return": float(excess_return_val)
                if not np.isnan(excess_return_val)
                else None,
                "information_ratio": float(info_ratio)
                if not np.isnan(info_ratio)
                else None,
                "regime_dependent": is_regime_dependent,
                "regime_ratio": float(perf_ratio),
                "best_regime": best_regime,
                "worst_regime": worst_regime,
            },
            "profile": self.profile,
        }

    def _is_tier1(
        self,
        sharpe: float,
        p_value: float,
        regime_ratio: float,
        drawdown: float,
        excess_sharpe: float = np.nan,
    ) -> bool:
        """
        Determine if model qualifies for TIER 1 based on profile.

        Args:
            sharpe: Mean Sharpe ratio
            p_value: Statistical p-value
            regime_ratio: Regime dependency ratio
            drawdown: Max drawdown (absolute value)
            excess_sharpe: Model Sharpe minus B&H Sharpe (alpha gate)

        Returns:
            True if TIER 1 qualified
        """
        # All profiles: model must beat buy-and-hold to be Tier 1
        if np.isnan(excess_sharpe) or excess_sharpe <= 0:
            return False

        if self.profile == "A":  # Maximum Returns
            # Relaxed: High Sharpe OR moderate Sharpe with strong significance
            return (sharpe > 0.8 and p_value < 0.10) or (
                sharpe > 0.5 and p_value < 0.05
            )

        elif self.profile == "B":  # Risk-Adjusted
            # Balanced: Good Sharpe AND significance AND acceptable regime risk
            basic_qual = sharpe > 0.6 and p_value < 0.05
            regime_ok = regime_ratio < self.regime_dependency_threshold
            dd_ok = drawdown < self.max_drawdown_threshold
            return basic_qual and regime_ok and dd_ok

        elif self.profile == "C":  # Institutional
            # Strict: High Sharpe AND strong significance AND low regime risk AND low drawdown
            return (
                sharpe > 0.7
                and p_value < 0.01
                and regime_ratio < self.regime_dependency_threshold
                and drawdown < self.max_drawdown_threshold
            )

        return False

    def rank_models(
        self, summary_files: List[Path], ticker: str = None
    ) -> pd.DataFrame:
        """
        Rank multiple models for comparison and selection.

        Applies Benjamini-Hochberg FDR correction across all models' p-values
        before tier assignment to control false discovery rate when comparing
        multiple models simultaneously.

        Args:
            summary_files: List of WFOV summary JSON file paths
            ticker: Optional ticker filter (only rank models for this ticker)

        Returns:
            DataFrame with ranked models, sorted by score (best to worst)
        """
        # --- Pass 1: Load all summaries and extract raw p-values ---
        loaded_models = []  # List of (summary_dict, summary_file) tuples

        for summary_file in summary_files:
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)

                # Filter by ticker if specified
                if ticker and summary.get("metadata", {}).get("ticker") != ticker:
                    continue

                loaded_models.append((summary, summary_file))

            except Exception as e:
                print(f"Warning: Could not process {summary_file}: {e}")
                continue

        if not loaded_models:
            return pd.DataFrame()

        # --- Dual-validation merging ---
        # When the workflow runs MC as primary + WF as companion, we get two
        # summaries per model.  Group by model_name and, when both exist, use
        # the WF summary for statistical inference while applying MC descriptors
        # (win_rate, failure_rate) as screening gates.
        model_summaries = {}  # model_name -> {"mc": (summary, file), "wf": (summary, file)}
        for summary, summary_file in loaded_models:
            model_name = summary.get("metadata", {}).get("model_name", "unknown")
            mode = summary.get("validation_mode_info", {}).get("mode", "monte_carlo")
            if model_name not in model_summaries:
                model_summaries[model_name] = {}
            if mode == "monte_carlo":
                model_summaries[model_name]["mc"] = (summary, summary_file)
            elif mode.startswith("walk_forward"):
                model_summaries[model_name]["wf"] = (summary, summary_file)

        # Rebuild loaded_models with dual-validation logic applied
        merged_models = []
        for model_name, sources in model_summaries.items():
            if "mc" in sources and "wf" in sources:
                # Both exist: use WF for statistical inference, MC for screening
                wf_summary, wf_file = sources["wf"]
                mc_summary, _ = sources["mc"]

                mc_desc = mc_summary.get("mc_descriptors", {})
                mc_win_rate = mc_desc.get("win_rate", 1.0)
                mc_failure_rate = mc_desc.get("failure_rate", 0.0)

                # Annotate the WF summary with MC screening info
                wf_summary["_mc_win_rate"] = mc_win_rate
                wf_summary["_mc_failure_rate"] = mc_failure_rate
                wf_summary["_mc_screened_out"] = (
                    mc_win_rate < 0.60 or mc_failure_rate > 0.10
                )

                merged_models.append((wf_summary, wf_file))
            elif "mc" in sources:
                # MC only (no companion WF) – existing single-mode behavior
                merged_models.append(sources["mc"])
            elif "wf" in sources:
                # WF only (user explicitly chose walk-forward) – use as-is
                merged_models.append(sources["wf"])

        loaded_models = merged_models

        # --- Apply BH-FDR correction across all models' p-values ---
        # Extract raw p-values from each loaded summary
        raw_p_values = []
        valid_p_indices = []  # Indices into loaded_models with valid (non-NaN) p-values

        for i, (summary, _) in enumerate(loaded_models):
            stat_rig = summary.get("statistical_rigor", {})
            sharpe_sig = stat_rig.get("sharpe_significance", {})
            p_val = sharpe_sig.get("p_value_two_tailed", np.nan)

            # Treat None as NaN
            if p_val is None:
                p_val = np.nan

            if not np.isnan(p_val) and 0.0 <= p_val <= 1.0:
                raw_p_values.append(p_val)
                valid_p_indices.append(i)
            # else: model has invalid/NaN p-value, skip from FDR correction

        # Apply BH-FDR if we have at least 2 valid p-values (correction is
        # meaningful only with multiple tests)
        adjusted_p_map = {}  # Maps index in loaded_models -> adjusted p-value
        raw_p_map = {}  # Maps index in loaded_models -> raw p-value

        if len(raw_p_values) >= 2:
            fdr_result = apply_benjamini_hochberg_fdr(
                np.array(raw_p_values), alpha=self.p_value_threshold
            )
            adjusted_p_values = fdr_result["p_values_adjusted"]

            for j, model_idx in enumerate(valid_p_indices):
                adjusted_p_map[model_idx] = float(adjusted_p_values[j])
                raw_p_map[model_idx] = raw_p_values[j]
        elif len(raw_p_values) == 1:
            # Single model: no correction needed, adjusted == raw
            model_idx = valid_p_indices[0]
            adjusted_p_map[model_idx] = raw_p_values[0]
            raw_p_map[model_idx] = raw_p_values[0]

        # --- Pass 2: Inject adjusted p-values and categorize ---
        results = []

        for i, (summary, summary_file) in enumerate(loaded_models):
            # If this model has an adjusted p-value, swap it into the summary
            # so that categorize_model() uses the FDR-corrected value
            p_value_raw = None
            if i in adjusted_p_map:
                p_value_raw = raw_p_map[i]
                # Inject adjusted p-value into the summary dict (in-place on our copy)
                stat_rig = summary.setdefault("statistical_rigor", {})
                sharpe_sig = stat_rig.setdefault("sharpe_significance", {})
                sharpe_sig["p_value_two_tailed"] = adjusted_p_map[i]

            # Categorize model (now uses adjusted p-value if available)
            category = self.categorize_model(summary)

            # Store raw p-value alongside adjusted for transparency
            if p_value_raw is not None:
                category["metrics"]["p_value_raw"] = float(p_value_raw)
            else:
                # Model had NaN/invalid p-value; raw == whatever categorize_model used
                category["metrics"]["p_value_raw"] = category["metrics"]["p_value"]

            # Add summary file reference
            category["summary_file"] = str(summary_file)
            category["timestamp"] = summary.get("metadata", {}).get(
                "timestamp", "unknown"
            )

            results.append(category)

        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by tier (ascending) then score (descending)
        df = df.sort_values(["tier", "score"], ascending=[True, False])
        df = df.reset_index(drop=True)

        return df

    def generate_deployment_report(
        self, ranked_models: pd.DataFrame, output_file: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable deployment report with recommendations.

        Args:
            ranked_models: DataFrame from rank_models()
            output_file: Optional file path to save report

        Returns:
            Formatted report string
        """
        if ranked_models.empty:
            return "No models to rank"

        lines = []
        lines.append("=" * 100)
        lines.append("MODEL RANKING & DEPLOYMENT RECOMMENDATIONS")
        lines.append("=" * 100)
        lines.append(f"\nProfile: {self.profile_name}")
        lines.append(f"Strategy: {self.profile_description}")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Models Analyzed: {len(ranked_models)}")
        lines.append(f"\nRanking Criteria (Profile {self.profile}):")
        lines.append(f"  p-value threshold: < {self.p_value_threshold}")
        lines.append(f"  Min Sharpe: > {self.min_sharpe_threshold}")
        lines.append(f"  Max regime dependency: < {self.regime_dependency_threshold}x")
        lines.append(f"  Min deflated Sharpe: > {self.deflated_sharpe_threshold}")

        # Group by tier
        for tier in [1, 2, 3]:
            tier_models = ranked_models[ranked_models["tier"] == tier]

            if tier_models.empty:
                continue

            tier_name = tier_models.iloc[0]["tier_name"]

            lines.append(f"\n{'━' * 100}")
            lines.append(f"TIER {tier}: {tier_name} ({len(tier_models)} models)")
            lines.append("━" * 100)

            for idx, row in tier_models.iterrows():
                metrics = row["metrics"]

                lines.append(
                    f"\n#{idx + 1}. {metrics['model_name'].upper()} for {metrics['ticker']}"
                )
                lines.append(
                    f"    Score: {row['score']:.3f} | Sharpe: {metrics['mean_sharpe']:.3f} | p-value: {metrics['p_value']:.4f}"
                )

                # Show regime info if dependent
                if metrics["regime_dependent"]:
                    lines.append(
                        f"    Regime: {metrics['regime_ratio']:.1f}x better in {metrics['best_regime']} vs {metrics['worst_regime']}"
                    )

                # Show deflated Sharpe if available
                if metrics["deflated_sharpe"] is not None:
                    lines.append(
                        f"    Deflated Sharpe: {metrics['deflated_sharpe']:.3f} (vs observed {metrics['mean_sharpe']:.3f})"
                    )

                # Show CI
                if metrics["ci_95"][0] is not None:
                    lines.append(
                        f"    95% CI: [{metrics['ci_95'][0]:.3f}, {metrics['ci_95'][1]:.3f}]"
                    )

                # Show recommendation
                lines.append(f"\n    {row['recommendation']}")

                # Show flags if any
                if row["flags"]:
                    lines.append(f"    Flags: {', '.join(row['flags'])}")

        # Summary recommendations
        lines.append(f"\n{'=' * 100}")
        lines.append("DEPLOYMENT STRATEGY")
        lines.append("=" * 100)

        tier1_models = ranked_models[ranked_models["tier"] == 1]
        tier2_models = ranked_models[ranked_models["tier"] == 2]
        tier3_models = ranked_models[ranked_models["tier"] == 3]

        if not tier1_models.empty:
            best_model = tier1_models.iloc[0]
            lines.append(f"\n🚀 RECOMMENDED FOR DEPLOYMENT:")
            lines.append(f"   Model: {best_model['metrics']['model_name'].upper()}")
            lines.append(f"   Ticker: {best_model['metrics']['ticker']}")
            lines.append(f"   Sharpe: {best_model['metrics']['mean_sharpe']:.3f}")
            lines.append(
                f"   Confidence: ✓ SIGNIFICANT (p={best_model['metrics']['p_value']:.4f})"
            )

            if best_model["flags"]:
                lines.append(f"\n   ⚠️  MONITOR CLOSELY:")
                for flag in best_model["flags"]:
                    if "regime_dependent" in flag:
                        regime = flag.split("_")[-1]
                        lines.append(f"      - Performs best in {regime} markets")
                    elif flag == "high_variance":
                        lines.append(
                            f"      - High performance variance (Sharpe std: {best_model['metrics']['sharpe_std']:.2f})"
                        )

        elif not tier2_models.empty:
            best_model = tier2_models.iloc[0]
            lines.append(f"\n⚠️  CONDITIONAL DEPLOYMENT (Review Required):")
            lines.append(f"   Model: {best_model['metrics']['model_name'].upper()}")
            lines.append(f"   Ticker: {best_model['metrics']['ticker']}")
            lines.append(f"   Sharpe: {best_model['metrics']['mean_sharpe']:.3f}")
            lines.append(f"   Issues: {', '.join(best_model['flags'])}")
            lines.append(f"\n   Consider: Deploy if you accept the flagged risks")

        else:
            lines.append(f"\n❌ NO DEPLOYABLE MODELS FOUND")
            lines.append(f"   All models failed validation criteria")
            lines.append(
                f"   Recommendation: Retrain with different hyperparameters or features"
            )

        # Alternative options
        if not tier1_models.empty and len(tier1_models) > 1:
            lines.append(f"\n📋 ALTERNATIVE OPTIONS (Tier 1):")
            for idx, row in tier1_models.iloc[1:].iterrows():
                m = row["metrics"]
                lines.append(
                    f"   - {m['model_name'].upper()}: Sharpe {m['mean_sharpe']:.3f}, p={m['p_value']:.4f}"
                )

        report = "\n".join(lines)

        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)
            print(f"\n✓ Deployment report saved to: {output_file}")

        return report

    def compare_models_for_ticker(
        self,
        ticker: str,
        results_dir: Path = Path("algos/wfov/results/summaries"),
        output_file: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Compare all models for a specific ticker and generate deployment recommendation.

        Args:
            ticker: Ticker symbol to analyze
            results_dir: Directory containing WFOV summary JSONs
            output_file: Optional file to save report

        Returns:
            Tuple of (ranked_dataframe, report_string)
        """
        # Find all summary files for this ticker
        summary_files = list(results_dir.glob(f"*_{ticker}_*.json"))

        if not summary_files:
            print(f"No WFOV results found for {ticker} in {results_dir}")
            return pd.DataFrame(), f"No results found for {ticker}"

        print(f"Found {len(summary_files)} WFOV results for {ticker}")

        # Rank models
        ranked_df = self.rank_models(summary_files, ticker=ticker)

        # Generate report
        report = self.generate_deployment_report(ranked_df, output_file)

        return ranked_df, report


def compare_all_tickers(
    tickers: List[str],
    results_dir: Path = Path("algos/wfov/results/summaries"),
    output_dir: Path = Path("algos/wfov/deployment_recommendations"),
) -> Dict[str, Tuple[pd.DataFrame, str]]:
    """
    Compare models for all tickers and generate deployment recommendations.

    Args:
        tickers: List of ticker symbols
        results_dir: Directory with WFOV summaries
        output_dir: Directory to save reports

    Returns:
        Dict mapping ticker to (ranked_df, report_string)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker = ModelRanker()

    results = {}

    print("\n" + "=" * 100)
    print("MULTI-TICKER MODEL RANKING")
    print("=" * 100)

    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")

        output_file = (
            output_dir
            / f"deployment_recommendation_{ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
        )

        ranked_df, report = ranker.compare_models_for_ticker(
            ticker=ticker, results_dir=results_dir, output_file=output_file
        )

        results[ticker] = (ranked_df, report)

        # Print quick summary
        if not ranked_df.empty:
            tier1_count = len(ranked_df[ranked_df["tier"] == 1])
            tier2_count = len(ranked_df[ranked_df["tier"] == 2])
            tier3_count = len(ranked_df[ranked_df["tier"] == 3])

            print(
                f"  Results: {tier1_count} DEPLOY | {tier2_count} REVIEW | {tier3_count} REJECT"
            )

            if tier1_count > 0:
                best = ranked_df.iloc[0]
                print(
                    f"  ✅ Best: {best['metrics']['model_name']} (Sharpe: {best['metrics']['mean_sharpe']:.3f}, Score: {best['score']:.3f})"
                )

    # Create combined summary
    combined_file = (
        output_dir
        / f"combined_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(combined_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("COMBINED DEPLOYMENT RECOMMENDATIONS - ALL TICKERS\n")
        f.write("=" * 100 + "\n\n")

        for ticker, (ranked_df, report) in results.items():
            f.write(f"\n{'━' * 100}\n")
            f.write(f"TICKER: {ticker}\n")
            f.write("━" * 100 + "\n")

            if not ranked_df.empty:
                best_model = ranked_df.iloc[0]
                f.write(
                    f"\nTop Choice: {best_model['metrics']['model_name'].upper()}\n"
                )
                f.write(f"Tier: {best_model['tier_name']}\n")
                f.write(f"Score: {best_model['score']:.3f}\n")
                f.write(f"Sharpe: {best_model['metrics']['mean_sharpe']:.3f}\n")
                f.write(f"p-value: {best_model['metrics']['p_value']:.4f}\n")
                f.write(f"\n{best_model['recommendation']}\n")
            else:
                f.write(f"\n⚠️  No WFOV results found for {ticker}\n")

    print(f"\n✓ Combined recommendations saved to: {combined_file}")

    return results


def main():
    """CLI interface for model ranking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Model Ranking & Deployment Recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile A: Maximum Returns (default)
  python -m algos.wfov.model_ranker --ticker NVDA --profile A

  # Profile B: Risk-Adjusted
  python -m algos.wfov.model_ranker --ticker NVDA --profile B

  # Profile C: Institutional
  python -m algos.wfov.model_ranker --ticker NVDA --profile C

  # Multiple tickers with profile
  python -m algos.wfov.model_ranker --tickers NVDA AVGO SPY --profile B

  # Custom thresholds (override profile)
  python -m algos.wfov.model_ranker \\
      --ticker SPY \\
      --profile A \\
      --p_value_threshold 0.20 \\
      --regime_dependency_threshold 10.0

Profile Descriptions:
  A: Maximum Returns (Aggressive)
     - Deploy highest Sharpe among significant models
     - Tolerate regime risk (monitor and switch fast)
     - Thresholds: p<0.15, Sharpe>0.25, regime<5x

  B: Risk-Adjusted (Balanced)
     - Balance returns vs risk
     - Moderate regime tolerance
     - Thresholds: p<0.05, Sharpe>0.4, regime<2.5x

  C: Institutional (Conservative)
     - Strict requirements, low risk tolerance
     - Thresholds: p<0.01, Sharpe>0.5, regime<1.5x
        """,
    )

    # Profile selection
    parser.add_argument(
        "--profile",
        type=str,
        choices=["A", "B", "C"],
        default="A",
        help="Trading profile: A (max returns), B (risk-adjusted), C (institutional)",
    )

    # Ticker selection
    parser.add_argument("--ticker", type=str, help="Single ticker to analyze")
    parser.add_argument(
        "--tickers", type=str, nargs="+", help="Multiple tickers to analyze"
    )

    # Directories
    parser.add_argument(
        "--results_dir",
        type=str,
        default="algos/wfov/results/summaries",
        help="Directory containing WFOV summary JSONs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="algos/wfov/deployment_recommendations",
        help="Directory to save recommendations",
    )

    # Threshold overrides (optional - override profile defaults)
    parser.add_argument(
        "--p_value_threshold", type=float, help="Override: Max p-value for significance"
    )
    parser.add_argument(
        "--min_sharpe_threshold", type=float, help="Override: Min Sharpe for deployment"
    )
    parser.add_argument(
        "--regime_dependency_threshold",
        type=float,
        help="Override: Max regime performance ratio",
    )
    parser.add_argument(
        "--deflated_sharpe_threshold", type=float, help="Override: Min deflated Sharpe"
    )

    args = parser.parse_args()

    # Determine tickers to analyze
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = args.tickers
    else:
        parser.error("Must specify --ticker or --tickers")

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Create ranker with profile and optional overrides
    ranker = ModelRanker(
        profile=args.profile,
        p_value_threshold=args.p_value_threshold,
        min_sharpe_threshold=args.min_sharpe_threshold,
        regime_dependency_threshold=args.regime_dependency_threshold,
        deflated_sharpe_threshold=args.deflated_sharpe_threshold,
    )

    print(f"\n{'=' * 100}")
    print(f"RANKING PROFILE: {ranker.profile_name}")
    print(f"Strategy: {ranker.profile_description}")
    print("=" * 100)

    # Analyze all tickers
    all_results = {}

    for ticker in tickers:
        print(f"\n{'=' * 100}")
        print(f"Analyzing: {ticker}")
        print("=" * 100)

        output_file = (
            output_dir
            / f"deployment_recommendation_{ticker}_{datetime.now().strftime('%Y%m%d')}.txt"
        )

        ranked_df, report = ranker.compare_models_for_ticker(
            ticker=ticker, results_dir=results_dir, output_file=output_file
        )

        all_results[ticker] = (ranked_df, report)

        # Print report to console
        print(report)

    # If multiple tickers, create combined summary
    if len(tickers) > 1:
        print(f"\n{'=' * 100}")
        print("COMBINED SUMMARY")
        print("=" * 100)

        for ticker, (ranked_df, _) in all_results.items():
            if not ranked_df.empty:
                best = ranked_df.iloc[0]
                tier_symbol = {1: "🚀", 2: "⚠️", 3: "❌"}[best["tier"]]
                print(
                    f"\n{ticker:12s}: {tier_symbol} {best['metrics']['model_name'].upper():20s} "
                    f"(Sharpe: {best['metrics']['mean_sharpe']:5.2f}, "
                    f"Score: {best['score']:5.2f}, "
                    f"Tier: {best['tier_name']})"
                )
            else:
                print(f"\n{ticker:12s}: ⚠️  No results found")

    print(f"\n✅ Analysis complete! Recommendations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
