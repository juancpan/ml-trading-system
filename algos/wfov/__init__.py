"""
Walk-Forward Out-of-Sample Validation (WFOV) module.

Implements Monte Carlo-style randomized backtesting to validate model robustness
across diverse market conditions and minimize overfitting.

Author: jcp
Date: 2025-12-02
"""

from algos.wfov.wfov_runner import WFOVRunner
from algos.wfov.window_generator import generate_random_windows
from algos.wfov.metrics_aggregator import extract_all_metrics
from algos.wfov.results_formatter import (
    save_iteration_to_csv,
    generate_summary_statistics,
    save_summary_json
)

__all__ = [
    'WFOVRunner',
    'generate_random_windows',
    'extract_all_metrics',
    'save_iteration_to_csv',
    'generate_summary_statistics',
    'save_summary_json'
]
