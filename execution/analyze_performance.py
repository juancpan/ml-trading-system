#!/usr/bin/env python3
"""
Quick script to analyze trading performance from reports.
Run this anytime to get current performance metrics.
"""

from performance_analytics import PerformanceAnalytics
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Analyze trading performance')
    parser.add_argument('--report-file', default='trading_report.log',
                       help='Path to trading report file')
    parser.add_argument('--lookback', type=int, default=30,
                       help='Days to look back for metrics calculation')
    parser.add_argument('--risk-free-rate', type=float, default=4.0,
                       help='Annual risk-free rate in percentage (default: 4.0)')
    parser.add_argument('--export', action='store_true',
                       help='Export metrics to JSON and CSV')
    parser.add_argument('--json', action='store_true',
                       help='Output raw metrics as JSON')
    
    args = parser.parse_args()
    
    # Create analytics instance
    analytics = PerformanceAnalytics(
        report_file=args.report_file,
        risk_free_rate=args.risk_free_rate
    )
    
    if args.json:
        # Output raw metrics as JSON
        import json
        metrics = analytics.calculate_performance_metrics(lookback_days=args.lookback)
        print(json.dumps(metrics, indent=2, default=str))
    else:
        # Generate formatted report
        report = analytics.generate_performance_report()
        print(report)
    
    if args.export:
        json_file = analytics.export_metrics_json()
        csv_file = analytics.export_metrics_csv()
        print(f"\n📁 Files exported:")
        print(f"  - JSON metrics: {json_file}")
        print(f"  - CSV history: {csv_file}")


if __name__ == "__main__":
    main()