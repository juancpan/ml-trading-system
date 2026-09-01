#!/usr/bin/env python3
"""
Performance Analytics Module for Trading System.
Parses trading reports and calculates key performance metrics.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging


class PerformanceAnalytics:
    """Calculates and tracks trading performance metrics."""
    
    def __init__(self, report_file: str = "trading_report.log", 
                 history_file: str = "performance_history.json",
                 risk_free_rate: float = 4.0):
        self.report_file = report_file
        self.history_file = history_file
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate in percentage
        self.logger = logging.getLogger("PerformanceAnalytics")
        
        # Load historical data if exists
        self.performance_history = self.load_history()
        
    def load_history(self) -> Dict:
        """Load historical performance data."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'daily_returns': [],
                'positions': [],
                'timestamps': [],
                'net_liquidations': [],
                'unrealized_pnls': [],
                'realized_pnls': []
            }
    
    def save_history(self):
        """Save performance history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def parse_latest_report(self) -> Dict:
        """Parse the latest trading report entry."""
        try:
            with open(self.report_file, 'r') as f:
                content = f.read()
            
            # Find the last report section
            reports = content.split('--- Trading Report - ')
            if len(reports) < 2:
                return {}
            
            latest_report = reports[-1]
            
            # Parse key metrics
            metrics = {}
            
            # Extract timestamp
            timestamp_match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', latest_report)
            if timestamp_match:
                metrics['timestamp'] = timestamp_match.group(1)
            
            # Extract Net Liquidation
            net_liq_match = re.search(r'NetLiquidation: ([\d.-]+)', latest_report)
            if net_liq_match:
                metrics['net_liquidation'] = float(net_liq_match.group(1))
            
            # Extract Cash Balance
            cash_match = re.search(r'TotalCashBalance: ([-\d.]+)', latest_report)
            if cash_match:
                metrics['cash_balance'] = float(cash_match.group(1))
            
            # Extract Unrealized PnL
            unrealized_match = re.search(r'UnrealizedPnL: ([-\d.]+)', latest_report)
            if unrealized_match:
                metrics['unrealized_pnl'] = float(unrealized_match.group(1))
            
            # Extract Realized PnL
            realized_match = re.search(r'RealizedPnL: ([-\d.]+)', latest_report)
            if realized_match:
                metrics['realized_pnl'] = float(realized_match.group(1))
            
            # Extract positions
            positions = {}
            portfolio_section = re.search(r'Current Portfolio:(.*?)(?:Open Orders:|$)', 
                                        latest_report, re.DOTALL)
            if portfolio_section:
                position_lines = portfolio_section.group(1).strip().split('\n')
                for line in position_lines:
                    if ':' in line and 'Position=' in line:
                        symbol_match = re.match(r'\s*(\w+):', line)
                        position_match = re.search(r'Position=([\d.-]+)', line)
                        avgcost_match = re.search(r'AvgCost=([\d.-]+)', line)
                        unrealized_match = re.search(r'UnrealizedPNL=([-\d.]+)', line)
                        
                        if symbol_match and position_match:
                            symbol = symbol_match.group(1)
                            positions[symbol] = {
                                'shares': float(position_match.group(1)),
                                'avg_cost': float(avgcost_match.group(1)) if avgcost_match else 0,
                                'unrealized_pnl': float(unrealized_match.group(1)) if unrealized_match else 0
                            }
            
            metrics['positions'] = positions
            
            # Extract leverage
            leverage_match = re.search(r'Leverage-S: ([\d.]+)', latest_report)
            if leverage_match:
                metrics['leverage'] = float(leverage_match.group(1))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error parsing report: {e}")
            return {}
    
    def calculate_performance_metrics(self, lookback_days: int = 30) -> Dict:
        """Calculate comprehensive performance metrics."""
        
        # Parse latest report
        latest = self.parse_latest_report()
        if not latest:
            return {}
        
        # Update history (avoid duplicates)
        if 'timestamp' in latest:
            # Check if this timestamp already exists
            if not self.performance_history['timestamps'] or \
               latest['timestamp'] != self.performance_history['timestamps'][-1]:
                self.performance_history['timestamps'].append(latest['timestamp'])
                if 'net_liquidation' in latest:
                    self.performance_history['net_liquidations'].append(latest['net_liquidation'])
                if 'unrealized_pnl' in latest:
                    self.performance_history['unrealized_pnls'].append(latest['unrealized_pnl'])
                if 'realized_pnl' in latest:
                    self.performance_history['realized_pnls'].append(latest['realized_pnl'])
                if 'positions' in latest:
                    self.performance_history['positions'].append(latest['positions'])
        
        # Calculate returns if we have history
        returns = []
        if len(self.performance_history['net_liquidations']) > 1:
            net_liq_series = pd.Series(self.performance_history['net_liquidations'])
            returns = net_liq_series.pct_change().dropna().tolist()
            self.performance_history['daily_returns'] = returns
        
        # Calculate metrics
        metrics = {
            'timestamp': latest.get('timestamp', ''),
            'net_liquidation': latest.get('net_liquidation', 0),
            'total_pnl': latest.get('unrealized_pnl', 0) + latest.get('realized_pnl', 0),
            'unrealized_pnl': latest.get('unrealized_pnl', 0),
            'realized_pnl': latest.get('realized_pnl', 0),
            'leverage': latest.get('leverage', 1.0),
            'positions': latest.get('positions', {})
        }
        
        # Calculate PnL percentage for each position
        for symbol, pos_data in metrics['positions'].items():
            if pos_data['shares'] > 0 and pos_data['avg_cost'] > 0:
                total_invested = pos_data['shares'] * pos_data['avg_cost']
                pos_data['pnl_percentage'] = (pos_data['unrealized_pnl'] / total_invested) * 100
                pos_data['total_invested'] = total_invested
        
        # Calculate overall PnL percentage
        total_invested = sum(p.get('total_invested', 0) for p in metrics['positions'].values())
        if total_invested > 0:
            metrics['total_pnl_percentage'] = (metrics['total_pnl'] / total_invested) * 100
        elif len(self.performance_history['net_liquidations']) > 0:
            # If no position data but we have history, use initial net liq as base
            initial_value = self.performance_history['net_liquidations'][0]
            if initial_value > 0:
                metrics['total_pnl_percentage'] = (metrics['total_pnl'] / initial_value) * 100
            else:
                metrics['total_pnl_percentage'] = 0
        else:
            metrics['total_pnl_percentage'] = 0
        
        # If we have enough history, calculate advanced metrics
        if len(returns) >= 2:
            returns_array = np.array(returns[-lookback_days:])  # Use last N days
            
            # Basic statistics
            metrics['returns_mean_daily'] = np.mean(returns_array) * 100  # As percentage
            metrics['returns_std_daily'] = np.std(returns_array) * 100
            
            # Annualized metrics (252 trading days)
            metrics['returns_annualized'] = metrics['returns_mean_daily'] * 252
            metrics['volatility_annualized'] = metrics['returns_std_daily'] * np.sqrt(252)
            
            # Sharpe Ratio (using configurable risk-free rate)
            if metrics['volatility_annualized'] > 0:
                metrics['sharpe_ratio'] = (metrics['returns_annualized'] - self.risk_free_rate) / metrics['volatility_annualized']
            else:
                metrics['sharpe_ratio'] = 0
            
            # Sortino Ratio (downside deviation)
            daily_rf_rate = self.risk_free_rate / 252  # Daily risk-free rate
            excess_returns = returns_array - (daily_rf_rate / 100)
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * 100 * np.sqrt(252)
                if downside_deviation > 0:
                    metrics['sortino_ratio'] = (metrics['returns_annualized'] - self.risk_free_rate) / downside_deviation
                else:
                    metrics['sortino_ratio'] = 0
            else:
                metrics['sortino_ratio'] = float('inf')  # No downside returns
            
            # Risk metrics
            metrics['skewness'] = float(pd.Series(returns_array).skew())
            metrics['kurtosis'] = float(pd.Series(returns_array).kurt())
            
            # Maximum Drawdown
            cumulative_returns = (1 + pd.Series(returns_array)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = float(drawdown.min() * 100)  # As percentage
            
            # Value at Risk (95% confidence)
            metrics['var_95'] = float(np.percentile(returns_array, 5) * 100)
            
            # Calmar Ratio (annualized return / max drawdown)
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = metrics['returns_annualized'] / abs(metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0
            
            # Win rate
            positive_returns = sum(1 for r in returns_array if r > 0)
            metrics['win_rate'] = (positive_returns / len(returns_array)) * 100 if returns_array.size > 0 else 0
            
            # Profit factor
            gains = sum(r for r in returns_array if r > 0)
            losses = abs(sum(r for r in returns_array if r < 0))
            metrics['profit_factor'] = gains / losses if losses > 0 else float('inf')
        
        # Save updated history
        self.save_history()
        
        return metrics
    
    def generate_performance_report(self) -> str:
        """Generate a formatted performance report."""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            return "No performance data available."
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ANALYTICS REPORT")
        report.append(f"Generated: {datetime.now()}")
        report.append("=" * 60)
        
        # Account Summary
        report.append("\n📊 ACCOUNT SUMMARY")
        report.append(f"Net Liquidation: ${metrics.get('net_liquidation', 0):,.2f}")
        report.append(f"Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
        report.append(f"Total P&L %: {metrics.get('total_pnl_percentage', 0):.2f}%")
        report.append(f"Leverage: {metrics.get('leverage', 1.0):.2f}x")
        
        # Position Details
        if metrics.get('positions'):
            report.append("\n📈 POSITIONS")
            for symbol, pos in metrics['positions'].items():
                report.append(f"\n{symbol}:")
                report.append(f"  Shares: {pos.get('shares', 0):.0f}")
                report.append(f"  Avg Cost: ${pos.get('avg_cost', 0):.2f}")
                report.append(f"  Invested: ${pos.get('total_invested', 0):,.2f}")
                report.append(f"  Unrealized P&L: ${pos.get('unrealized_pnl', 0):,.2f}")
                report.append(f"  P&L %: {pos.get('pnl_percentage', 0):.2f}%")
        
        # Performance Metrics (if available)
        if 'sharpe_ratio' in metrics:
            report.append("\n📉 PERFORMANCE METRICS")
            report.append(f"Risk-Free Rate: {self.risk_free_rate:.2f}%")
            report.append(f"Daily Return: {metrics.get('returns_mean_daily', 0):.3f}%")
            report.append(f"Daily Volatility: {metrics.get('returns_std_daily', 0):.3f}%")
            report.append(f"Annualized Return: {metrics.get('returns_annualized', 0):.2f}%")
            report.append(f"Annualized Volatility: {metrics.get('volatility_annualized', 0):.2f}%")
            report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            
            sortino = metrics.get('sortino_ratio', 0)
            if sortino == float('inf'):
                report.append(f"Sortino Ratio: ∞ (no downside)")
            else:
                report.append(f"Sortino Ratio: {sortino:.3f}")
            
            report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
            
            report.append("\n⚠️ RISK METRICS")
            report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            report.append(f"Value at Risk (95%): {metrics.get('var_95', 0):.2f}%")
            report.append(f"Skewness: {metrics.get('skewness', 0):.3f}")
            report.append(f"Kurtosis: {metrics.get('kurtosis', 0):.3f}")
            
            report.append("\n🎯 TRADING STATISTICS")
            report.append(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
            profit_factor = metrics.get('profit_factor', 0)
            if profit_factor == float('inf'):
                report.append(f"Profit Factor: ∞ (no losses)")
            else:
                report.append(f"Profit Factor: {profit_factor:.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def export_metrics_json(self, filename: str = "performance_metrics.json"):
        """Export metrics to JSON file."""
        metrics = self.calculate_performance_metrics()
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        return filename
    
    def export_metrics_csv(self, filename: str = "performance_history.csv"):
        """Export historical metrics to CSV."""
        df = pd.DataFrame({
            'timestamp': self.performance_history.get('timestamps', []),
            'net_liquidation': self.performance_history.get('net_liquidations', []),
            'unrealized_pnl': self.performance_history.get('unrealized_pnls', []),
            'realized_pnl': self.performance_history.get('realized_pnls', [])
        })
        
        if not df.empty:
            df.to_csv(filename, index=False)
        return filename


def main():
    """Example usage of PerformanceAnalytics."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create analytics instance
    analytics = PerformanceAnalytics()
    
    # Generate and print performance report
    report = analytics.generate_performance_report()
    print(report)
    
    # Export metrics
    json_file = analytics.export_metrics_json()
    print(f"\nMetrics exported to: {json_file}")
    
    csv_file = analytics.export_metrics_csv()
    print(f"History exported to: {csv_file}")


if __name__ == "__main__":
    main()