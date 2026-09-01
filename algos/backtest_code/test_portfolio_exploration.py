"""
Test Suite for portfolio_exploration_global.py

Simple tests to validate workflow works correctly on first run.

Tests:
    1. Synthetic data workflow (no real CSV needed)
    2. Small real data (8-asset CSV)
    3. Stage 1 isolation test
    4. Covariance singularity prevention
    5. Division by zero prevention

Usage:
    python test_portfolio_exploration.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from portfolio_exploration_global import (
    stage1_multi_criteria_screening,
    stage2_direct_selection,
    stage3_global_optimization,
    stage4_analysis_and_reporting,
    run_complete_workflow,
    DEFAULT_CONFIG
)

from portimization import load_and_preprocess_data


def test_1_synthetic_data_workflow():
    """
    Test 1: Complete workflow with synthetic data (no CSV needed).

    portimization.py generates 9 synthetic assets if CSV not found.
    We'll adjust config for small asset count.
    """
    print("\n" + "="*80)
    print("TEST 1: Synthetic Data Workflow")
    print("="*80)

    config = DEFAULT_CONFIG.copy()
    config['stage1_top_n'] = 8          # 9 synthetic assets → 8
    config['stage1_min_assets'] = 5     # Lower threshold for small dataset
    config['stage2_target_n'] = 6       # 8 → 6
    config['min_sharpe'] = -0.5         # Permissive for synthetic data

    result = run_complete_workflow(
        'nonexistent_file_for_synthetic.csv',
        '2023-01-01',
        '2025-01-01',
        config
    )

    # Assertions
    assert result is not None, "Workflow returned None"
    assert result['status'] == 'success', f"Workflow failed: {result.get('error')}"
    assert 'stage3' in result, "Missing stage3 results"
    assert result['stage3']['max_sharpe']['status'] == 'success', "Max Sharpe optimization failed"
    assert result['stage3']['hrp']['status'] == 'success', "HRP failed"

    print("✅ TEST 1 PASSED")
    print(f"   Execution time: {result['execution_time']:.1f}s")
    print(f"   Final portfolios: {sum(1 for k,v in result['stage3'].items() if k != 'efficient_frontier' and v.get('status') == 'success')}")
    return result


def test_2_small_real_data():
    """
    Test 2: Workflow with small real CSV (8 assets).

    Uses: data/financial_data_combined_prices_2023-01-01_2025-06-01_1d.csv
    This file has 8 assets (SPY, GLD, NVDA, TSLA, AAPL, MSFT, AMZN, QQQ).
    """
    print("\n" + "="*80)
    print("TEST 2: Small Real Data (8 assets)")
    print("="*80)

    csv_path = os.path.join('..', '..', 'data', 'financial_data_combined_prices_2023-01-01_2025-06-01_1d.csv')

    config = DEFAULT_CONFIG.copy()
    config['stage1_top_n'] = 7
    config['stage1_min_assets'] = 5     # Lower threshold for small dataset
    config['stage2_target_n'] = 5
    config['min_sharpe'] = -0.5         # Permissive for small dataset

    result = run_complete_workflow(
        csv_path,
        '2023-01-01',
        '2025-01-01',
        config
    )

    # Assertions
    assert result is not None, "Workflow returned None"
    assert result['status'] == 'success', f"Workflow failed: {result.get('error')}"
    assert result['stage3']['hrp']['n_positions'] <= 8, "HRP has more positions than input assets"

    print("✅ TEST 2 PASSED")
    print(f"   Execution time: {result['execution_time']:.1f}s")
    return result


def test_3_stage1_isolation():
    """
    Test 3: Isolate Stage 1 for debugging.

    Creates 27 synthetic assets (3 copies of 9), tests screening logic.
    """
    print("\n" + "="*80)
    print("TEST 3: Stage 1 Isolation")
    print("="*80)

    # Load synthetic data
    returns = load_and_preprocess_data('nonexistent.csv', '2023-01-01', '2025-01-01')

    # Duplicate to create 27 assets (test filtering)
    duplicated = pd.concat([returns, returns, returns], axis=1)
    duplicated.columns = [f"Asset_{i}" for i in range(len(duplicated.columns))]

    config = DEFAULT_CONFIG.copy()
    config['stage1_top_n'] = 20
    config['min_trading_days'] = 100  # Lower threshold for synthetic

    stage1_out = stage1_multi_criteria_screening(duplicated, config)

    # Assertions
    assert 'metrics' in stage1_out, "Missing metrics in output"
    assert 'returns' in stage1_out, "Missing returns in output"
    assert len(stage1_out['returns'].columns) <= config['stage1_top_n'], \
        f"Returned {len(stage1_out['returns'].columns)} assets, expected <= {config['stage1_top_n']}"

    print("✅ TEST 3 PASSED")
    print(f"   Input: {len(duplicated.columns)} assets")
    print(f"   Output: {len(stage1_out['returns'].columns)} assets")
    return stage1_out


def test_4_covariance_singularity():
    """
    Test 4: Ensure Ledoit-Wolf handles singular covariance matrices.

    Creates 3 assets where one is perfectly correlated with another.
    Old method would crash with LinAlgError; Ledoit-Wolf handles it.
    """
    print("\n" + "="*80)
    print("TEST 4: Covariance Singularity Prevention")
    print("="*80)

    # Create perfectly correlated assets
    np.random.seed(42)
    returns = pd.DataFrame({
        'Asset_A': np.random.randn(100) * 0.01,
        'Asset_B': np.random.randn(100) * 0.01,
    })
    returns['Asset_C'] = returns['Asset_A']  # Perfect correlation → singular covariance

    config = DEFAULT_CONFIG.copy()
    config['stage2_target_n'] = 3
    config['stage2_min_assets'] = 2

    # Manually construct Stage 2 output
    stage2_out = {
        'returns': returns,
        'metrics': pd.DataFrame({'composite_score': [1, 1, 1]}, index=returns.columns)
    }

    # This should NOT crash due to Ledoit-Wolf shrinkage
    try:
        stage3_out = stage3_global_optimization(stage2_out, config)

        assert stage3_out['max_sharpe']['status'] == 'success', "Max Sharpe failed on singular matrix"
        assert stage3_out['hrp']['status'] == 'success', "HRP failed"

        print("✅ TEST 4 PASSED")
        print(f"   Ledoit-Wolf shrinkage successfully handled singular matrix")
        return stage3_out

    except np.linalg.LinAlgError as e:
        print(f"❌ TEST 4 FAILED: {e}")
        print("   Ledoit-Wolf did NOT prevent covariance singularity")
        raise


def test_5_zero_volatility():
    """
    Test 5: Ensure volatility floor prevents division by zero.

    Creates 2 assets where one has zero volatility (stablecoin scenario).
    Sharpe calculation would be inf/nan without floor.
    """
    print("\n" + "="*80)
    print("TEST 5: Zero Volatility Prevention")
    print("="*80)

    np.random.seed(42)
    returns = pd.DataFrame({
        'Normal_Asset': np.random.randn(100) * 0.01,
        'Stablecoin': [0.0] * 100,  # Zero volatility → σ = 0
    })

    config = DEFAULT_CONFIG.copy()
    config['min_trading_days'] = 50

    try:
        stage1_out = stage1_multi_criteria_screening(returns, config)

        # Should not crash
        assert stage1_out is not None, "Stage 1 returned None"

        # Check if stablecoin was handled properly
        if 'Stablecoin' in stage1_out['metrics'].index:
            sharpe = stage1_out['metrics'].loc['Stablecoin', 'sharpe']
            assert not np.isinf(sharpe), "Sharpe is inf (volatility floor not applied)"
            assert not np.isnan(sharpe), "Sharpe is nan (volatility floor not applied)"
            print(f"   Stablecoin Sharpe: {sharpe:.4f} (floor applied successfully)")
        else:
            print(f"   Stablecoin filtered out (acceptable)")

        print("✅ TEST 5 PASSED")
        print(f"   Zero volatility handled without crash")
        return stage1_out

    except (ZeroDivisionError, FloatingPointError) as e:
        print(f"❌ TEST 5 FAILED: {e}")
        print("   Volatility floor NOT applied")
        raise


def run_all_tests():
    """Run all 5 tests in sequence."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR portfolio_exploration_global.py")
    print("="*80)

    tests = [
        ('Test 1: Synthetic Data Workflow', test_1_synthetic_data_workflow),
        ('Test 2: Small Real Data', test_2_small_real_data),
        ('Test 3: Stage 1 Isolation', test_3_stage1_isolation),
        ('Test 4: Covariance Singularity', test_4_covariance_singularity),
        ('Test 5: Zero Volatility', test_5_zero_volatility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, 'PASSED'))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            results.append((test_name, 'FAILED'))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {test_name}: {status}")

    passed = sum(1 for _, status in results if status == 'PASSED')
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Workflow ready for production use")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review errors above")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
