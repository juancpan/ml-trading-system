"""Tests for portfolio JSON loading and deployment pipeline."""

import json
import tempfile
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestPortfolioJsonParser:
    """Test parsing of portfolio weight JSON files."""

    def test_parse_valid_json(self, tmp_path):
        from deploy_models import PortfolioDeployer

        portfolio = {"NVDA": 0.10, "ABC": 0.20, "BIL": 0.70}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        assert deployer.weights == portfolio
        assert abs(sum(deployer.weights.values()) - 1.0) < 0.01

    def test_parse_empty_json_raises(self, tmp_path):
        from deploy_models import PortfolioDeployer

        json_path = tmp_path / "empty.json"
        json_path.write_text("{}")
        with pytest.raises(ValueError, match="empty"):
            PortfolioDeployer(str(json_path))

    def test_parse_nonexistent_file_raises(self):
        from deploy_models import PortfolioDeployer

        with pytest.raises(FileNotFoundError):
            PortfolioDeployer("/nonexistent/weights.json")

    def test_negative_weight_raises(self, tmp_path):
        from deploy_models import PortfolioDeployer

        portfolio = {"NVDA": -0.10, "ABC": 1.10}
        json_path = tmp_path / "bad.json"
        json_path.write_text(json.dumps(portfolio))
        with pytest.raises(ValueError, match="negative"):
            PortfolioDeployer(str(json_path))

    def test_weights_sum_warning(self, tmp_path, capsys):
        from deploy_models import PortfolioDeployer

        portfolio = {"NVDA": 0.10, "ABC": 0.20}
        json_path = tmp_path / "partial.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        assert deployer.weights == portfolio
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestModelDiscovery:
    """Test auto-discovery of trained models for portfolio tickers."""

    def test_discovers_existing_model(self, tmp_path):
        from deploy_models import PortfolioDeployer

        model_dumps = tmp_path / "algos" / "model_dumps"
        model_dumps.mkdir(parents=True)
        (model_dumps / "svm_optimized_algorithm_NVDA_20260101.pkl").touch()
        portfolio = {"NVDA": 0.50, "BIL": 0.50}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        deployer.discover_models(model_source_dir=model_dumps)
        assert deployer.model_assignments["NVDA"]["strategy_type"] == "ml_signal"
        assert deployer.model_assignments["NVDA"]["model_type"] == "svm"
        assert deployer.model_assignments["BIL"]["strategy_type"] == "buy_and_hold"

    def test_no_model_defaults_to_buy_and_hold(self, tmp_path):
        from deploy_models import PortfolioDeployer

        model_dumps = tmp_path / "algos" / "model_dumps"
        model_dumps.mkdir(parents=True)
        portfolio = {"UNKNOWN_TICKER": 1.0}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        deployer.discover_models(model_source_dir=model_dumps)
        assert (
            deployer.model_assignments["UNKNOWN_TICKER"]["strategy_type"]
            == "buy_and_hold"
        )
        assert deployer.model_assignments["UNKNOWN_TICKER"]["model_type"] is None

    def test_picks_latest_model(self, tmp_path):
        from deploy_models import PortfolioDeployer
        import time

        model_dumps = tmp_path / "algos" / "model_dumps"
        model_dumps.mkdir(parents=True)
        old = model_dumps / "svm_optimized_algorithm_NVDA_20250101.pkl"
        old.touch()
        time.sleep(0.05)
        new = model_dumps / "lstm_algorithm_NVDA_20260301.pkl"
        new.touch()
        portfolio = {"NVDA": 1.0}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        deployer.discover_models(model_source_dir=model_dumps)
        assert deployer.model_assignments["NVDA"]["model_type"] == "lstm"

    def test_discovers_keras_model(self, tmp_path):
        from deploy_models import PortfolioDeployer

        model_dumps = tmp_path / "algos" / "model_dumps"
        model_dumps.mkdir(parents=True)
        (model_dumps / "cnn_algorithm_PLTR_20260101.keras").touch()
        portfolio = {"PLTR": 1.0}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        deployer.discover_models(model_source_dir=model_dumps)
        assert deployer.model_assignments["PLTR"]["strategy_type"] == "ml_signal"
        assert deployer.model_assignments["PLTR"]["model_type"] == "cnn"


class TestConfigRewriter:
    """Test programmatic rewriting of execution/config.py."""

    def _make_config_file(self, tmp_path):
        """Create a minimal config.py with the same structure as the real one."""
        config_content = """\
# config.py

import logging
import pytz

IB_HOST = "127.0.0.1"
IB_PORT = 4888

# Portfolio Optimization Weights (relative allocations - should sum to ~1.0)
# These weights are applied to the leveraged capital in portfolio_mode
# or to base capital in isolated_mode

TARGET_ALLOCATION = {
    # "NVDA": 0.50,
    # "ABC": 0.50,
}

# Asset-Specific Configurations: Strategy Model Path & Kelly Fraction
# ASSET_SPECIFIC_CONFIGS = {
#     "NVDA": {
#         "kelly_fraction": 2.0,
#         "strategy_type": "ml_signal",
#         "model_type": "svm",
#         "strategy_model_path": "strategy_models/NVDA_trading_model_svm.pkl",
#         "sequence_length": 5,
#         "lags": 5,
#         "min_position_shares": None,
#     },
# }

# Derive SYMBOLS list from ASSET_SPECIFIC_CONFIGS keys
# SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())

BLACKLISTED_SYMBOLS = []

LOG_LEVEL = logging.INFO
"""
        config_path = tmp_path / "config.py"
        config_path.write_text(config_content)
        return config_path

    def _make_deployer(self, tmp_path, portfolio, model_assignments):
        """Helper to create a deployer with pre-set assignments."""
        from deploy_models import PortfolioDeployer

        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))
        deployer = PortfolioDeployer(str(json_path))
        deployer.model_assignments = model_assignments
        return deployer

    def test_generates_target_allocation(self, tmp_path):
        config_path = self._make_config_file(tmp_path)
        deployer = self._make_deployer(
            tmp_path,
            {"QQQ": 0.60, "BIL": 0.40},
            {
                "QQQ": {
                    "strategy_type": "ml_signal",
                    "model_type": "svm",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                },
                "BIL": {
                    "strategy_type": "buy_and_hold",
                    "model_type": None,
                    "source_model_path": None,
                    "model_extension": None,
                },
            },
        )
        deployer.rewrite_config(config_path)
        content = config_path.read_text()
        assert '"QQQ": 0.6' in content
        assert '"BIL": 0.4' in content
        assert '# "NVDA"' not in content

    def test_generates_asset_specific_configs(self, tmp_path):
        config_path = self._make_config_file(tmp_path)
        deployer = self._make_deployer(
            tmp_path,
            {"NVDA": 0.70, "BIL": 0.30},
            {
                "NVDA": {
                    "strategy_type": "ml_signal",
                    "model_type": "svm",
                    "source_model_path": "/some/path.pkl",
                    "model_extension": ".pkl",
                },
                "BIL": {
                    "strategy_type": "buy_and_hold",
                    "model_type": None,
                    "source_model_path": None,
                    "model_extension": None,
                },
            },
        )
        deployer.rewrite_config(config_path)
        content = config_path.read_text()
        assert "ASSET_SPECIFIC_CONFIGS = {" in content
        assert '"NVDA"' in content
        assert '"model_type": "svm"' in content
        assert '"strategy_type": "ml_signal"' in content
        assert '"BIL"' in content
        assert '"strategy_type": "buy_and_hold"' in content

    def test_symbols_line_uncommented(self, tmp_path):
        config_path = self._make_config_file(tmp_path)
        deployer = self._make_deployer(
            tmp_path,
            {"NVDA": 1.0},
            {
                "NVDA": {
                    "strategy_type": "ml_signal",
                    "model_type": "svm",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                }
            },
        )
        deployer.rewrite_config(config_path)
        content = config_path.read_text()
        assert "SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())" in content
        lines = content.split("\n")
        symbols_lines = [l for l in lines if "SYMBOLS = list(" in l]
        assert len(symbols_lines) == 1
        assert not symbols_lines[0].strip().startswith("#")

    def test_preserves_surrounding_config(self, tmp_path):
        config_path = self._make_config_file(tmp_path)
        deployer = self._make_deployer(
            tmp_path,
            {"NVDA": 1.0},
            {
                "NVDA": {
                    "strategy_type": "ml_signal",
                    "model_type": "svm",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                }
            },
        )
        deployer.rewrite_config(config_path)
        content = config_path.read_text()
        assert 'IB_HOST = "127.0.0.1"' in content
        assert "IB_PORT = 4888" in content
        assert "BLACKLISTED_SYMBOLS = []" in content
        assert "LOG_LEVEL = logging.INFO" in content

    def test_rewrite_is_valid_python(self, tmp_path):
        config_path = self._make_config_file(tmp_path)
        deployer = self._make_deployer(
            tmp_path,
            {"NVDA": 0.50, "BIL": 0.50},
            {
                "NVDA": {
                    "strategy_type": "ml_signal",
                    "model_type": "svm",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                },
                "BIL": {
                    "strategy_type": "buy_and_hold",
                    "model_type": None,
                    "source_model_path": None,
                    "model_extension": None,
                },
            },
        )
        deployer.rewrite_config(config_path)
        content = config_path.read_text()
        compile(content, str(config_path), "exec")


class TestDryRun:
    """Test --dry-run mode shows changes without writing."""

    def test_dry_run_does_not_modify_config(self, tmp_path, capsys):
        from deploy_models import PortfolioDeployer

        config_content = 'TARGET_ALLOCATION = {\n    # "OLD": 1.0,\n}\n'
        config_path = tmp_path / "config.py"
        config_path.write_text(config_content)

        portfolio = {"NEW": 1.0}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))

        deployer = PortfolioDeployer(str(json_path))
        deployer.model_assignments = {
            "NEW": {
                "strategy_type": "buy_and_hold",
                "model_type": None,
                "source_model_path": None,
                "model_extension": None,
            },
        }

        deployer.dry_run(config_path=config_path)

        # Config file should be unchanged
        assert config_path.read_text() == config_content

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "NEW" in captured.out

    def test_dry_run_shows_summary(self, tmp_path, capsys):
        from deploy_models import PortfolioDeployer

        portfolio = {"NVDA": 0.60, "BIL": 0.40}
        json_path = tmp_path / "weights.json"
        json_path.write_text(json.dumps(portfolio))

        deployer = PortfolioDeployer(str(json_path))
        deployer.model_assignments = {
            "NVDA": {
                "strategy_type": "ml_signal",
                "model_type": "svm",
                "source_model_path": None,
                "model_extension": ".pkl",
            },
            "BIL": {
                "strategy_type": "buy_and_hold",
                "model_type": None,
                "source_model_path": None,
                "model_extension": None,
            },
        }

        deployer.dry_run(config_path=tmp_path / "nonexistent_config.py")

        captured = capsys.readouterr()
        assert "NVDA" in captured.out
        assert "ml_signal" in captured.out
        assert "svm" in captured.out
        assert "BIL" in captured.out
        assert "buy_and_hold" in captured.out


import subprocess


class TestCLIIntegration:
    """Test the CLI interface of deploy_models.py."""

    def test_help_shows_portfolio_flag(self):
        result = subprocess.run(
            [sys.executable, "deploy_models.py", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert "--portfolio" in result.stdout
        assert "--dry-run" in result.stdout

    def test_dry_run_with_real_json(self, tmp_path):
        portfolio = {"NVDA": 0.50, "ABC": 0.50}
        json_path = tmp_path / "test_weights.json"
        json_path.write_text(json.dumps(portfolio))

        result = subprocess.run(
            [
                sys.executable,
                "deploy_models.py",
                "--portfolio",
                str(json_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout
        assert "NVDA" in result.stdout


class TestRealConfigIntegration:
    """Integration tests using the actual config.py structure."""

    def test_rewrite_real_config_structure(self):
        """Rewrite the real config.py and verify it's valid Python."""
        from deploy_models import PortfolioDeployer
        import shutil

        project_root = Path(__file__).resolve().parents[1]
        real_config = project_root / "execution" / "config.py"
        assert real_config.exists(), "execution/config.py must exist"

        # Work on a copy so we never damage the real file
        tmp_config = project_root / "execution" / "config.py.test_backup"
        shutil.copy2(real_config, tmp_config)

        try:
            portfolio = {
                "QQQ": 0.089,
                "CSH-UN.TO": 0.102,
                "BIL": 0.12,
                "ABC": 0.095,
                "GLD": 0.058,
            }

            json_path = project_root / "tests" / "_temp_test_weights.json"
            json_path.write_text(json.dumps(portfolio))

            deployer = PortfolioDeployer(str(json_path))
            deployer.model_assignments = {
                "QQQ": {
                    "strategy_type": "ml_signal",
                    "model_type": "var",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                },
                "CSH-UN.TO": {
                    "strategy_type": "ml_signal",
                    "model_type": "lstm",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                },
                "BIL": {
                    "strategy_type": "buy_and_hold",
                    "model_type": None,
                    "source_model_path": None,
                    "model_extension": None,
                },
                "ABC": {
                    "strategy_type": "ml_signal",
                    "model_type": "svm",
                    "source_model_path": None,
                    "model_extension": ".pkl",
                },
                "GLD": {
                    "strategy_type": "buy_and_hold",
                    "model_type": None,
                    "source_model_path": None,
                    "model_extension": None,
                },
            }

            deployer.rewrite_config(real_config)

            content = real_config.read_text()

            # 1. Valid Python
            compile(content, str(real_config), "exec")

            # 2. Has our tickers
            assert '"QQQ": 0.089' in content
            assert '"BIL": 0.12' in content
            assert "ASSET_SPECIFIC_CONFIGS = {" in content
            assert "SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())" in content

            # 3. Preserved other sections
            assert "IB_HOST" in content
            assert "LEVERAGE_MODE" in content
            assert "REGION_EXCHANGES" in content
            assert "CURRENCY_RATE_FALLBACKS" in content
            assert "CASH_PORTFOLIO_CONFIG" in content
            assert "LOG_LEVEL" in content

            # 4. No old commented entries leaked
            assert '# "NVDA": 0.0201' not in content

        finally:
            # Restore original
            shutil.copy2(tmp_config, real_config)
            tmp_config.unlink(missing_ok=True)
            json_path = project_root / "tests" / "_temp_test_weights.json"
            json_path.unlink(missing_ok=True)
