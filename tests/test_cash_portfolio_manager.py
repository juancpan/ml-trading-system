import logging

from cash_portfolio_manager import CashPortfolioManager


class DummyIB:
    def __init__(self, currency_balances):
        self.currency_balances = currency_balances


def test_phase2_revert_sells_usdjpy_to_cover_jpy_quote_debt(monkeypatch, caplog):
    """JPY quote-currency debt is reduced by SELL USD.JPY, not BUY USD.JPY."""
    logger = logging.getLogger("test_phase2_revert_direction")
    manager = CashPortfolioManager(
        DummyIB({"USD": 0.0, "JPY": -546_772.22}),
        logger,
        dry_run=True,
    )
    manager._hold_state = {}

    monkeypatch.setattr(manager, "_get_carry_signal", lambda pair, cfg: -1)
    monkeypatch.setattr(manager, "_get_pair_rate", lambda pair: 162.26)
    monkeypatch.setattr(manager, "_save_hold_state", lambda: None)

    caplog.set_level(logging.INFO, logger=logger.name)

    result = manager.run_phase2_carry_trade({"NetLiquidation": 11_209.59})

    assert result["reverted"] == 1
    assert "REVERT: SELL 3,369 USD.JPY" in caplog.text
    assert "[DRY RUN] Would SELL 3,369 USD.JPY" in caplog.text
    assert "REVERT: BUY" not in caplog.text
