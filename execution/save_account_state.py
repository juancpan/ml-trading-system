#!/usr/bin/env python3
"""
Save current IBKR account state for oversight system
Run this while IBKR trading system is active
"""

import json
import time
from pathlib import Path
from ib_client import IBClient
from portfolio_manager import PortfolioManager
from order_manager import OrderManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_account_state():
    """Connect to IBKR and save current account state"""
    
    logger.info("Connecting to IBKR...")
    
    # Create managers
    portfolio_manager = PortfolioManager(logger)
    order_manager = OrderManager(logger)
    
    # Create and connect client
    ib_client = IBClient(
        portfolio_manager=portfolio_manager,
        order_manager=order_manager,
        logger=logger
    )
    
    # Connect
    if not ib_client.connect_and_run():
        logger.error("Failed to connect to IBKR")
        return False
    
    # Request account updates
    ib_client.reqAccountUpdates(True, "")
    
    # Wait for data
    logger.info("Waiting for account data...")
    time.sleep(5)
    
    # Save state
    state = {
        'timestamp': time.time(),
        'account_values': portfolio_manager.account_values,
        'positions': portfolio_manager.positions
    }
    
    state_file = Path(__file__).parent / "account_state.json"
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"Account state saved to {state_file}")
    
    # Disconnect
    ib_client.disconnect()
    
    return True

if __name__ == "__main__":
    save_account_state()