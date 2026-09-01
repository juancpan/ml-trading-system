''' Demonstrates how an application can submit orders and request information '''

from threading import Thread, Event
import sys
import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper
from ibapi.utils import iswrapper

# Core user-defined variables
ticker = 'TSLA'         # Dynamic variable, stock ticker symbol
security_type = 'STK'   # Dynamic variable, security type (STK for stock)
exchange = 'SMART'
currency = 'USD'
search_query = ticker   # Note: search_query is not used in this streamlined script
action = "BUY"          # Dynamic variable, action to take (BUY or SELL)
quantity = 5            # Dynamic variable, number of shares to buy
order_type = "MKT"      # Dynamic variable, order type (LMT for limit, MKT for market)
limit_price = 131       # Dynamic variable, limit price for limit orders (only for LMT orders)
port = 4001  # Live trading port
# port = 4002  # Paper trading port
# port = 7497  # TWS paper trading port

# A user-defined wrapper of TWS API to submit order
class SubmitOrder(EWrapper, EClient):
    ''' Serves as the client and the wrapper '''

    def __init__(self, addr, port, client_id):
        # Initialize Event for nextValidId before EClient.__init__
        self.next_order_id_event = Event()
        self.nextValidOrderId = -1 # Renamed to match ibapi convention

        EClient.__init__(self, self)

        # Connect to TWS
        self.connect(addr, port, client_id)

        # Launch the client thread
        thread = Thread(target=self.run)
        thread.start()

        # Wait for the connection to be established using EClient.isConnected()
        max_wait_time = 10  # seconds, increased for robustness
        start_time = time.time()
        print(f"Waiting for connection to TWS (max {max_wait_time} seconds)...")
        while not self.isConnected() and (time.time() - start_time) < max_wait_time:
            time.sleep(0.1) # Small sleep to avoid busy-waiting

        if not self.isConnected():
            print("Failed to connect to TWS/IB Gateway within the given time. Exiting.")
            self.disconnect()
            sys.exit() # Exit if connection failed

        print("Connected to TWS/IB Gateway!")

    @iswrapper
    def nextValidId(self, orderId: int): # Renamed to orderId for consistency with ibapi
        ''' Provides the next order ID '''
        super().nextValidId(orderId) # Call super class method
        self.nextValidOrderId = orderId
        print(f'Received next valid order ID: {orderId}')
        self.next_order_id_event.set() # Signal that ID is received

    @iswrapper
    def openOrder(self, orderId, contract, order, orderState): # Renamed state to orderState
        ''' Called in response to the submitted order '''
        print(f'\n--- Open Order (OrderId: {orderId}) ---')
        print(f'  Contract: {contract.symbol} ({contract.secType}) @ {contract.exchange}')
        print(f'  Action: {order.action}, Quantity: {order.totalQuantity}, Order Type: {order.orderType}')
        print(f'  Current Status: {orderState.status}')

    @iswrapper
    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float,
                    avgFillPrice: float, permId: int, parentId: int,
                    lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        ''' Check the status of the submitted order '''
        print(f'\n--- Order Status (OrderId: {orderId}) ---')
        print(f'  Status: {status}, Filled: {filled}, Remaining: {remaining}')
        print(f'  Average fill price: {avgFillPrice}, Last fill price: {lastFillPrice}')
        if whyHeld:
            print(f'  Why Held: {whyHeld}')

    @iswrapper
    def position(self, account: str, contract: Contract, pos: float, avgCost: float):
        ''' Read information about the account's open positions '''
        print(f'\n--- Position ---')
        print(f'  Account: {account}, Symbol: {contract.symbol}, Position: {pos}, Avg Cost: {avgCost}')

    @iswrapper
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        ''' Read information about the account '''
        print(f'--- Account Summary (ReqId: {reqId}) ---')
        print(f'  Account: {account}, Tag: {tag}, Value: {value}, Currency: {currency}')

    @iswrapper
    def error(self, reqId: int, errorTime: int, code: int, msg: str, advancedOrderReject=''):
        ''' Callback for API errors (IBAPI >= 10.37 signature with errorTime).

        See MEMORY.md "IBKR API Gotcha". '''
        # Suppress common informational messages (2103, 2104, 2106, 2158)
        if code in [2103, 2104, 2106, 2158]:
            pass
        else:
            print(f'Error {code} (ReqId: {reqId}): {msg}')
            if advancedOrderRejectJson:
                print(f'  Advanced Order Reject JSON: {advancedOrderRejectJson}')

def main():

    # Create the client and connect to TWS
    client = SubmitOrder('127.0.0.1', port, 0) # 0 is the default client ID

    # Define a contract for the ticker
    contract = Contract()
    contract.symbol = ticker
    contract.secType = security_type
    contract.exchange = exchange
    contract.currency = currency

    # Define the order
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = order_type
    if order_type == "LMT": # Only set lmtPrice for Limit orders
        order.lmtPrice = limit_price
    order.transmit = True # This transmits the order to the exchange

    # Obtain a valid ID for the order
    print("\nRequesting next valid order ID...")
    client.reqIds(-1) # Request ID -1 is commonly used for this

    # Wait for the nextValidId to be assigned
    max_id_wait_time = 5 # seconds
    start_id_time = time.time()
    # Loop until nextValidOrderId is no longer -1 (its initial value) or timeout
    while client.nextValidOrderId == -1 and (time.time() - start_id_time) < max_id_wait_time:
        client.next_order_id_event.wait(0.1) # Wait briefly, then re-check
        # Reset the event to allow re-waiting if it gets set multiple times unexpectedly
        # or if the loop runs again before nextValidId is actually assigned.
        # This prevents the wait from being instantly satisfied on subsequent iterations
        # if the event was set but ID not yet processed.
        client.next_order_id_event.clear()

    if client.nextValidOrderId == -1:
        print('Error: Next valid order ID not received within timeout. Ending application.')
        client.disconnect()
        sys.exit()
    else:
        print(f"Next valid order ID successfully retrieved: {client.nextValidOrderId}")

    # Place the order
    print(f"\nPlacing order (ID: {client.nextValidOrderId}) for {quantity} of {ticker}...")
    client.placeOrder(client.nextValidOrderId, contract, order)
    # Give time for order status updates
    time.sleep(5)

    # Obtain information about open positions
    print("\nRequesting open positions...")
    client.reqPositions()
    time.sleep(2)

    # Obtain information about account summary
    print("\nRequesting account summary...")
    client.reqAccountSummary(0, 'All', 'AccountType,AvailableFunds,NetLiquidation')
    time.sleep(2)

    # Disconnect from TWS
    print("\nDisconnecting from TWS...")
    client.disconnect()
    print("Disconnected.")

if __name__ == '__main__':
    main()