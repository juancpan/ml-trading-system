'''
IBKR TWS API Getting Familiar with classes.

This example script may contains some pseudo codes for illustrating only.

For production, first understand how everything works then write real code.

@Author: jcp
@Date: 05/15/2025
'''

# Before running do the following: 
# 1. Activaet the tws_env conda environment;
# 2. Run TWS Destkop or IB Gateway.

# The EClient Class simple illustration

# After environment activated, some normal imports
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

# Instantiate EClient.
client = EClient(wrapper)
client.connect("127.0.0.1", 7497, 0)

# Retrieve response/responses continuously
thread = Thread(target=self.run)
thread.start()

# The EWrapper Class simple illustration

# Show Errors
@iswrapper
def error(self, req_id, code, msg):
    print('Error {}: {}'.format(code, msg))