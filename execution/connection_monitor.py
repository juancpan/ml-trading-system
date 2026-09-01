"""
Connection Health Monitor for IBKR Trading System
Monitors connection status and handles reconnection logic
"""

import time
from datetime import datetime, timedelta
from enum import Enum

class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"  # Some farms disconnected but core functions work

class ConnectionMonitor:
    """
    Monitors IBKR connection health and manages reconnection
    """
    
    def __init__(self, ib_client, logger, max_retries=5, retry_interval=30):
        """
        Initialize connection monitor
        
        Args:
            ib_client: IBClient instance
            logger: Logger instance
            max_retries: Maximum reconnection attempts
            retry_interval: Seconds between retry attempts
        """
        self.ib_client = ib_client
        self.logger = logger
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        
        # Connection tracking
        self.status = ConnectionStatus.DISCONNECTED
        self.last_heartbeat = datetime.now()
        self.heartbeat_interval = 60  # Check every 60 seconds
        self.connection_start_time = None
        self.reconnect_attempts = 0
        
        # Farm status tracking
        self.farm_status = {
            'market_data': False,
            'historical_data': False,
            'security_def': False
        }
        
        # Session management
        self.session_duration_hours = 23  # IB Gateway session lasts ~24 hours
        self.restart_warning_minutes = 30  # Warn 30 min before session expires
        
    def update_farm_status(self, farm_type, is_connected):
        """Update status of specific data farm"""
        self.farm_status[farm_type] = is_connected
        
        # Update overall status based on farms
        if all(self.farm_status.values()):
            self.status = ConnectionStatus.CONNECTED
        elif any(self.farm_status.values()):
            self.status = ConnectionStatus.DEGRADED
        else:
            self.status = ConnectionStatus.DISCONNECTED
    
    def check_connection_health(self):
        """
        Perform comprehensive connection health check
        
        Returns:
            tuple: (is_healthy, status_message)
        """
        try:
            # Check basic connection
            if not self.ib_client.isConnected():
                self.status = ConnectionStatus.DISCONNECTED
                return False, "Not connected to IBKR"
            
            # Check if we have received account updates (good test of connection)
            if hasattr(self.ib_client, 'portfolio_manager'):
                if not self.ib_client.portfolio_manager.account_values:
                    return False, "No account data available"
            elif not self.ib_client.account_updates_ready.is_set():
                return False, "Account updates not ready"
            
            # Update heartbeat
            self.last_heartbeat = datetime.now()
            
            # Check session expiry
            if self.connection_start_time:
                session_age = datetime.now() - self.connection_start_time
                remaining_hours = self.session_duration_hours - (session_age.total_seconds() / 3600)
                
                if remaining_hours < 0:
                    return False, "Session expired"
                elif remaining_hours < (self.restart_warning_minutes / 60):
                    self.logger.warning(f"Session expiring in {remaining_hours:.1f} hours")
            
            # Determine health based on farm status
            if self.status == ConnectionStatus.CONNECTED:
                return True, "All systems operational"
            elif self.status == ConnectionStatus.DEGRADED:
                disconnected = [k for k, v in self.farm_status.items() if not v]
                return True, f"Degraded: {', '.join(disconnected)} offline"
            else:
                return False, "Connection unhealthy"
                
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return False, f"Health check failed: {str(e)}"
    
    def handle_disconnect(self):
        """Handle disconnection event"""
        self.logger.warning("Connection lost - initiating recovery procedures")
        self.status = ConnectionStatus.DISCONNECTED
        
        # Reset farm status
        for farm in self.farm_status:
            self.farm_status[farm] = False
        
        # Start reconnection
        self.attempt_reconnection()
    
    def attempt_reconnection(self):
        """
        Attempt to reconnect to IBKR with exponential backoff
        
        Returns:
            bool: True if reconnected successfully
        """
        self.status = ConnectionStatus.RECONNECTING
        
        for attempt in range(1, self.max_retries + 1):
            self.reconnect_attempts = attempt
            self.logger.info(f"Reconnection attempt {attempt}/{self.max_retries}")
            
            try:
                # Disconnect first if needed
                if self.ib_client.isConnected():
                    self.logger.info("Disconnecting existing connection...")
                    self.ib_client.disconnect()
                    time.sleep(2)
                
                # Attempt reconnection. Reuse the client's OWN id (assigned by
                # the rotator at startup) so we reconnect as the same logical
                # client rather than a static id that may now be in use.
                from config import IB_HOST, IB_PORT, IB_CLIENT_ID
                reconnect_id = getattr(self.ib_client, "client_id", IB_CLIENT_ID)
                self.ib_client.connect(IB_HOST, IB_PORT, reconnect_id)
                
                # Wait for connection
                time.sleep(5)
                
                # Verify connection
                if self.ib_client.isConnected():
                    self.logger.info("Reconnection successful!")
                    self.status = ConnectionStatus.CONNECTED
                    self.connection_start_time = datetime.now()
                    self.reconnect_attempts = 0
                    
                    # Request initial data
                    self.ib_client.request_account_updates()
                    time.sleep(2)
                    
                    return True
                    
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt} failed: {e}")
            
            # Exponential backoff
            wait_time = min(self.retry_interval * (2 ** (attempt - 1)), 300)  # Max 5 min
            self.logger.info(f"Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)
        
        self.logger.error(f"Failed to reconnect after {self.max_retries} attempts")
        self.status = ConnectionStatus.DISCONNECTED
        return False
    
    def get_session_info(self):
        """
        Get current session information
        
        Returns:
            dict: Session details
        """
        if not self.connection_start_time:
            return {
                'status': 'No session',
                'uptime': 'N/A',
                'remaining': 'N/A'
            }
        
        uptime = datetime.now() - self.connection_start_time
        remaining = timedelta(hours=self.session_duration_hours) - uptime
        
        return {
            'status': self.status.value,
            'uptime': str(uptime).split('.')[0],
            'remaining': str(remaining).split('.')[0] if remaining.total_seconds() > 0 else 'Expired',
            'farms': self.farm_status,
            'reconnect_attempts': self.reconnect_attempts
        }
    
    def should_perform_heartbeat(self):
        """
        Check if it's time for a heartbeat check
        
        Returns:
            bool: True if heartbeat should be performed
        """
        time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        return time_since_heartbeat >= self.heartbeat_interval
    
    def log_connection_status(self):
        """Log current connection status"""
        session_info = self.get_session_info()
        self.logger.info("="*50)
        self.logger.info("CONNECTION STATUS")
        self.logger.info(f"Status: {session_info['status']}")
        self.logger.info(f"Uptime: {session_info['uptime']}")
        self.logger.info(f"Session remaining: {session_info['remaining']}")
        self.logger.info(f"Market data: {'✓' if self.farm_status['market_data'] else '✗'}")
        self.logger.info(f"Historical data: {'✓' if self.farm_status['historical_data'] else '✗'}")
        self.logger.info(f"Security def: {'✓' if self.farm_status['security_def'] else '✗'}")
        self.logger.info("="*50)