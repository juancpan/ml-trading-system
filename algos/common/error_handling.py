"""
Centralized error handling and logging utilities.
Provides consistent error management across the codebase.
"""

import logging
import sys
import traceback
import functools
import time
from typing import Optional, Callable, Any, Dict
from enum import Enum
from pathlib import Path
import json
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class TradingError(Exception):
    """Base exception for trading system errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR,
                 details: Optional[Dict] = None):
        super().__init__(message)
        self.severity = severity
        self.details = details or {}
        self.timestamp = datetime.now()


class DataError(TradingError):
    """Exception for data-related errors."""
    pass


class ModelError(TradingError):
    """Exception for model-related errors."""
    pass


class ExecutionError(TradingError):
    """Exception for trade execution errors."""
    pass


class ConnectionError(TradingError):
    """Exception for connection-related errors."""
    pass


class ErrorHandler:
    """
    Centralized error handler with logging and recovery mechanisms.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Error statistics
        self.error_counts = {}
        self.last_errors = {}
        
        # Setup error log
        self.error_log_path = self.log_dir / 'errors.json'
        
    def handle_error(self, error: Exception, context: str = "Unknown",
                    reraise: bool = False) -> None:
        """
        Handle an error with appropriate logging and actions.
        """
        # Determine severity
        if isinstance(error, TradingError):
            severity = error.severity
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            severity = ErrorSeverity.INFO
        else:
            severity = ErrorSeverity.ERROR
        
        # Update statistics
        self.error_counts[context] = self.error_counts.get(context, 0) + 1
        self.last_errors[context] = {
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }
        
        # Log error
        self._log_error(error, context, severity)
        
        # Take action based on severity
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error, context)
        elif severity == ErrorSeverity.ERROR:
            self._handle_regular_error(error, context)
        
        if reraise:
            raise error
    
    def _log_error(self, error: Exception, context: str, severity: ErrorSeverity):
        """Log error to file."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'severity': severity.value,
            'error_type': type(error).__name__,
            'message': str(error),
            'details': getattr(error, 'details', {}),
            'traceback': traceback.format_exc() if severity != ErrorSeverity.INFO else None
        }
        
        # Append to error log
        try:
            with open(self.error_log_path, 'a') as f:
                json.dump(error_entry, f)
                f.write('\n')
        except Exception as e:
            print(f"Failed to write error log: {e}")
    
    def _handle_critical_error(self, error: Exception, context: str):
        """Handle critical errors that require system shutdown."""
        print(f"CRITICAL ERROR in {context}: {error}")
        print("System will shutdown to prevent further damage")
        # Could send alerts here (email, SMS, etc.)
    
    def _handle_regular_error(self, error: Exception, context: str):
        """Handle regular errors with potential recovery."""
        print(f"ERROR in {context}: {error}")
        # Log for later analysis
    
    def get_error_summary(self) -> Dict:
        """Get summary of errors for monitoring."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts_by_context': self.error_counts,
            'last_errors': self.last_errors
        }


def retry_on_error(max_retries: int = 3, delay: float = 1.0,
                  backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying functions on error.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"All {max_retries + 1} attempts failed")
                        raise
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def safe_execute(func: Callable, default_value: Any = None,
                error_handler: Optional[ErrorHandler] = None,
                context: Optional[str] = None) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_value: Value to return if function fails
        error_handler: ErrorHandler instance for logging
        context: Context string for error tracking
    
    Returns:
        Function result or default_value on error
    """
    try:
        return func()
    except Exception as e:
        if error_handler:
            error_handler.handle_error(e, context or func.__name__, reraise=False)
        else:
            print(f"Error in {context or func.__name__}: {e}")
        return default_value


class LoggerAdapter:
    """
    Enhanced logger with structured logging and error tracking.
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None,
                 level: str = "INFO", enable_console: bool = True):
        self.name = name
        self.log_dir = log_dir or Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            self.log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(file_handler)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter('%(levelname)s: %(message)s')
            )
            self.logger.addHandler(console_handler)
        
        # Error handler
        self.error_handler = ErrorHandler(log_dir)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception."""
        self.logger.error(message, exc_info=error is not None, extra=kwargs)
        if error:
            self.error_handler.handle_error(error, self.name)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        self.logger.critical(message, exc_info=error is not None, extra=kwargs)
        if error:
            self.error_handler.handle_error(error, self.name)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.info(f"Performance: {operation} took {duration:.3f}s", 
                 operation=operation, duration=duration, **metrics)
    
    def get_error_summary(self) -> Dict:
        """Get error summary from error handler."""
        return self.error_handler.get_error_summary()


class PerformanceMonitor:
    """
    Monitor and log performance metrics.
    """
    
    def __init__(self, logger: Optional[LoggerAdapter] = None):
        self.logger = logger
        self.metrics = {}
        self.timers = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timers[operation] = time.time()
    
    def end_timer(self, operation: str, log: bool = True) -> float:
        """End timing and return duration."""
        if operation not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[operation]
        del self.timers[operation]
        
        # Update metrics
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        metrics = self.metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)
        
        if log and self.logger:
            self.logger.log_performance(operation, duration)
        
        return duration
    
    def get_metrics(self) -> Dict:
        """Get performance metrics summary."""
        summary = {}
        for operation, metrics in self.metrics.items():
            if metrics['count'] > 0:
                summary[operation] = {
                    'count': metrics['count'],
                    'avg_time': metrics['total_time'] / metrics['count'],
                    'min_time': metrics['min_time'],
                    'max_time': metrics['max_time'],
                    'total_time': metrics['total_time']
                }
        return summary
    
    @contextlib.contextmanager
    def measure(self, operation: str, log: bool = True):
        """Context manager for measuring operation duration."""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, log)


# Import for context manager
import contextlib