import os
import sys
from typing import Optional, Union, Dict, Any
from loguru import logger
from datetime import datetime

class LoggerManager:
    """
    Centralized logging management system for the trading bot.
    Provides advanced logging capabilities with multiple output streams.
    """
    
    def __init__(
        self, 
        log_dir: Optional[str] = None,
        log_level: str = 'INFO',
        max_log_size: str = '10 MB',
        backup_count: int = 5
    ):
        """
        Initialize logging system with configurable parameters.
        
        :param log_dir: Directory to store log files
        :param log_level: Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        :param max_log_size: Maximum size of log files before rotation
        :param backup_count: Number of backup log files to keep
        """
        # Normalize and validate log level
        self.log_level = log_level.upper()
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            raise ValueError(f"Invalid log level. Choose from {valid_levels}")
        
        # Create log directory
        self.log_dir = log_dir or os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Configure console logging
        logger.add(
            sys.stderr, 
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # Configure file logging
        self._setup_file_logging(max_log_size, backup_count)
    
    def _setup_file_logging(
        self, 
        max_log_size: str, 
        backup_count: int
    ):
        """
        Set up file logging with rotation and retention.
        
        :param max_log_size: Maximum log file size
        :param backup_count: Number of backup log files
        """
        # General application log
        logger.add(
            os.path.join(self.log_dir, 'app_{time}.log'),
            level=self.log_level,
            rotation=max_log_size,
            compression="zip",
            retention=backup_count,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )
        
        # Error-specific log
        logger.add(
            os.path.join(self.log_dir, 'errors_{time}.log'),
            level="ERROR",
            rotation=max_log_size,
            compression="zip",
            retention=backup_count,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )
    
    def log_trade(
        self, 
        trade_details: Dict[str, Any],
        log_level: str = 'INFO'
    ):
        """
        Log trade-specific information.
        
        :param trade_details: Dictionary of trade details
        :param log_level: Logging level for the trade
        """
        log_method = getattr(logger, log_level.lower(), logger.info)
        
        trade_log = (
            "TRADE DETAILS\n"
            f"Symbol: {trade_details.get('symbol', 'N/A')}\n"
            f"Type: {trade_details.get('type', 'N/A')}\n"
            f"Entry Price: {trade_details.get('entry_price', 'N/A')}\n"
            f"Exit Price: {trade_details.get('exit_price', 'N/A')}\n"
            f"Volume: {trade_details.get('volume', 'N/A')}\n"
            f"Profit/Loss: {trade_details.get('profit_loss', 'N/A')}"
        )
        
        log_method(trade_log)
    
    def log_system_event(
        self, 
        event_type: str, 
        message: str, 
        log_level: str = 'INFO'
    ):
        """
        Log system-wide events.
        
        :param event_type: Type of system event
        :param message: Detailed event message
        :param log_level: Logging level for the event
        """
        log_method = getattr(logger, log_level.lower(), logger.info)
        
        system_log = (
            f"SYSTEM EVENT: {event_type}\n"
            f"Timestamp: {datetime.now()}\n"
            f"Message: {message}"
        )
        
        log_method(system_log)
    
    def log_error(
        self, 
        error: Union[Exception, str], 
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log detailed error information.
        
        :param error: Exception or error message
        :param context: Additional context for the error
        """
        error_log = "ERROR DETAILS\n"
        
        if isinstance(error, Exception):
            error_log += (
                f"Type: {type(error).__name__}\n"
                f"Message: {str(error)}\n"
            )
        else:
            error_log += f"Message: {error}\n"
        
        if context:
            error_log += "Context:\n"
            for key, value in context.items():
                error_log += f"{key}: {value}\n"
        
        logger.exception(error_log)
    
    def set_log_level(self, new_level: str):
        """
        Dynamically change logging level.
        
        :param new_level: New logging level
        """
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        if new_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Choose from {valid_levels}")
        
        # Remove existing handlers
        logger.remove()
        
        # Reconfigure with new level
        self.__init__(
            log_dir=self.log_dir, 
            log_level=new_level
        )
    
    def get_log_path(self, log_type: str = 'app') -> str:
        """
        Get the path to the most recent log file.
        
        :param log_type: Type of log (app/error)
        :return: Path to the most recent log file
        """
        log_pattern = f'{log_type}_*.log'
        
        # Find most recent log file
        log_files = [
            os.path.join(self.log_dir, f) 
            for f in os.listdir(self.log_dir) 
            if f.startswith(log_pattern.replace('*', ''))
        ]
        
        if not log_files:
            raise FileNotFoundError(f"No {log_type} log files found")
        
        return max(log_files, key=os.path.getctime)

# Global logger instance
trading_logger = LoggerManager()

# Expose loguru logger for direct use
log = logger

# Example usage demonstrator
if __name__ == "__main__":
    # Demonstrate different logging methods
    
    # System event logging
    trading_logger.log_system_event(
        "BOT_STARTUP", 
        "Trading bot initialized successfully"
    )
    
    # Trade logging
    trade_details = {
        'symbol': 'NZDUSD',
        'type': 'BUY',
        'entry_price': 0.6500,
        'exit_price': 0.6600,
        'volume': 0.1,
        'profit_loss': 100
    }
    trading_logger.log_trade(trade_details)
    
    # Error logging
    try:
        # Simulate an error
        1 / 0
    except Exception as e:
        trading_logger.log_error(
            e, 
            context={'operation': 'division', 'values': (1, 0)}
        )
    
    # Demonstrate log level change
    trading_logger.set_log_level('DEBUG')
    
    # Get latest log path
    latest_log = trading_logger.get_log_path()
    print(f"Latest log file: {latest_log}")