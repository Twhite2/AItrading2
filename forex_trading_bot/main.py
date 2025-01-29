import os
import sys
import argparse
import logging
from typing import Optional

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import key components
from api.app import start_server
from strategies.custom_strategy import CustomTradingStrategy
from utils.data_loader import DataLoader
from utils.trade_executor import TradeExecutor
from models.model_utils import ModelUtils
from strategies.risk_management import RiskManager
from utils.logger import trading_logger
from backtesting.lumibot_backtest import run_backtest, generate_backtest_report
from utils.commission_tracker import CommissionTracker

class TradingBotManager:
    """
    Comprehensive manager for the AI-powered forex trading bot.
    Handles initialization, configuration, and execution of bot components.
    """
    
    def __init__(
        self, 
        symbol: str = 'NZDUSD', 
        initial_capital: float = 10000.0
    ):
        """
        Initialize trading bot components.
        
        :param symbol: Primary trading symbol
        :param initial_capital: Initial trading capital
        """
        # Core components
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Component instances
        self.data_loader = None
        self.trade_executor = None
        self.strategy = None
        self.model_utils = None
        self.risk_manager = None
        self.commission_tracker = None
        
        # Initialization flag
        self.is_initialized = False
    
    def initialize_components(self):
        """
        Initialize all trading bot components.
        """
        try:
            # Initialize components
            self.data_loader = DataLoader(symbol=self.symbol)
            self.trade_executor = TradeExecutor(symbol=self.symbol)
            self.strategy = CustomTradingStrategy(symbol=self.symbol)
            self.model_utils = ModelUtils()
            self.risk_manager = RiskManager(
                initial_balance=self.initial_capital
            )
            self.commission_tracker = CommissionTracker()
            
            # Log initialization
            trading_logger.log_system_event(
                "BOT_INITIALIZATION", 
                f"Trading bot initialized for {self.symbol}"
            )
            
            self.is_initialized = True
        
        except Exception as e:
            trading_logger.log_error(
                e, 
                {"context": "Bot component initialization"}
            )
            raise
    
    def run_backtest(
        self, 
        symbols: Optional[list] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Execute backtesting for specified symbols.
        
        :param symbols: List of symbols to backtest
        :param start_date: Backtest start date
        :param end_date: Backtest end date
        :return: Backtest results
        """
        try:
            # Use default symbol if not specified
            test_symbols = symbols or [self.symbol]
            
            # Run backtest
            backtest_results = run_backtest(
                symbols=test_symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital
            )
            
            # Generate report
            generate_backtest_report(backtest_results)
            
            # Log backtest results
            trading_logger.log_system_event(
                "BACKTEST_COMPLETE", 
                f"Backtest completed for {test_symbols}"
            )
            
            return backtest_results
        
        except Exception as e:
            trading_logger.log_error(
                e, 
                {"context": "Backtesting execution"}
            )
            raise
    
    def start_live_trading(self):
        """
        Start live trading operations.
        """
        try:
            # Ensure components are initialized
            if not self.is_initialized:
                self.initialize_components()
            
            # Start data stream
            self.data_loader.start_realtime_data_stream()
            
            # Main trading logic
            while True:
                try:
                    # Fetch real-time market data
                    market_data = self.data_loader.get_realtime_data()
                    
                    if market_data:
                        # Generate AI insights
                        ai_insights = self.model_utils.generate_market_insight(
                            market_data
                        )
                        
                        # Generate trading signal
                        entry_signal = self.strategy.generate_entry_signal(
                            market_data
                        )
                        
                        # Execute trade if signal is valid
                        if entry_signal and entry_signal['signal']:
                            trade = self.trade_executor.place_order(
                                order_type='market',
                                side=entry_signal['signal'].lower(),
                                volume=0.1,  # Configurable
                                price=entry_signal['entry_price']
                            )
                            
                            # Track commission
                            self.commission_tracker.calculate_commission(
                                symbol=self.symbol,
                                trade_type=entry_signal['signal'].lower(),
                                entry_price=entry_signal['entry_price'],
                                exit_price=entry_signal['take_profit'],
                                volume=0.1
                            )
                
                except Exception as trade_error:
                    trading_logger.log_error(
                        trade_error, 
                        {"context": "Live trading iteration"}
                    )
        
        except Exception as e:
            trading_logger.log_error(
                e, 
                {"context": "Live trading startup"}
            )
    
    def stop_trading(self):
        """
        Gracefully stop trading operations.
        """
        try:
            # Stop data stream
            if self.data_loader:
                self.data_loader.stop_realtime_data_stream()
            
            # Log shutdown
            trading_logger.log_system_event(
                "BOT_SHUTDOWN", 
                f"Trading bot for {self.symbol} stopped"
            )
        
        except Exception as e:
            trading_logger.log_error(
                e, 
                {"context": "Trading shutdown"}
            )

def main():
    """
    Main entry point for the trading bot application.
    Parse command-line arguments and execute appropriate action.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="AI-Powered Forex Trading Bot"
    )
    
    # Define command-line arguments
    parser.add_argument(
        '--mode', 
        choices=['backtest', 'live', 'server'], 
        default='server', 
        help='Execution mode for the trading bot'
    )
    parser.add_argument(
        '--symbol', 
        default='NZDUSD', 
        help='Primary trading symbol'
    )
    parser.add_argument(
        '--capital', 
        type=float, 
        default=10000.0, 
        help='Initial trading capital'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create bot manager
    bot_manager = TradingBotManager(
        symbol=args.symbol, 
        initial_capital=args.capital
    )
    
    # Execute based on mode
    try:
        if args.mode == 'backtest':
            # Run backtesting
            bot_manager.run_backtest()
        
        elif args.mode == 'live':
            # Start live trading
            bot_manager.start_live_trading()
        
        else:  # Default to server mode
            # Start web service
            start_server()
    
    except Exception as e:
        trading_logger.log_error(e, {"context": "Main execution"})
        sys.exit(1)

if __name__ == "__main__":
    main()