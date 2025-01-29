import os
import yaml
import json
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List, Optional, Type

# Lumibot imports
from lumibot.brokers import Broker
from lumibot.strategies import Strategy
from lumibot.backtesting import BacktestingBroker
from lumibot.traders import Trader

class LumibotBacktestConfiguration:
    """
    Comprehensive Lumibot backtesting configuration and execution manager.
    """
    
    def __init__(
        self, 
        config_file: Optional[str] = None,
        strategy_class: Optional[Type[Strategy]] = None
    ):
        """
        Initialize backtesting configuration.
        
        :param config_file: Path to configuration file
        :param strategy_class: Custom strategy class for backtesting
        """
        # Default configuration
        self.config = {
            'backtesting': {
                'start_date': datetime.now(pytz.UTC) - timedelta(days=365),
                'end_date': datetime.now(pytz.UTC),
                'initial_capital': 10000.0,
                'benchmark': '^GSPC',  # S&P 500 as default benchmark
                'commission_rate': 0.0005,  # 0.05%
            },
            'symbols': [
                'NZDUSD',
                'EURUSD',
                'GBPUSD'
            ],
            'strategy_params': {
                'risk_management': {
                    'max_risk_per_trade': 0.02,  # 2% of account
                    'stop_loss_multiplier': 2.0
                },
                'technical_indicators': {
                    'rsi_period': 14,
                    'sma_short': 50,
                    'sma_long': 200
                }
            }
        }
        
        # Strategy class
        self.strategy_class = strategy_class
        
        # Load configuration if file provided
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """
        Load configuration from file.
        
        :param config_file: Path to configuration file
        """
        try:
            # Determine file type
            file_ext = os.path.splitext(config_file)[1].lower()
            
            with open(config_file, 'r') as f:
                if file_ext in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif file_ext == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
                
                # Deep merge configurations
                self._deep_merge(self.config, loaded_config)
        
        except Exception as e:
            print(f"Configuration loading error: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict):
        """
        Recursively merge dictionaries.
        
        :param base: Base configuration
        :param update: Update configuration
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    def create_backtest_broker(self) -> BacktestingBroker:
        """
        Create a Lumibot backtesting broker.
        
        :return: Configured BacktestingBroker
        """
        return BacktestingBroker(
            start_date=self.config['backtesting']['start_date'],
            end_date=self.config['backtesting']['end_date'],
            benchmark=self.config['backtesting']['benchmark'],
            budget=self.config['backtesting']['initial_capital']
        )
    
    def run_backtest(
        self, 
        symbols: Optional[List[str]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtesting.
        
        :param symbols: List of symbols to test (overrides config)
        :param save_results: Whether to save backtest results
        :return: Backtest results
        """
        # Determine symbols to test
        test_symbols = symbols or self.config['symbols']
        
        # Validate strategy class
        if not self.strategy_class:
            raise ValueError("No strategy class provided for backtesting")
        
        # Create backtesting broker
        broker = self.create_backtest_broker()
        
        # Prepare results storage
        backtest_results = {
            'overall_results': {},
            'symbol_results': {}
        }
        
        # Run backtests for each symbol
        for symbol in test_symbols:
            try:
                # Instantiate strategy with symbol-specific parameters
                strategy_params = self.config['strategy_params'].copy()
                strategy_params['symbol'] = symbol
                
                strategy = self.strategy_class(
                    broker=broker,
                    **strategy_params
                )
                
                # Run backtest
                trader = Trader()
                trader.add_strategy(strategy)
                
                # Collect and store results
                symbol_results = trader.run_backtests()
                backtest_results['symbol_results'][symbol] = symbol_results
            
            except Exception as e:
                print(f"Backtest failed for {symbol}: {e}")
        
        # Aggregate overall results
        backtest_results['overall_results'] = self._aggregate_backtest_results(
            backtest_results['symbol_results']
        )
        
        # Save results if requested
        if save_results:
            self._save_backtest_results(backtest_results)
        
        return backtest_results
    
    def _aggregate_backtest_results(
        self, 
        symbol_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate backtest results across symbols.
        
        :param symbol_results: Results for individual symbols
        :return: Aggregated results
        """
        # Basic aggregation logic
        total_return = sum(
            results.get('total_return', 0) 
            for results in symbol_results.values()
        )
        
        avg_annual_return = sum(
            results.get('annual_return', 0) 
            for results in symbol_results.values()
        ) / len(symbol_results)
        
        return {
            'total_return': total_return,
            'average_annual_return': avg_annual_return,
            'symbols_tested': list(symbol_results.keys())
        }
    
    def _save_backtest_results(
        self, 
        results: Dict[str, Any]
    ):
        """
        Save backtest results to file.
        
        :param results: Backtest results to save
        """
        # Create results directory
        results_dir = os.path.join(os.getcwd(), 'backtest_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"Backtest results saved to {filepath}")
    
    def export_config(
        self, 
        export_format: str = 'yaml',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export current configuration to a file.
        
        :param export_format: Output file format
        :param output_path: Custom output path
        :return: Path to exported configuration file
        """
        # Determine output path
        if not output_path:
            config_dir = os.path.join(os.getcwd(), 'config')
            os.makedirs(config_dir, exist_ok=True)
            output_path = os.path.join(
                config_dir, 
                f'lumibot_backtest_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
        
        # Export based on format
        try:
            if export_format.lower() == 'yaml':
                full_path = f"{output_path}.yaml"
                with open(full_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif export_format.lower() == 'json':
                full_path = f"{output_path}.json"
                with open(full_path, 'w') as f:
                    json.dump(self.config, f, indent=4)
            else:
                raise ValueError("Supported formats: yaml, json")
            
            return full_path
        
        except Exception as e:
            print(f"Configuration export error: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Custom strategy class would be defined here
    class ExampleTradingStrategy(Strategy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
    # Initialize backtest configuration
    backtest_config = LumibotBacktestConfiguration(
        strategy_class=ExampleTradingStrategy
    )
    
    # Run backtest
    results = backtest_config.run_backtest()
    print("Backtest Results:", results)