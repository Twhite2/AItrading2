import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from lumibot.brokers import Broker
from lumibot.strategies import Strategy
from lumibot.backtesting import BacktestingBroker
from lumibot.traders import Trader

# Import custom components
from models.model_utils import ModelUtils
from utils.technical_indicators import TechnicalIndicators
from strategies.risk_management import RiskManager

class LumibotBacktestStrategy(Strategy):
    """
    Advanced backtesting strategy combining 
    AI insights and technical analysis.
    """
    
    def __init__(
        self, 
        broker: Broker, 
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize backtesting strategy.
        
        :param broker: Broker instance
        :param params: Strategy configuration parameters
        """
        # Call parent constructor
        super().__init__(broker, params)
        
        # Default parameters
        self.symbol = params.get('symbol', 'NZDUSD')
        self.initial_capital = params.get('initial_capital', 10000)
        self.risk_per_trade = params.get('risk_per_trade', 0.01)
        
        # Component initialization
        self.model_utils = ModelUtils()
        self.risk_manager = RiskManager(initial_balance=self.initial_capital)
    
    def initialize(self):
        """
        Strategy initialization method.
        Set up trading frequency and other initial configurations.
        """
        # Set trading frequency (e.g., daily)
        self.sleeptime = 60 * 60 * 24  # 24 hours
        
        # Set initial portfolio allocation
        self.portfolio_method = 'risk_parity'
    
    def prepare_data(self):
        """
        Prepare market data for strategy analysis.
        
        :return: Processed market data DataFrame
        """
        try:
            # Fetch historical data
            historical_data = self.get_historical_prices(
                self.symbol, 
                days_back=365  # One year of historical data
            )
            
            # Convert to DataFrame
            df = historical_data.get_df()
            
            # Calculate technical indicators
            tech_indicators = TechnicalIndicators(df)
            
            # Prepare comprehensive feature set
            features = {
                'sma_50': tech_indicators.calculate_sma(timeperiod=50),
                'sma_200': tech_indicators.calculate_sma(timeperiod=200),
                'rsi': tech_indicators.calculate_rsi(),
                'macd': tech_indicators.calculate_macd(),
                'market_structure': tech_indicators.get_market_structure()
            }
            
            return df, features
        
        except Exception as e:
            self.logger.error(f"Data preparation error: {e}")
            raise
    
    def generate_signals(self, data: pd.DataFrame, features: Dict[str, Any]):
        """
        Generate trading signals based on AI and technical analysis.
        
        :param data: Market price data
        :param features: Technical indicator features
        :return: Trading signal details
        """
        try:
            # Generate AI insights
            ai_insights = self.model_utils.generate_market_insight(
                data, 
                insight_type='trading_signal'
            )
            
            # Assess AI confidence
            confidence = self.model_utils.evaluate_model_confidence(data)
            
            # Technical analysis signals
            rsi = features['rsi']
            macd = features['macd']
            market_structure = features['market_structure']
            
            # Signal generation logic
            current_price = data['close'].iloc[-1]
            
            # Bullish signal conditions
            if (rsi[-1] < 30 and  # Oversold
                macd['macd'][-1] > 0 and  # Positive MACD
                confidence['trend_prediction'] > 0.6):  # High AI confidence
                
                stop_loss = market_structure['last_swing_low']
                take_profit = current_price + (current_price - stop_loss) * 2
                
                return {
                    'signal': 'BUY',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence
                }
            
            # Bearish signal conditions
            elif (rsi[-1] > 70 and  # Overbought
                  macd['macd'][-1] < 0 and  # Negative MACD
                  confidence['trend_prediction'] > 0.6):  # High AI confidence
                
                stop_loss = market_structure['last_swing_high']
                take_profit = current_price - (stop_loss - current_price) * 2
                
                return {
                    'signal': 'SELL',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence
                }
            
            # No strong signal
            return None
        
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            raise
    
    def on_trading_iteration(self):
        """
        Main trading logic for each iteration.
        """
        try:
            # Prepare market data
            data, features = self.prepare_data()
            
            # Generate trading signal
            signal = self.generate_signals(data, features)
            
            if signal:
                # Calculate position size based on risk management
                risk_assessment = self.risk_manager.assess_trade_risk({
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['take_profit']
                })
                
                # Determine trade quantity
                quantity = risk_assessment['position_size']
                
                # Execute trade based on signal
                if signal['signal'] == 'BUY':
                    self.buy(self.symbol, quantity)
                elif signal['signal'] == 'SELL':
                    self.sell(self.symbol, quantity)
        
        except Exception as e:
            self.logger.error(f"Trading iteration error: {e}")
    
    def on_trade_result(self, trade_result):
        """
        Process trade results and update risk management.
        
        :param trade_result: Details of completed trade
        """
        try:
            # Update risk management metrics
            self.risk_manager.update_trade_metrics({
                'profit_loss': trade_result.get('realized_pnl', 0),
                'risk_reward_ratio': trade_result.get('risk_reward_ratio', 0)
            })
        
        except Exception as e:
            self.logger.error(f"Trade result processing error: {e}")

def run_backtest(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run comprehensive Lumibot backtest.
    
    :param symbols: List of symbols to backtest
    :param start_date: Backtest start date
    :param end_date: Backtest end date
    :param initial_capital: Initial trading capital
    :return: Backtest results
    """
    # Default symbols if not provided
    symbols = symbols or ['NZDUSD', 'EURUSD', 'GBPUSD']
    
    # Backtest results storage
    backtest_results = {}
    
    for symbol in symbols:
        try:
            # Create backtesting broker
            broker = BacktestingBroker(
                start_date=start_date or pd.Timestamp.now() - pd.Timedelta(days=365),
                end_date=end_date or pd.Timestamp.now(),
                benchmark='^GSPC'  # S&P 500 benchmark
            )
            
            # Initialize strategy
            strategy_params = {
                'symbol': symbol,
                'initial_capital': initial_capital,
                'risk_per_trade': 0.01
            }
            
            strategy = LumibotBacktestStrategy(
                broker=broker, 
                params=strategy_params
            )
            
            # Run backtest
            trader = Trader()
            trader.add_strategy(strategy)
            results = trader.run_backtests()
            
            # Store results
            backtest_results[symbol] = results
        
        except Exception as e:
            print(f"Backtest failed for {symbol}: {e}")
    
    return backtest_results

# Visualization and reporting
def generate_backtest_report(backtest_results: Dict[str, Any]):
    """
    Generate comprehensive backtest report.
    
    :param backtest_results: Backtest results dictionary
    """
    # Create reports directory
    reports_dir = os.path.join(os.getcwd(), 'backtest_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Process and save results
    for symbol, results in backtest_results.items():
        report_path = os.path.join(reports_dir, f'{symbol}_backtest_report.csv')
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(report_path)
        
        print(f"Backtest report for {symbol} saved to {report_path}")

# Main execution
if __name__ == "__main__":
    # Run backtest
    results = run_backtest()
    
    # Generate report
    generate_backtest_report(results)
    
    # Print summary
    for symbol, result in results.items():
        print(f"\nBacktest Results for {symbol}:")
        for metric, value in result.items():
            print(f"{metric}: {value}")