import os
import datetime
import pandas as pd
import numpy as np
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers

# Import custom components
from models.model_utils import ModelUtils
from utils.technical_indicators import TechnicalIndicators
from strategies.risk_management import RiskManager

class AITradingStrategy(bt.Strategy):
    """
    Advanced Backtrader trading strategy 
    combining AI insights and technical analysis.
    """
    
    params = (
        ('symbol', 'NZDUSD'),
        ('initial_capital', 10000),
        ('risk_per_trade', 0.01),
        ('rsi_period', 14),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9)
    )
    
    def __init__(self):
        """
        Initialize strategy components.
        """
        # Store references to data feeds
        self.dataclose = self.datas[0].close
        self.volume = self.datas[0].volume
        
        # Indicators
        self.rsi = bt.indicators.RSI(
            self.data.close, 
            period=self.params.rsi_period
        )
        
        self.macd = bt.indicators.MACD(
            self.data.close, 
            period_me1=self.params.macd_fast, 
            period_me2=self.params.macd_slow, 
            period_signal=self.params.macd_signal
        )
        
        # AI and risk management components
        self.model_utils = ModelUtils()
        self.risk_manager = RiskManager(
            initial_balance=self.params.initial_capital
        )
        
        # Tracking variables
        self.order = None
        self.position_size = 0
    
    def generate_ai_insights(self):
        """
        Generate AI-powered market insights.
        
        :return: AI insights dictionary
        """
        try:
            # Convert current data to DataFrame
            df = pd.DataFrame({
                'open': [self.data.open[0]],
                'high': [self.data.high[0]],
                'low': [self.data.low[0]],
                'close': [self.data.close[0]],
                'volume': [self.data.volume[0]]
            })
            
            # Generate AI insights
            insights = self.model_utils.generate_market_insight(
                df, 
                insight_type='trading_signal'
            )
            
            # Evaluate model confidence
            confidence = self.model_utils.evaluate_model_confidence(df)
            
            return {
                'insights': insights,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"AI insight generation error: {e}")
            return None
    
    def assess_trade_risk(self, entry_price, stop_loss, take_profit):
        """
        Assess trade risk parameters.
        
        :param entry_price: Trade entry price
        :param stop_loss: Stop loss price
        :param take_profit: Take profit price
        :return: Risk assessment details
        """
        return self.risk_manager.assess_trade_risk({
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })
    
    def next(self):
        """
        Main trading logic for each iteration.
        """
        # Skip if an order is pending
        if self.order:
            return
        
        # Generate AI insights
        ai_insights = self.generate_ai_insights()
        
        # Current market conditions
        current_close = self.dataclose[0]
        
        # Trading logic
        if not self.position:
            # Buy conditions
            if (self.rsi[0] < 30 and  # Oversold
                self.macd.macd[0] > 0 and  # Positive MACD
                ai_insights and 
                ai_insights['confidence']['trend_prediction'] > 0.6):
                
                # Calculate position size
                risk_assessment = self.assess_trade_risk(
                    entry_price=current_close,
                    stop_loss=self.data.low[0],
                    take_profit=current_close * 1.02  # 2% take profit
                )
                
                # Place buy order
                self.position_size = risk_assessment['position_size']
                self.order = self.buy(size=self.position_size)
        
        else:
            # Sell conditions
            if (self.rsi[0] > 70 and  # Overbought
                self.macd.macd[0] < 0 and  # Negative MACD
                ai_insights and 
                ai_insights['confidence']['trend_prediction'] > 0.6):
                
                # Place sell order
                self.order = self.close()
    
    def notify_order(self, order):
        """
        Order status tracking.
        
        :param order: Backtrader order object
        """
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        # Check if order is complete
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'BUY EXECUTED, {order.executed.price}')
            elif order.issell():
                print(f'SELL EXECUTED, {order.executed.price}')
            
            # Update trade metrics
            self.risk_manager.update_trade_metrics({
                'profit_loss': order.executed.value,
                'trade_type': 'buy' if order.isbuy() else 'sell'
            })
        
        # Reset order
        self.order = None

def prepare_data_feed(
    symbol: str, 
    start_date: datetime.date, 
    end_date: datetime.date
) -> btfeeds.PandasData:
    """
    Prepare Backtrader data feed from historical data.
    
    :param symbol: Trading symbol
    :param start_date: Start date for backtesting
    :param end_date: End date for backtesting
    :return: Backtrader data feed
    """
    # Fetch historical data (replace with your data fetching method)
    # This is a placeholder - you'd typically use your data loader
    df = pd.DataFrame({
        'datetime': pd.date_range(start=start_date, end=end_date),
        'open': np.random.random(len(pd.date_range(start=start_date, end=end_date))) + 50,
        'high': np.random.random(len(pd.date_range(start=start_date, end=end_date))) + 51,
        'low': np.random.random(len(pd.date_range(start=start_date, end=end_date))) + 49,
        'close': np.random.random(len(pd.date_range(start=start_date, end=end_date))) + 50,
        'volume': np.random.randint(1000, 10000, len(pd.date_range(start=start_date, end=end_date)))
    })
    df.set_index('datetime', inplace=True)
    
    # Convert to Backtrader data feed
    data = btfeeds.PandasData(dataname=df)
    return data

def run_backtest(
    symbols: list = ['NZDUSD', 'EURUSD', 'GBPUSD'],
    start_date: datetime.date = datetime.date(2022, 1, 1),
    end_date: datetime.date = datetime.date(2023, 1, 1),
    initial_capital: float = 10000.0
) -> dict:
    """
    Run comprehensive Backtrader backtest.
    
    :param symbols: List of symbols to backtest
    :param start_date: Backtest start date
    :param end_date: Backtest end date
    :param initial_capital: Initial trading capital
    :return: Backtest results dictionary
    """
    # Backtest results storage
    backtest_results = {}
    
    for symbol in symbols:
        try:
            # Create Cerebro engine
            cerebro = bt.Cerebro()
            
            # Add data feed
            data = prepare_data_feed(symbol, start_date, end_date)
            cerebro.adddata(data)
            
            # Add strategy
            cerebro.addstrategy(
                AITradingStrategy,
                symbol=symbol,
                initial_capital=initial_capital,
                risk_per_trade=0.01
            )
            
            # Set initial capital
            cerebro.broker.setcash(initial_capital)
            
            # Set commission
            cerebro.broker.setcommission(commission=0.0005)  # 0.05% commission
            
            # Add analyzers
            cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_ratio')
            cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
            cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trade_analyzer')
            
            # Run backtest
            results = cerebro.run()[0]
            
            # Collect performance metrics
            performance = {
                'final_value': cerebro.broker.getvalue(),
                'total_return': results.analyzers.returns.get_analysis()['rtot'],
                'sharpe_ratio': results.analyzers.sharpe_ratio.get_analysis()['sharperatio'],
                'max_drawdown': results.analyzers.drawdown.get_analysis()['max']['drawdown'],
                'trades': results.analyzers.trade_analyzer.get_analysis()
            }
            
            backtest_results[symbol] = performance
            
            # Optional: Plot results
            # cerebro.plot()
        
        except Exception as e:
            print(f"Backtest failed for {symbol}: {e}")
    
    return backtest_results

def generate_backtest_report(backtest_results: dict):
    """
    Generate comprehensive backtest report.
    
    :param backtest_results: Backtest results dictionary
    """
    # Create reports directory
    reports_dir = os.path.join(os.getcwd(), 'backtest_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Process and save results
    for symbol, results in backtest_results.items():
        report_path = os.path.join(reports_dir, f'{symbol}_backtrader_report.csv')
        
        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(report_path)
        
        print(f"Backtrader report for {symbol} saved to {report_path}")

# Main execution
if __name__ == "__main__":
    # Run backtest
    results = run_backtest()
    
    # Generate report
    generate_backtest_report(results)
    
    # Print summary
    for symbol, result in results.items():
        print(f"\nBacktrader Results for {symbol}:")
        for metric, value in result.items():
            print(f"{metric}: {value}")