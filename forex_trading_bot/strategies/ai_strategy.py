import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Type

# Import AI and technical analysis components
from models.model_utils import ModelUtils
from utils.technical_indicators import TechnicalIndicators
from strategies.risk_management import RiskManager
from loguru import logger

class AITradingStrategy:
    """
    Advanced AI-driven trading strategy combining 
    machine learning insights with technical analysis.
    """
    
    SUPPORTED_PAIRS = [
        'NZDUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 
        'USDCAD', 'USDCHF', 'USDJPY'
    ]
    
    def __init__(
        self, 
        symbol: str = 'NZDUSD',
        initial_capital: float = 10000.0,
        model_name: str = 'Isotonic/flan-t5-base-trading_candles'
    ):
        """
        Initialize AI-driven trading strategy.
        
        :param symbol: Trading symbol
        :param initial_capital: Initial trading capital
        :param model_name: AI model to use for insights
        """
        # Validate symbol
        if symbol not in self.SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported symbol. Supported pairs: {self.SUPPORTED_PAIRS}")
        
        # Core components
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # AI Model Utilities
        self.model_utils = ModelUtils(model_name)
        
        # Risk Management
        self.risk_manager = RiskManager(initial_balance=initial_capital)
        
        # Logging
        logger.add(f"logs/ai_strategy_{symbol}.log", rotation="10 MB")
    
    def prepare_model_input(
        self, 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Preprocess market data for AI model input.
        
        :param market_data: DataFrame with market data
        :return: Processed input for AI model
        """
        try:
            # Calculate technical indicators
            tech_indicators = TechnicalIndicators(market_data)
            
            # Generate additional features
            indicators = {
                'sma_50': tech_indicators.calculate_sma(timeperiod=50),
                'sma_200': tech_indicators.calculate_sma(timeperiod=200),
                'rsi': tech_indicators.calculate_rsi(),
                'macd': tech_indicators.calculate_macd(),
                'market_structure': tech_indicators.get_market_structure()
            }
            
            # Prepare comprehensive input dictionary
            model_input = {
                'price_data': market_data,
                'technical_indicators': indicators,
                'symbol': self.symbol
            }
            
            return model_input
        
        except Exception as e:
            logger.error(f"Model input preparation error: {e}")
            raise
    
    def generate_ai_insights(
        self, 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate AI-powered market insights.
        
        :param market_data: DataFrame with market data
        :return: AI-generated trading insights
        """
        try:
            # Prepare input for AI model
            model_input = self.prepare_model_input(market_data)
            
            # Generate multiple types of insights
            insights = {
                'trend_prediction': self.model_utils.generate_market_insight(
                    market_data, 
                    insight_type='trend_prediction'
                ),
                'trading_signal': self.model_utils.generate_market_insight(
                    market_data, 
                    insight_type='trading_signal'
                ),
                'volatility_analysis': self.model_utils.generate_market_insight(
                    market_data, 
                    insight_type='volatility_analysis'
                )
            }
            
            # Evaluate model confidence
            confidence = self.model_utils.evaluate_model_confidence(market_data)
            
            return {
                'insights': insights,
                'confidence': confidence
            }
        
        except Exception as e:
            logger.error(f"AI insight generation error: {e}")
            raise
    
    def generate_entry_signal(
        self, 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive trading entry signal.
        
        :param market_data: DataFrame with market data
        :return: Trading entry signal details
        """
        try:
            # Generate AI insights
            ai_insights = self.generate_ai_insights(market_data)
            
            # Technical indicator analysis
            tech_indicators = TechnicalIndicators(market_data)
            rsi = tech_indicators.calculate_rsi()
            macd = tech_indicators.calculate_macd()
            market_structure = tech_indicators.get_market_structure()
            
            # Combine AI and technical indicator signals
            signal_confidence = ai_insights['confidence']
            
            # Default entry signal
            entry_signal = {
                'signal': None,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'confidence': signal_confidence
            }
            
            # Signal generation logic
            # Bullish conditions
            if (rsi[-1] < 30 and  # Oversold RSI
                macd['macd'][-1] > 0 and  # Positive MACD
                signal_confidence['trend_prediction'] > 0.6):  # High AI confidence
                
                entry_price = market_data['close'].iloc[-1]
                stop_loss = market_structure['last_swing_low']
                take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 risk-reward
                
                entry_signal = {
                    'signal': 'BUY',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': signal_confidence
                }
            
            # Bearish conditions
            elif (rsi[-1] > 70 and  # Overbought RSI
                  macd['macd'][-1] < 0 and  # Negative MACD
                  signal_confidence['trend_prediction'] > 0.6):  # High AI confidence
                
                entry_price = market_data['close'].iloc[-1]
                stop_loss = market_structure['last_swing_high']
                take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 risk-reward
                
                entry_signal = {
                    'signal': 'SELL',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': signal_confidence
                }
            
            # Risk assessment
            risk_assessment = self.risk_manager.assess_trade_risk(
                {
                    'entry_price': entry_signal['entry_price'],
                    'stop_loss': entry_signal['stop_loss'],
                    'take_profit': entry_signal['take_profit'],
                    'technical_levels': market_structure
                }
            )
            
            # Combine risk assessment with entry signal
            entry_signal['risk_assessment'] = risk_assessment
            
            logger.info(f"Generated entry signal: {entry_signal}")
            return entry_signal
        
        except Exception as e:
            logger.error(f"Entry signal generation error: {e}")
            raise
    
    def evaluate_trade_performance(
        self, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate performance of executed trade.
        
        :param trade_details: Details of the executed trade
        :return: Trade performance analysis
        """
        try:
            # Update risk management metrics
            self.risk_manager.update_trade_metrics(trade_details)
            
            # Performance analysis
            performance = {
                'profit_loss': trade_details.get('profit_loss', 0),
                'risk_reward_ratio': trade_details.get('risk_reward_ratio', 0),
                'trade_duration': trade_details.get('duration', 0)
            }
            
            # Log performance
            logger.info(f"Trade performance: {performance}")
            
            return performance
        
        except Exception as e:
            logger.error(f"Trade performance evaluation error: {e}")
            raise
    
    @classmethod
    def get_supported_pairs(cls) -> List[str]:
        """
        Get list of supported currency pairs.
        
        :return: List of supported pairs
        """
        return cls.SUPPORTED_PAIRS

# Example usage and testing
if __name__ == "__main__":
    # Create sample market data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'open': np.random.random(100) + 50,
        'high': np.random.random(100) + 51,
        'low': np.random.random(100) + 49,
        'close': np.random.random(100) + 50,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Initialize AI Trading Strategy
    ai_strategy = AITradingStrategy(symbol='NZDUSD')
    
    # Generate entry signal
    entry_signal = ai_strategy.generate_entry_signal(sample_data)
    print("Entry Signal:", entry_signal)
    
    # Simulate trade performance evaluation
    trade_details = {
        'profit_loss': 100,
        'risk_reward_ratio': 2.0,
        'duration': 24  # hours
    }
    performance = ai_strategy.evaluate_trade_performance(trade_details)
    print("Trade Performance:", performance)