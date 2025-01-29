import pandas as pd
import numpy as np
import talib
from loguru import logger
from typing import Dict, Any, List, Optional

class CustomTradingStrategy:
    SUPPORTED_PAIRS = [
        'NZDUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 
        'USDCAD', 'USDCHF', 'USDJPY'
    ]

    def __init__(
        self, 
        symbol: str = 'NZDUSD', 
        timeframes: List[str] = ['daily', 'weekly', 'hourly', '15min', '5min']
    ):
        """
        Initialize the trading strategy with a specific currency pair.
        
        :param symbol: Trading symbol (default is NZDUSD)
        :param timeframes: List of timeframes to analyze
        """
        if symbol not in self.SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported currency pair. Supported pairs are: {self.SUPPORTED_PAIRS}")
        
        self.symbol = symbol
        self.timeframes = timeframes
        
        # Strategy state variables
        self.daily_bias: Optional[str] = None
        self.weekly_range: Optional[Dict[str, float]] = None
        self.liquidity_levels: Dict[str, List[float]] = {}
        self.fair_value_gaps: Dict[str, List[Dict]] = {}
        
        # Logging setup
        logger.add(f"logs/trade_logs/{symbol}_strategy.log", rotation="10 MB")

    def analyze_daily_bias(self, daily_data: pd.DataFrame) -> str:
        """
        Determine market bias using technical indicators.
        
        :param daily_data: Daily price data
        :return: Market bias
        """
        try:
            # Calculate moving averages
            sma_50 = talib.SMA(daily_data['close'], timeperiod=50)
            sma_200 = talib.SMA(daily_data['close'], timeperiod=200)
            
            # Recent close price
            recent_close = daily_data['close'].iloc[-1]
            
            # Determine bias based on moving average crossover and recent price action
            if recent_close > sma_50.iloc[-1] and sma_50.iloc[-1] > sma_200.iloc[-1]:
                self.daily_bias = 'bullish'
            elif recent_close < sma_50.iloc[-1] and sma_50.iloc[-1] < sma_200.iloc[-1]:
                self.daily_bias = 'bearish'
            else:
                self.daily_bias = 'neutral'
            
            logger.info(f"{self.symbol} Daily Bias: {self.daily_bias}")
            return self.daily_bias
        
        except Exception as e:
            logger.error(f"Error in daily bias analysis: {e}")
            return 'neutral'

    def identify_liquidity_levels(self, hourly_data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key liquidity levels using technical analysis.
        
        :param hourly_data: Hourly price data
        :return: Dictionary of liquidity levels
        """
        try:
            # Use pivots and swing highs/lows as liquidity levels
            high = hourly_data['high']
            low = hourly_data['low']
            
            # Calculate pivots
            pivot_high = talib.CDLHAMMER(high, low, high)
            pivot_low = talib.CDLHANGINGMAN(high, low, low)
            
            # Find significant swing points
            swing_highs = [high.iloc[i] for i in range(1, len(high)-1) 
                           if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]]
            
            swing_lows = [low.iloc[i] for i in range(1, len(low)-1) 
                          if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i+1]]
            
            self.liquidity_levels = {
                'major_support': swing_lows[:3],  # Top 3 low levels
                'major_resistance': swing_highs[:3],  # Top 3 high levels
                'pivot_highs': [high.iloc[i] for i, val in enumerate(pivot_high) if val != 0],
                'pivot_lows': [low.iloc[i] for i, val in enumerate(pivot_low) if val != 0]
            }
            
            logger.info(f"{self.symbol} Liquidity Levels Identified")
            return self.liquidity_levels
        
        except Exception as e:
            logger.error(f"Error identifying liquidity levels: {e}")
            return {}

    def generate_entry_signal(
        self, 
        current_price: float, 
        lower_timeframe_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate trading entry signal based on strategy rules.
        
        :param current_price: Current market price
        :param lower_timeframe_data: Lower timeframe price data
        :return: Entry signal details
        """
        try:
            # Default signal structure
            entry_signal = {
                'signal': None,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward_ratio': None
            }
            
            # Check liquidity level breakout conditions
            major_resistance = max(self.liquidity_levels.get('major_resistance', [0]))
            major_support = min(self.liquidity_levels.get('major_support', [0]))
            
            # Bullish entry conditions
            if (self.daily_bias == 'bullish' and 
                current_price > major_resistance):
                
                # Find fair value gap for bullish entry
                bullish_gaps = self._find_fair_value_gaps(lower_timeframe_data, 'bullish')
                
                if bullish_gaps:
                    entry_price = bullish_gaps[0]['entry']
                    stop_loss = major_support
                    take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 risk-reward
                    
                    entry_signal = {
                        'signal': 'BUY',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': 2
                    }
            
            # Bearish entry conditions
            elif (self.daily_bias == 'bearish' and 
                  current_price < major_support):
                
                # Find fair value gap for bearish entry
                bearish_gaps = self._find_fair_value_gaps(lower_timeframe_data, 'bearish')
                
                if bearish_gaps:
                    entry_price = bearish_gaps[0]['entry']
                    stop_loss = major_resistance
                    take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 risk-reward
                    
                    entry_signal = {
                        'signal': 'SELL',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk_reward_ratio': 2
                    }
            
            logger.info(f"{self.symbol} Entry Signal: {entry_signal}")
            return entry_signal
        
        except Exception as e:
            logger.error(f"Error generating entry signal: {e}")
            return entry_signal

    def _find_fair_value_gaps(
        self, 
        lower_timeframe_data: pd.DataFrame, 
        gap_type: str
    ) -> List[Dict[str, float]]:
        """
        Identify fair value gaps in lower timeframes.
        
        :param lower_timeframe_data: Lower timeframe price data
        :param gap_type: Type of gap ('bullish' or 'bearish')
        :return: List of fair value gaps
        """
        try:
            gaps = []
            for i in range(1, len(lower_timeframe_data) - 1):
                prev_candle = lower_timeframe_data.iloc[i-1]
                curr_candle = lower_timeframe_data.iloc[i]
                next_candle = lower_timeframe_data.iloc[i+1]
                
                # Bullish fair value gap
                if (gap_type == 'bullish' and 
                    curr_candle['close'] > prev_candle['high'] and 
                    curr_candle['close'] > next_candle['high']):
                    gaps.append({
                        'entry': curr_candle['close'],
                        'gap_start': prev_candle['high'],
                        'gap_end': curr_candle['close']
                    })
                
                # Bearish fair value gap
                elif (gap_type == 'bearish' and 
                      curr_candle['close'] < prev_candle['low'] and 
                      curr_candle['close'] < next_candle['low']):
                    gaps.append({
                        'entry': curr_candle['close'],
                        'gap_start': prev_candle['low'],
                        'gap_end': curr_candle['close']
                    })
            
            return gaps
        
        except Exception as e:
            logger.error(f"Error finding fair value gaps: {e}")
            return []

    @classmethod
    def get_supported_pairs(cls) -> List[str]:
        """
        Retrieve list of supported currency pairs.
        
        :return: List of supported pairs
        """
        return cls.SUPPORTED_PAIRS

# Example of dynamic pair selection
if __name__ == "__main__":
    # Get supported pairs
    supported_pairs = CustomTradingStrategy.get_supported_pairs()
    print("Supported Currency Pairs:", supported_pairs)
    
    # Initialize strategy with a specific pair
    strategy = CustomTradingStrategy(symbol='NZDUSD')