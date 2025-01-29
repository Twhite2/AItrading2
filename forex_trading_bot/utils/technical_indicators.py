import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Union, Optional, Any
from loguru import logger

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator using TA-Lib.
    Supports multiple timeframes and calculations.
    """
    
    SUPPORTED_INDICATORS = [
        # Trend Indicators
        'SMA', 'EMA', 'TEMA', 'WMA', 'TRIMA',
        
        # Momentum Indicators
        'RSI', 'MACD', 'STOCH', 'CCI', 'ROC', 'MFI',
        
        # Volatility Indicators
        'BBANDS', 'ATR', 'NATR',
        
        # Pattern Recognition
        'CDLHAMMER', 'CDLENGULFING', 'CDLDOJI',
        
        # Overlap Studies
        'SAR', 'T3',
        
        # Volume Indicators
        'ADOSC', 'OBV'
    ]
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.
        
        :param data: DataFrame with OHLCV data
        """
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.data = data
        self.indicators = {}
        
        # Logging setup
        logger.add("logs/technical_indicators.log", rotation="10 MB")
    
    def calculate_sma(
        self, 
        price_type: str = 'close', 
        timeperiod: int = 14
    ) -> np.ndarray:
        """
        Calculate Simple Moving Average (SMA).
        
        :param price_type: Price type to use (open/high/low/close)
        :param timeperiod: Number of periods for SMA
        :return: SMA values
        """
        try:
            price = self.data[price_type].values
            sma = talib.SMA(price, timeperiod=timeperiod)
            self.indicators[f'SMA_{price_type}_{timeperiod}'] = sma
            return sma
        except Exception as e:
            logger.error(f"SMA calculation error: {e}")
            raise
    
    def calculate_ema(
        self, 
        price_type: str = 'close', 
        timeperiod: int = 14
    ) -> np.ndarray:
        """
        Calculate Exponential Moving Average (EMA).
        
        :param price_type: Price type to use (open/high/low/close)
        :param timeperiod: Number of periods for EMA
        :return: EMA values
        """
        try:
            price = self.data[price_type].values
            ema = talib.EMA(price, timeperiod=timeperiod)
            self.indicators[f'EMA_{price_type}_{timeperiod}'] = ema
            return ema
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            raise
    
    def calculate_rsi(
        self, 
        price_type: str = 'close', 
        timeperiod: int = 14
    ) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        :param price_type: Price type to use (open/high/low/close)
        :param timeperiod: Number of periods for RSI
        :return: RSI values
        """
        try:
            price = self.data[price_type].values
            rsi = talib.RSI(price, timeperiod=timeperiod)
            self.indicators[f'RSI_{price_type}_{timeperiod}'] = rsi
            return rsi
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            raise
    
    def calculate_macd(
        self, 
        price_type: str = 'close', 
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        :param price_type: Price type to use (open/high/low/close)
        :param fastperiod: Fast period
        :param slowperiod: Slow period
        :param signalperiod: Signal period
        :return: Dictionary with MACD components
        """
        try:
            price = self.data[price_type].values
            macd, macdsignal, macdhist = talib.MACD(
                price, 
                fastperiod=fastperiod, 
                slowperiod=slowperiod, 
                signalperiod=signalperiod
            )
            
            self.indicators[f'MACD_{price_type}'] = {
                'macd': macd,
                'signal': macdsignal,
                'histogram': macdhist
            }
            
            return self.indicators[f'MACD_{price_type}']
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            raise
    
    def calculate_bollinger_bands(
        self, 
        price_type: str = 'close', 
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        :param price_type: Price type to use (open/high/low/close)
        :param timeperiod: Number of periods
        :param nbdevup: Number of standard deviations for upper band
        :param nbdevdn: Number of standard deviations for lower band
        :return: Dictionary with Bollinger Bands
        """
        try:
            price = self.data[price_type].values
            upperband, middleband, lowerband = talib.BBANDS(
                price, 
                timeperiod=timeperiod, 
                nbdevup=nbdevup, 
                nbdevdn=nbdevdn,
                matype=0  # Simple Moving Average
            )
            
            self.indicators[f'BBANDS_{price_type}'] = {
                'upper': upperband,
                'middle': middleband,
                'lower': lowerband
            }
            
            return self.indicators[f'BBANDS_{price_type}']
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            raise
    
    def identify_candlestick_patterns(self) -> Dict[str, np.ndarray]:
        """
        Identify multiple candlestick patterns.
        
        :return: Dictionary of candlestick pattern indicators
        """
        try:
            patterns = {
                'HAMMER': talib.CDLHAMMER(
                    self.data['open'].values, 
                    self.data['high'].values, 
                    self.data['low'].values, 
                    self.data['close'].values
                ),
                'ENGULFING': talib.CDLENGULFING(
                    self.data['open'].values, 
                    self.data['high'].values, 
                    self.data['low'].values, 
                    self.data['close'].values
                ),
                'DOJI': talib.CDLDOJI(
                    self.data['open'].values, 
                    self.data['high'].values, 
                    self.data['low'].values, 
                    self.data['close'].values
                )
            }
            
            self.indicators['CANDLESTICK_PATTERNS'] = patterns
            return patterns
        except Exception as e:
            logger.error(f"Candlestick pattern identification error: {e}")
            raise
    
    def calculate_volatility(
        self, 
        method: str = 'ATR', 
        timeperiod: int = 14
    ) -> np.ndarray:
        """
        Calculate volatility indicators.
        
        :param method: Volatility calculation method (ATR or NATR)
        :param timeperiod: Number of periods
        :return: Volatility values
        """
        try:
            if method.upper() == 'ATR':
                volatility = talib.ATR(
                    self.data['high'].values, 
                    self.data['low'].values, 
                    self.data['close'].values, 
                    timeperiod=timeperiod
                )
                indicator_name = f'ATR_{timeperiod}'
            elif method.upper() == 'NATR':
                volatility = talib.NATR(
                    self.data['high'].values, 
                    self.data['low'].values, 
                    self.data['close'].values, 
                    timeperiod=timeperiod
                )
                indicator_name = f'NATR_{timeperiod}'
            else:
                raise ValueError("Supported volatility methods: ATR, NATR")
            
            self.indicators[indicator_name] = volatility
            return volatility
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            raise
    
    def get_market_structure(
        self, 
        lookback: int = 14
    ) -> Dict[str, List[float]]:
        """
        Identify key market structure points.
        
        :param lookback: Number of periods to look back
        :return: Dictionary of market structure points
        """
        try:
            highs = self.data['high'].values
            lows = self.data['low'].values
            
            # Identify swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(lookback, len(highs) - lookback):
                # Swing High
                if all(highs[i] > highs[j] for j in range(i-lookback, i)) and \
                   all(highs[i] > highs[j] for j in range(i+1, i+lookback+1)):
                    swing_highs.append(highs[i])
                
                # Swing Low
                if all(lows[i] < lows[j] for j in range(i-lookback, i)) and \
                   all(lows[i] < lows[j] for j in range(i+1, i+lookback+1)):
                    swing_lows.append(lows[i])
            
            market_structure = {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'last_swing_high': max(swing_highs) if swing_highs else None,
                'last_swing_low': min(swing_lows) if swing_lows else None
            }
            
            self.indicators['MARKET_STRUCTURE'] = market_structure
            return market_structure
        except Exception as e:
            logger.error(f"Market structure analysis error: {e}")
            raise
    
    def generate_trade_signals(
        self, 
        indicator: str = 'RSI',
        buy_threshold: float = 30.0,
        sell_threshold: float = 70.0
    ) -> Dict[str, List[int]]:
        """
        Generate trading signals based on specified indicator.
        
        :param indicator: Indicator to use for signals
        :param buy_threshold: Buy signal threshold
        :param sell_threshold: Sell signal threshold
        :return: Dictionary of trade signals
        """
        try:
            # Ensure indicator is calculated
            if indicator == 'RSI':
                indicator_values = self.calculate_rsi()
            elif indicator == 'MACD':
                indicator_values = self.calculate_macd()['macd']
            else:
                raise ValueError("Supported indicators for signals: RSI, MACD")
            
            # Generate signals
            buy_signals = [1 if val <= buy_threshold else 0 for val in indicator_values]
            sell_signals = [1 if val >= sell_threshold else 0 for val in indicator_values]
            
            signals = {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals
            }
            
            self.indicators['TRADE_SIGNALS'] = signals
            return signals
        except Exception as e:
            logger.error(f"Trade signal generation error: {e}")
            raise
    
    def get_all_indicators(self) -> Dict[str, Any]:
        """
        Retrieve all calculated indicators.
        
        :return: Dictionary of all indicators
        """
        return self.indicators
    
    @classmethod
    def get_supported_indicators(cls) -> List[str]:
        """
        Get list of supported indicators.
        
        :return: List of supported indicator names
        """
        return cls.SUPPORTED_INDICATORS

# Example usage
if __name__ == "__main__":
    # Create sample DataFrame
    import numpy as np
    
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'open': np.random.random(100) + 50,
        'high': np.random.random(100) + 51,
        'low': np.random.random(100) + 49,
        'close': np.random.random(100) + 50,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Initialize Technical Indicators
    tech_indicators = TechnicalIndicators(sample_data)
    
    # Calculate various indicators
    sma = tech_indicators.calculate_sma()
    rsi = tech_indicators.calculate_rsi()
    macd = tech_indicators.calculate_macd()
    bbands = tech_indicators.calculate_bollinger_bands()
    
    # Identify candlestick patterns
    patterns = tech_indicators.identify_candlestick_patterns()
    
    # Get market structure
    market_structure = tech_indicators.get_market_structure()
    
    # Generate trade signals
    trade_signals = tech_indicators.generate_trade_signals()