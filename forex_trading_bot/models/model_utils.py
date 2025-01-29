import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    pipeline
)
from loguru import logger
from dotenv import load_dotenv

class ModelUtils:
    """
    Utility class for loading, preprocessing, 
    and generating insights using Flan-T5 model for trading.
    """
    
    def __init__(
        self, 
        model_name: str = 'Isotonic/flan-t5-base-trading_candles',
        device: Optional[str] = None
    ):
        """
        Initialize the model utilities.
        
        :param model_name: Hugging Face model name
        :param device: Compute device (cuda/cpu)
        """
        # Load environment variables
        load_dotenv()
        
        # Logging setup
        logger.add("logs/model_utils.log", rotation="10 MB")
        
        # Device configuration
        self.device = device or self._select_device()
        
        # Model and tokenizer
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Load model and tokenizer
        self._load_model()
        
        # Create generation pipeline
        self.generation_pipeline = pipeline(
            'text2text-generation', 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=self._get_device_index()
        )
    
    def _select_device(self) -> str:
        """
        Select optimal computation device.
        
        :return: Selected device (cuda or cpu)
        """
        try:
            if torch.cuda.is_available():
                logger.info("CUDA device available. Using GPU.")
                return 'cuda'
            else:
                logger.info("No CUDA device. Falling back to CPU.")
                return 'cpu'
        except Exception as e:
            logger.error(f"Device selection error: {e}")
            return 'cpu'
    
    def _get_device_index(self) -> int:
        """
        Get device index for pipeline.
        
        :return: Device index
        """
        return 0 if self.device == 'cuda' else -1
    
    def _load_model(self):
        """
        Load Flan-T5 model and tokenizer.
        """
        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
            
            logger.info(f"Model {self.model_name} loaded successfully")
        
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def preprocess_market_data(
        self, 
        market_data: pd.DataFrame
    ) -> str:
        """
        Preprocess market data for model input.
        
        :param market_data: DataFrame with market data
        :return: Formatted string input for model
        """
        try:
            # Select relevant columns
            columns = ['open', 'high', 'low', 'close', 'volume']
            data = market_data[columns].tail(20)  # Last 20 candles
            
            # Format data as a structured prompt
            data_str = "Candle Data:\n"
            for _, row in data.iterrows():
                data_str += (
                    f"O:{row['open']:.4f} "
                    f"H:{row['high']:.4f} "
                    f"L:{row['low']:.4f} "
                    f"C:{row['close']:.4f} "
                    f"V:{row['volume']}\n"
                )
            
            return data_str
        
        except Exception as e:
            logger.error(f"Market data preprocessing error: {e}")
            raise
    
    def generate_market_insight(
        self, 
        market_data: pd.DataFrame,
        insight_type: str = 'trend_prediction'
    ) -> Dict[str, Any]:
        """
        Generate AI-powered market insights.
        
        :param market_data: DataFrame with market data
        :param insight_type: Type of insight to generate
        :return: Dictionary of market insights
        """
        try:
            # Preprocess market data
            input_data = self.preprocess_market_data(market_data)
            
            # Prepare prompt based on insight type
            prompts = {
                'trend_prediction': (
                    f"{input_data}\n"
                    "Predict the next market trend. "
                    "Provide a detailed analysis of potential "
                    "market direction with confidence level."
                ),
                'trading_signal': (
                    f"{input_data}\n"
                    "Generate a trading signal: Buy, Sell, or Hold. "
                    "Explain the reasoning behind the signal."
                ),
                'volatility_analysis': (
                    f"{input_data}\n"
                    "Analyze potential market volatility. "
                    "Predict price movement range and key support/resistance levels."
                )
            }
            
            # Select prompt
            prompt = prompts.get(
                insight_type, 
                prompts['trend_prediction']
            )
            
            # Generate insight
            model_output = self.generation_pipeline(
                prompt, 
                max_length=200, 
                num_return_sequences=1
            )[0]['generated_text']
            
            # Process and structure output
            insight = {
                'type': insight_type,
                'raw_output': model_output,
                'timestamp': pd.Timestamp.now()
            }
            
            # Additional parsing could be added here
            logger.info(f"Generated {insight_type} insight")
            return insight
        
        except Exception as e:
            logger.error(f"Market insight generation error: {e}")
            raise
    
    def evaluate_model_confidence(
        self, 
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate model's confidence in generated insights.
        
        :param market_data: DataFrame with market data
        :return: Dictionary of confidence metrics
        """
        try:
            # Prepare input data
            input_data = self.preprocess_market_data(market_data)
            
            # Generate multiple insights
            insights = [
                self.generate_market_insight(market_data, 'trend_prediction'),
                self.generate_market_insight(market_data, 'trading_signal'),
                self.generate_market_insight(market_data, 'volatility_analysis')
            ]
            
            # Basic confidence evaluation
            confidence_metrics = {
                'trend_consistency': self._calculate_trend_consistency(insights),
                'signal_confidence': self._calculate_signal_confidence(insights),
                'volatility_reliability': self._calculate_volatility_reliability(insights)
            }
            
            logger.info("Model confidence evaluated")
            return confidence_metrics
        
        except Exception as e:
            logger.error(f"Model confidence evaluation error: {e}")
            raise
    
    def _calculate_trend_consistency(
        self, 
        insights: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate consistency of trend predictions.
        
        :param insights: List of generated insights
        :return: Trend consistency score
        """
        try:
            # Basic trend keyword matching
            trend_keywords = {
                'bullish': ['buy', 'uptrend', 'positive', 'strong'],
                'bearish': ['sell', 'downtrend', 'negative', 'weak']
            }
            
            trend_scores = {'bullish': 0, 'bearish': 0}
            
            for insight in insights:
                text = insight['raw_output'].lower()
                
                for trend, keywords in trend_keywords.items():
                    trend_scores[trend] += sum(
                        1 for keyword in keywords if keyword in text
                    )
            
            # Normalize and calculate confidence
            total_score = sum(trend_scores.values())
            confidence = (
                trend_scores['bullish'] / total_score if total_score > 0 else 0.5
            )
            
            return confidence
        
        except Exception as e:
            logger.error(f"Trend consistency calculation error: {e}")
            return 0.5
    
    def _calculate_signal_confidence(
        self, 
        insights: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence in trading signals.
        
        :param insights: List of generated insights
        :return: Signal confidence score
        """
        try:
            # Signal keyword matching
            signal_keywords = {
                'buy': ['buy', 'long', 'enter', 'bullish'],
                'sell': ['sell', 'short', 'exit', 'bearish'],
                'hold': ['hold', 'neutral', 'wait']
            }
            
            signal_scores = {
                'buy': 0,
                'sell': 0,
                'hold': 0
            }
            
            for insight in insights:
                text = insight['raw_output'].lower()
                
                for signal, keywords in signal_keywords.items():
                    signal_scores[signal] += sum(
                        1 for keyword in keywords if keyword in text
                    )
            
            # Determine dominant signal
            total_score = sum(signal_scores.values())
            dominant_signal = max(signal_scores, key=signal_scores.get)
            
            # Calculate confidence
            confidence = (
                signal_scores[dominant_signal] / total_score 
                if total_score > 0 else 0.5
            )
            
            return confidence
        
        except Exception as e:
            logger.error(f"Signal confidence calculation error: {e}")
            return 0.5
    
    def _calculate_volatility_reliability(
        self, 
        insights: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate reliability of volatility predictions.
        
        :param insights: List of generated insights
        :return: Volatility reliability score
        """
        try:
            # Volatility keyword matching
            volatility_keywords = {
                'high': ['high', 'increase', 'volatile', 'extreme'],
                'low': ['low', 'stable', 'calm', 'steady']
            }
            
            volatility_scores = {
                'high': 0,
                'low': 0
            }
            
            for insight in insights:
                text = insight['raw_output'].lower()
                
                for vol_type, keywords in volatility_keywords.items():
                    volatility_scores[vol_type] += sum(
                        1 for keyword in keywords if keyword in text
                    )
            
            # Calculate reliability
            total_score = sum(volatility_scores.values())
            reliability = (
                volatility_scores['high'] / total_score 
                if total_score > 0 else 0.5
            )
            
            return reliability
        
        except Exception as e:
            logger.error(f"Volatility reliability calculation error: {e}")
            return 0.5

# Example usage
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
    
    # Initialize Model Utilities
    model_utils = ModelUtils()
    
    # Generate market insights
    trend_insight = model_utils.generate_market_insight(
        sample_data, 
        insight_type='trend_prediction'
    )
    print("Trend Insight:", trend_insight)
    
    # Evaluate model confidence
    confidence_metrics = model_utils.evaluate_model_confidence(sample_data)
    print("Confidence Metrics:", confidence_metrics)