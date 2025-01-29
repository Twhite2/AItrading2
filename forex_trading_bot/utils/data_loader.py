import os
import json
import pandas as pd
import numpy as np
import requests
import websocket
import threading
import queue
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger

class DataLoader:
    """
    Comprehensive data loader for forex trading with support for 
    historical and real-time data fetching.
    """
    
    SUPPORTED_PAIRS = [
        'NZDUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 
        'USDCAD', 'USDCHF', 'USDJPY'
    ]
    
    TIMEFRAMES = {
        '1min': 60,
        '5min': 300,
        '15min': 900,
        'hourly': 3600,
        'daily': 86400,
        'weekly': 604800
    }
    
    def __init__(
        self, 
        symbol: str = 'NZDUSD', 
        data_sources: List[str] = ['deriv'],
        timeframe: str = 'hourly'
    ):
        """
        Initialize the DataLoader with specific configurations.
        
        :param symbol: Currency pair to fetch data for
        :param data_sources: List of data sources to use
        :param timeframe: Default timeframe for data retrieval
        """
        # Load environment variables
        load_dotenv()
        
        # Validate inputs
        if symbol not in self.SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported symbol. Supported pairs: {self.SUPPORTED_PAIRS}")
        
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe. Supported: {list(self.TIMEFRAMES.keys())}")
        
        # Configuration
        self.symbol = symbol
        self.data_sources = data_sources
        self.timeframe = timeframe
        
        # Data storage
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.realtime_data_queue = queue.Queue()
        
        # Websocket connection
        self._websocket_thread = None
        self._websocket_stop_event = threading.Event()
        
        # Logging setup
        logger.add(f"logs/data_loader_{symbol}.log", rotation="10 MB")
        
        # API configurations
        self._load_api_configs()
    
    def _load_api_configs(self):
        """
        Load API configurations from environment variables.
        """
        try:
            # Deriv API configuration
            self.deriv_app_id = os.getenv('DERIV_APP_ID')
            self.deriv_token = os.getenv('DERIV_API_TOKEN')
            
            # Validate API credentials
            if not self.deriv_app_id or not self.deriv_token:
                logger.warning("Deriv API credentials not fully configured")
        
        except Exception as e:
            logger.error(f"Error loading API configurations: {e}")
    
    def fetch_historical_data(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch historical market data from specified sources.
        
        :param start_date: Start date for historical data
        :param end_date: End date for historical data
        :param limit: Maximum number of data points to retrieve
        :return: DataFrame with historical market data
        """
        # Set default date range if not provided
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        historical_data = []
        
        for source in self.data_sources:
            try:
                if source == 'deriv':
                    source_data = self._fetch_deriv_historical_data(
                        start_date, end_date, limit
                    )
                    historical_data.append(source_data)
                # Add more data sources as needed
            
            except Exception as e:
                logger.error(f"Error fetching data from {source}: {e}")
        
        # Combine data from multiple sources
        if historical_data:
            combined_data = pd.concat(historical_data, ignore_index=True)
            combined_data = combined_data.drop_duplicates().sort_values('timestamp')
            
            # Store and return the data
            self.historical_data[self.timeframe] = combined_data
            return combined_data
        
        raise ValueError("No historical data could be retrieved")
    
    def _fetch_deriv_historical_data(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: int
    ) -> pd.DataFrame:
        """
        Fetch historical data specifically from Deriv API.
        
        :param start_date: Start date for data retrieval
        :param end_date: End date for data retrieval
        :param limit: Maximum number of data points
        :return: DataFrame with Deriv historical data
        """
        # Construct Deriv API request
        params = {
            'symbol': self.symbol.replace('USD', ''),
            'granularity': self.TIMEFRAMES[self.timeframe],
            'start': int(start_date.timestamp()),
            'end': int(end_date.timestamp()),
            'limit': limit
        }
        
        # Make API request
        try:
            response = requests.get(
                'https://api.deriv.com/api-explorer/candles',
                params=params,
                headers={
                    'Authorization': f'Bearer {self.deriv_token}',
                    'X-App-Id': self.deriv_app_id
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Transform Deriv data to DataFrame
            df = pd.DataFrame(data['candles'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except requests.RequestException as e:
            logger.error(f"Deriv API request failed: {e}")
            raise
    
    def start_realtime_data_stream(self):
        """
        Start a real-time data stream using websocket.
        """
        try:
            # Prepare websocket connection
            websocket.enableTrace(False)
            ws = websocket.WebSocketApp(
                'wss://ws.binaryws.com/websockets/v3',
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Authentication and subscription
            def on_open(ws):
                # Authenticate and subscribe to symbol
                auth_data = {
                    'authorize': self.deriv_token,
                    'app_id': self.deriv_app_id
                }
                ws.send(json.dumps(auth_data))
                
                subscribe_data = {
                    'ticks': self.symbol,
                    'subscribe': 1
                }
                ws.send(json.dumps(subscribe_data))
            
            ws.on_open = on_open
            
            # Run websocket in a separate thread
            self._websocket_thread = threading.Thread(
                target=ws.run_forever, 
                daemon=True
            )
            self._websocket_thread.start()
            
            logger.info(f"Real-time data stream started for {self.symbol}")
        
        except Exception as e:
            logger.error(f"Failed to start real-time data stream: {e}")
    
    def _on_message(self, ws, message):
        """
        Handle incoming websocket messages.
        
        :param ws: Websocket connection
        :param message: Received message
        """
        try:
            data = json.loads(message)
            
            # Process tick data
            if 'tick' in data:
                tick_data = {
                    'timestamp': datetime.now(),
                    'price': data['tick']['quote']
                }
                self.realtime_data_queue.put(tick_data)
        
        except Exception as e:
            logger.error(f"Error processing websocket message: {e}")
    
    def _on_error(self, ws, error):
        """
        Handle websocket errors.
        
        :param ws: Websocket connection
        :param error: Error details
        """
        logger.error(f"Websocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Handle websocket connection closure.
        
        :param ws: Websocket connection
        :param close_status_code: Closure status code
        :param close_msg: Closure message
        """
        logger.info("Websocket connection closed")
    
    def stop_realtime_data_stream(self):
        """
        Stop the real-time data stream.
        """
        try:
            self._websocket_stop_event.set()
            if self._websocket_thread:
                self._websocket_thread.join(timeout=5)
            
            logger.info("Real-time data stream stopped")
        
        except Exception as e:
            logger.error(f"Error stopping data stream: {e}")
    
    def get_realtime_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Retrieve the latest real-time data point.
        
        :param timeout: Maximum time to wait for data
        :return: Latest data point or None
        """
        try:
            return self.realtime_data_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("No real-time data available")
            return None
    
    @classmethod
    def get_supported_pairs(cls) -> List[str]:
        """
        Get list of supported currency pairs.
        
        :return: List of supported pairs
        """
        return cls.SUPPORTED_PAIRS
    
    @classmethod
    def get_supported_timeframes(cls) -> Dict[str, int]:
        """
        Get supported timeframes with their corresponding seconds.
        
        :return: Dictionary of timeframes
        """
        return cls.TIMEFRAMES

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    data_loader = DataLoader(symbol='NZDUSD', timeframe='hourly')
    
    # Fetch historical data
    historical_data = data_loader.fetch_historical_data()
    print("Historical Data Sample:")
    print(historical_data.head())
    
    # Start real-time data stream
    data_loader.start_realtime_data_stream()
    
    try:
        # Simulate real-time data retrieval
        for _ in range(5):
            realtime_data = data_loader.get_realtime_data()
            if realtime_data:
                print("Real-time Data:", realtime_data)
    
    finally:
        # Stop the data stream
        data_loader.stop_realtime_data_stream()