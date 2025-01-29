import os
import json
import requests
import websocket
import threading
import queue
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
from datetime import datetime

class TradeExecutor:
    """
    Comprehensive trade execution system for forex trading.
    Handles order placement, management, and tracking via Deriv API.
    """
    
    SUPPORTED_PAIRS = [
        'NZDUSD', 'EURUSD', 'GBPUSD', 'AUDUSD', 
        'USDCAD', 'USDCHF', 'USDJPY'
    ]
    
    ORDER_TYPES = {
        'market': 'MARKET',
        'limit': 'LIMIT',
        'stop': 'STOP'
    }
    
    def __init__(
        self, 
        symbol: str = 'NZDUSD', 
        initial_balance: float = 10000.0
    ):
        """
        Initialize TradeExecutor with specific trading parameters.
        
        :param symbol: Currency pair to trade
        :param initial_balance: Initial trading account balance
        """
        # Load environment variables
        load_dotenv()
        
        # Validate inputs
        if symbol not in self.SUPPORTED_PAIRS:
            raise ValueError(f"Unsupported symbol. Supported pairs: {self.SUPPORTED_PAIRS}")
        
        # Configuration
        self.symbol = symbol
        self.initial_balance = initial_balance
        
        # Trading state
        self.current_balance = initial_balance
        self.open_positions: List[Dict[str, Any]] = []
        self.order_history: List[Dict[str, Any]] = []
        
        # Deriv API configuration
        self._load_api_configs()
        
        # Websocket connection
        self._websocket = None
        self._websocket_thread = None
        self._websocket_stop_event = threading.Event()
        self._trade_queue = queue.Queue()
        
        # Logging setup
        logger.add(f"logs/trade_executor_{symbol}.log", rotation="10 MB")
        
        # Performance tracking
        self.trade_log = pd.DataFrame(columns=[
            'timestamp', 'type', 'symbol', 'entry_price', 'exit_price', 
            'volume', 'profit_loss', 'commission'
        ])
    
    def _load_api_configs(self):
        """
        Load Deriv API configurations from environment variables.
        """
        try:
            self.deriv_app_id = os.getenv('DERIV_APP_ID')
            self.deriv_token = os.getenv('DERIV_API_TOKEN')
            
            if not self.deriv_app_id or not self.deriv_token:
                raise ValueError("Deriv API credentials are missing")
        
        except Exception as e:
            logger.error(f"API configuration error: {e}")
            raise
    
    def connect_websocket(self):
        """
        Establish a WebSocket connection to Deriv API for real-time trading.
        """
        try:
            # Disable WebSocket trace for production
            websocket.enableTrace(False)
            
            # Create WebSocket application
            self._websocket = websocket.WebSocketApp(
                'wss://ws.binaryws.com/websockets/v3',
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Authentication and connection handler
            def on_open(ws):
                # Authenticate
                auth_data = {
                    'authorize': self.deriv_token,
                    'app_id': self.deriv_app_id
                }
                ws.send(json.dumps(auth_data))
                
                # Subscribe to relevant markets
                subscribe_data = {
                    'ticks': self.symbol,
                    'subscribe': 1
                }
                ws.send(json.dumps(subscribe_data))
                
                logger.info(f"WebSocket connected for {self.symbol}")
            
            self._websocket.on_open = on_open
            
            # Run WebSocket in a separate thread
            self._websocket_thread = threading.Thread(
                target=self._websocket.run_forever, 
                daemon=True
            )
            self._websocket_thread.start()
        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    def place_order(
        self, 
        order_type: str, 
        side: str, 
        volume: float, 
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place a trading order via Deriv API.
        
        :param order_type: Type of order (market/limit/stop)
        :param side: 'buy' or 'sell'
        :param volume: Trading volume
        :param price: Limit/stop order price
        :param stop_loss: Stop loss price
        :param take_profit: Take profit price
        :return: Order details
        """
        try:
            # Validate inputs
            if order_type not in self.ORDER_TYPES:
                raise ValueError(f"Invalid order type. Supported: {list(self.ORDER_TYPES.keys())}")
            
            if side not in ['buy', 'sell']:
                raise ValueError("Side must be 'buy' or 'sell'")
            
            # Prepare order payload
            order_payload = {
                'symbol': self.symbol,
                'type': self.ORDER_TYPES[order_type],
                'side': side.upper(),
                'volume': volume,
            }
            
            # Add optional parameters
            if price:
                order_payload['price'] = price
            
            if stop_loss:
                order_payload['stop_loss'] = stop_loss
            
            if take_profit:
                order_payload['take_profit'] = take_profit
            
            # Send order via WebSocket
            if not self._websocket:
                self.connect_websocket()
            
            self._websocket.send(json.dumps({
                'trading_servers': 1,
                'payload': order_payload
            }))
            
            # Wait for order confirmation
            order_response = self._trade_queue.get(timeout=10)
            
            # Record order in history
            order_details = {
                'timestamp': datetime.now(),
                'symbol': self.symbol,
                'type': order_type,
                'side': side,
                'volume': volume,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': order_response.get('status', 'PENDING')
            }
            
            self.order_history.append(order_details)
            
            logger.info(f"Order placed: {order_details}")
            return order_details
        
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise
    
    def modify_order(
        self, 
        order_id: str, 
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Modify an existing order.
        
        :param order_id: ID of the order to modify
        :param price: New order price
        :param stop_loss: New stop loss price
        :param take_profit: New take profit price
        :return: Modified order details
        """
        try:
            modify_payload = {
                'order_id': order_id
            }
            
            # Add modifiable parameters
            if price:
                modify_payload['price'] = price
            if stop_loss:
                modify_payload['stop_loss'] = stop_loss
            if take_profit:
                modify_payload['take_profit'] = take_profit
            
            # Send modification request
            self._websocket.send(json.dumps({
                'trading_servers': 1,
                'payload': modify_payload
            }))
            
            # Wait for modification confirmation
            modify_response = self._trade_queue.get(timeout=10)
            
            logger.info(f"Order {order_id} modified: {modify_payload}")
            return modify_response
        
        except Exception as e:
            logger.error(f"Order modification failed: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        :param order_id: ID of the order to cancel
        :return: Cancellation details
        """
        try:
            # Send cancellation request
            self._websocket.send(json.dumps({
                'trading_servers': 1,
                'payload': {
                    'order_id': order_id,
                    'action': 'CANCEL'
                }
            }))
            
            # Wait for cancellation confirmation
            cancel_response = self._trade_queue.get(timeout=10)
            
            logger.info(f"Order {order_id} cancelled")
            return cancel_response
        
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            raise
    
    def close_position(
        self, 
        position_id: str, 
        volume: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Close an open trading position.
        
        :param position_id: ID of the position to close
        :param volume: Volume to close (defaults to full position)
        :return: Position closure details
        """
        try:
            close_payload = {
                'position_id': position_id
            }
            
            if volume:
                close_payload['volume'] = volume
            
            # Send close position request
            self._websocket.send(json.dumps({
                'trading_servers': 1,
                'payload': close_payload
            }))
            
            # Wait for closure confirmation
            close_response = self._trade_queue.get(timeout=10)
            
            logger.info(f"Position {position_id} closed")
            return close_response
        
        except Exception as e:
            logger.error(f"Position closure failed: {e}")
            raise
    
    def _on_message(self, ws, message):
        """
        Handle incoming WebSocket messages.
        
        :param ws: WebSocket connection
        :param message: Received message
        """
        try:
            data = json.loads(message)
            
            # Process trade-related messages
            if 'trading_servers' in data:
                self._trade_queue.put(data)
            
            # Update trade log
            self._update_trade_log(data)
        
        except Exception as e:
            logger.error(f"WebSocket message processing error: {e}")
    
    def _update_trade_log(self, trade_data: Dict[str, Any]):
        """
        Update trade log with latest trade information.
        
        :param trade_data: Trade-related data
        """
        try:
            # Extract relevant trade details
            if 'trade' in trade_data:
                trade = trade_data['trade']
                log_entry = {
                    'timestamp': datetime.now(),
                    'type': trade.get('type', 'UNKNOWN'),
                    'symbol': trade.get('symbol', self.symbol),
                    'entry_price': trade.get('entry_price'),
                    'exit_price': trade.get('exit_price'),
                    'volume': trade.get('volume'),
                    'profit_loss': trade.get('profit_loss'),
                    'commission': trade.get('commission')
                }
                
                # Append to trade log DataFrame
                self.trade_log = self.trade_log.append(log_entry, ignore_index=True)
        
        except Exception as e:
            logger.error(f"Trade log update failed: {e}")
    
    def _on_error(self, ws, error):
        """
        Handle WebSocket errors.
        
        :param ws: WebSocket connection
        :param error: Error details
        """
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Handle WebSocket connection closure.
        
        :param ws: WebSocket connection
        :param close_status_code: Closure status code
        :param close_msg: Closure message
        """
        logger.info("WebSocket connection closed")
    
    def disconnect(self):
        """
        Disconnect WebSocket and clean up resources.
        """
        try:
            if self._websocket:
                self._websocket.close()
            
            if self._websocket_thread:
                self._websocket_thread.join(timeout=5)
            
            logger.info("Trade executor disconnected")
        
        except Exception as e:
            logger.error(f"Disconnection error: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a performance summary of trades.
        
        :return: Dictionary with performance metrics
        """
        try:
            # Calculate performance metrics
            total_trades = len(self.trade_log)
            winning_trades = self.trade_log[self.trade_log['profit_loss'] > 0]
            losing_trades = self.trade_log[self.trade_log['profit_loss'] < 0]
            
            performance_summary = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / total_trades if total_trades > 0 else 0,
                'total_profit': self.trade_log['profit_loss'].sum(),
                'average_profit_per_trade': self.trade_log['profit_loss'].mean(),
                'total_commission': self.trade_log['commission'].sum()
            }
            
            return performance_summary
        
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Initialize trade executor
    trade_executor = TradeExecutor(symbol='NZDUSD')
    
    try:
        # Connect to WebSocket
        trade_executor.connect_websocket()
        
        # Example trade placement
        order = trade_executor.place_order(
            order_type='market', 
            side='buy', 
            volume=0.1,
            stop_loss=0.6500,
            take_profit=0.6700
        )
        
        # Get performance summary
        performance = trade_executor.get_performance_summary()
        print("Performance Summary:", performance)
    
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
    
    finally:
        # Disconnect and clean up
        trade_executor.disconnect()