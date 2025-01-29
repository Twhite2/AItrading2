import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from loguru import logger

class RiskManager:
    """
    Comprehensive risk management system for forex trading.
    
    Handles position sizing, stop loss, take profit, 
    and overall portfolio risk management.
    """
    
    def __init__(
        self, 
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        max_portfolio_risk: float = 0.05  # 5% max portfolio risk
    ):
        """
        Initialize risk management parameters.
        
        :param initial_balance: Starting trading account balance
        :param risk_per_trade: Percentage of account to risk per trade
        :param max_portfolio_risk: Maximum total portfolio risk
        """
        # Account parameters
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Risk parameters
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        
        # Trade tracking
        self.open_trades = []
        self.trade_history = []
        
        # Risk metrics
        self.risk_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0
        }
        
        # Logging setup
        logger.add("logs/risk_management.log", rotation="10 MB")
    
    def calculate_position_size(
        self, 
        entry_price: float, 
        stop_loss_price: float, 
        account_risk_percentage: Optional[float] = None
    ) -> float:
        """
        Calculate optimal position size based on account risk.
        
        :param entry_price: Trade entry price
        :param stop_loss_price: Stop loss price
        :param account_risk_percentage: Percentage of account to risk (default: self.risk_per_trade)
        :return: Optimal position size
        """
        try:
            # Determine risk percentage
            risk_percentage = account_risk_percentage or self.risk_per_trade
            
            # Calculate risk amount
            risk_amount = self.current_balance * risk_percentage
            
            # Calculate risk per unit
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            # Calculate position size
            if risk_per_unit == 0:
                logger.warning("Risk per unit is zero. Unable to calculate position size.")
                return 0.0
            
            position_size = risk_amount / risk_per_unit
            
            # Validate position size against portfolio risk
            total_risk = position_size * risk_per_unit
            if total_risk > self.current_balance * self.max_portfolio_risk:
                logger.warning("Position size exceeds maximum portfolio risk.")
                position_size = (self.current_balance * self.max_portfolio_risk) / risk_per_unit
            
            return round(position_size, 2)
        
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            raise
    
    def calculate_stop_loss(
        self, 
        entry_price: float, 
        technical_levels: Dict[str, float],
        risk_reward_ratio: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calculate dynamic stop loss based on technical levels.
        
        :param entry_price: Trade entry price
        :param technical_levels: Dictionary of technical support/resistance levels
        :param risk_reward_ratio: Desired risk-reward ratio
        :return: Tuple of (stop_loss_price, take_profit_price)
        """
        try:
            # Use technical levels for stop loss placement
            swing_low = technical_levels.get('swing_low')
            swing_high = technical_levels.get('swing_high')
            
            # Default to percentage-based stop loss
            default_stop_loss_percentage = 0.005  # 0.5%
            
            if swing_low and swing_high:
                # Use swing levels for more precise stop loss
                if entry_price > swing_high:  # Long trade
                    stop_loss_price = swing_low
                else:  # Short trade
                    stop_loss_price = swing_high
            else:
                # Percentage-based fallback
                if entry_price > technical_levels.get('resistance', entry_price):
                    # Long trade
                    stop_loss_price = entry_price * (1 - default_stop_loss_percentage)
                else:
                    # Short trade
                    stop_loss_price = entry_price * (1 + default_stop_loss_percentage)
            
            # Calculate take profit based on risk-reward ratio
            risk_amount = abs(entry_price - stop_loss_price)
            take_profit_price = (
                entry_price + risk_amount * risk_reward_ratio 
                if entry_price > stop_loss_price 
                else entry_price - risk_amount * risk_reward_ratio
            )
            
            return stop_loss_price, take_profit_price
        
        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            raise
    
    def assess_trade_risk(
        self, 
        trade_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a potential trade.
        
        :param trade_details: Dictionary containing trade parameters
        :return: Risk assessment results
        """
        try:
            # Extract trade parameters
            entry_price = trade_details.get('entry_price')
            stop_loss_price = trade_details.get('stop_loss')
            take_profit_price = trade_details.get('take_profit')
            technical_levels = trade_details.get('technical_levels', {})
            
            # Risk assessment components
            risk_assessment = {
                'entry_price': entry_price,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'risk_per_trade': self.risk_per_trade,
                'position_size': 0.0,
                'risk_reward_ratio': 0.0,
                'risk_status': 'INVALID'
            }
            
            # Validate trade parameters
            if not all([entry_price, stop_loss_price, take_profit_price]):
                logger.warning("Incomplete trade parameters for risk assessment")
                return risk_assessment
            
            # Calculate position size
            position_size = self.calculate_position_size(
                entry_price, 
                stop_loss_price
            )
            
            # Calculate risk-reward ratio
            risk_amount = abs(entry_price - stop_loss_price)
            reward_amount = abs(take_profit_price - entry_price)
            risk_reward_ratio = reward_amount / risk_amount
            
            # Update risk assessment
            risk_assessment.update({
                'position_size': position_size,
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'risk_status': 'ACCEPTABLE' if risk_reward_ratio >= 1.5 else 'RISKY'
            })
            
            return risk_assessment
        
        except Exception as e:
            logger.error(f"Trade risk assessment error: {e}")
            raise
    
    def update_trade_metrics(
        self, 
        trade_result: Dict[str, Any]
    ):
        """
        Update trade metrics and account balance.
        
        :param trade_result: Dictionary containing trade outcome
        """
        try:
            # Increment total trades
            self.risk_metrics['total_trades'] += 1
            
            # Update winning/losing trades
            profit_loss = trade_result.get('profit_loss', 0)
            if profit_loss > 0:
                self.risk_metrics['winning_trades'] += 1
            else:
                self.risk_metrics['losing_trades'] += 1
            
            # Update current balance
            self.current_balance += profit_loss
            
            # Update max drawdown
            drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
            self.risk_metrics['max_drawdown'] = max(
                self.risk_metrics['max_drawdown'], 
                drawdown
            )
            
            # Log trade metrics
            logger.info(f"Trade Metrics Updated: {self.risk_metrics}")
        
        except Exception as e:
            logger.error(f"Trade metrics update error: {e}")
    
    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio risk summary.
        
        :return: Dictionary with portfolio risk metrics
        """
        try:
            portfolio_risk_summary = {
                'current_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'total_trades': self.risk_metrics['total_trades'],
                'winning_trades': self.risk_metrics['winning_trades'],
                'losing_trades': self.risk_metrics['losing_trades'],
                'win_rate': (self.risk_metrics['winning_trades'] / 
                             self.risk_metrics['total_trades'] 
                             if self.risk_metrics['total_trades'] > 0 else 0),
                'max_drawdown': self.risk_metrics['max_drawdown'],
                'current_risk_per_trade': self.risk_per_trade,
                'max_portfolio_risk': self.max_portfolio_risk
            }
            
            return portfolio_risk_summary
        
        except Exception as e:
            logger.error(f"Portfolio risk summary generation error: {e}")
            raise
    
    def adjust_risk_parameters(
        self, 
        new_risk_per_trade: Optional[float] = None,
        new_max_portfolio_risk: Optional[float] = None
    ):
        """
        Dynamically adjust risk management parameters.
        
        :param new_risk_per_trade: New percentage to risk per trade
        :param new_max_portfolio_risk: New maximum portfolio risk
        """
        try:
            if new_risk_per_trade is not None:
                if 0 < new_risk_per_trade <= 0.1:  # 0-10% risk per trade
                    self.risk_per_trade = new_risk_per_trade
                    logger.info(f"Risk per trade updated to {new_risk_per_trade}")
                else:
                    logger.warning("Invalid risk per trade percentage")
            
            if new_max_portfolio_risk is not None:
                if 0 < new_max_portfolio_risk <= 0.2:  # 0-20% max portfolio risk
                    self.max_portfolio_risk = new_max_portfolio_risk
                    logger.info(f"Max portfolio risk updated to {new_max_portfolio_risk}")
                else:
                    logger.warning("Invalid max portfolio risk percentage")
        
        except Exception as e:
            logger.error(f"Risk parameter adjustment error: {e}")
    
# Example usage
if __name__ == "__main__":
    # Initialize Risk Manager
    risk_manager = RiskManager(
        initial_balance=10000.0,
        risk_per_trade=0.01,  # 1% risk per trade
        max_portfolio_risk=0.05  # 5% max portfolio risk
    )
    
    # Example trade scenario
    technical_levels = {
        'swing_low': 0.6500,
        'swing_high': 0.6700,
        'resistance': 0.6750
    }
    
    trade_details = {
        'entry_price': 0.6600,
        'stop_loss': 0.6500,
        'take_profit': 0.6800,
        'technical_levels': technical_levels
    }
    
    # Assess trade risk
    risk_assessment = risk_manager.assess_trade_risk(trade_details)
    print("Risk Assessment:", risk_assessment)
    
    # Calculate position size
    position_size = risk_manager.calculate_position_size(
        entry_price=0.6600, 
        stop_loss_price=0.6500
    )
    print("Position Size:", position_size)
    
    # Get portfolio risk summary
    portfolio_summary = risk_manager.get_portfolio_risk_summary()
    print("Portfolio Risk Summary:", portfolio_summary)