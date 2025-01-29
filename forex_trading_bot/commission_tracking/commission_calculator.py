import os
import pandas as pd
import numpy as np
from decimal import Decimal, getcontext
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import sqlite3
from loguru import logger

class CommissionCalculator:
    """
    Advanced commission calculation system for forex trading.
    Supports multiple commission structures and detailed tracking.
    """
    
    def __init__(
        self, 
        base_commission_rate: float = 0.0005,  # 0.05% base rate
        min_commission: float = 5.0,  # Minimum commission per trade
        max_commission: float = 50.0,  # Maximum commission per trade
        db_path: Optional[str] = None
    ):
        """
        Initialize commission calculator.
        
        :param base_commission_rate: Base commission rate
        :param min_commission: Minimum commission amount
        :param max_commission: Maximum commission amount
        :param db_path: Path to SQLite database for commission tracking
        """
        # Set precise decimal calculations
        getcontext().prec = 10
        
        # Commission parameters
        self.base_commission_rate = Decimal(str(base_commission_rate))
        self.min_commission = Decimal(str(min_commission))
        self.max_commission = Decimal(str(max_commission))
        
        # Database configuration
        self.db_path = db_path or os.path.join(
            os.getcwd(), 'commission_tracking', 'commissions.db'
        )
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Logging setup
        logger.add(
            "logs/commission_calculator.log", 
            rotation="10 MB"
        )
    
    def _initialize_database(self):
        """
        Create SQLite database and necessary tables.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create commission tracking table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS commissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        trade_type TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        volume REAL NOT NULL,
                        commission_amount REAL NOT NULL,
                        commission_rate REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create summary table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS commission_summary (
                        period TEXT NOT NULL,
                        total_trades INTEGER NOT NULL,
                        total_commission REAL NOT NULL,
                        avg_commission_per_trade REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Commission tracking database initialized")
        
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def calculate_commission(
        self, 
        symbol: str, 
        trade_type: str, 
        entry_price: float, 
        exit_price: float, 
        volume: float,
        custom_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate commission for a specific trade.
        
        :param symbol: Trading symbol
        :param trade_type: Type of trade (buy/sell)
        :param entry_price: Trade entry price
        :param exit_price: Trade exit price
        :param volume: Trading volume
        :param custom_rate: Optional custom commission rate
        :return: Commission calculation details
        """
        try:
            # Convert to Decimal for precise calculations
            entry_price_dec = Decimal(str(entry_price))
            exit_price_dec = Decimal(str(exit_price))
            volume_dec = Decimal(str(volume))
            
            # Determine commission rate
            commission_rate = (
                Decimal(str(custom_rate)) 
                if custom_rate is not None 
                else self.base_commission_rate
            )
            
            # Calculate trade value
            trade_value = volume_dec * abs(exit_price_dec - entry_price_dec)
            
            # Calculate commission
            commission_amount = trade_value * commission_rate
            
            # Apply min/max commission constraints
            commission_amount = max(
                self.min_commission, 
                min(commission_amount, self.max_commission)
            )
            
            # Prepare commission details
            commission_details = {
                'symbol': symbol,
                'trade_type': trade_type.upper(),
                'entry_price': float(entry_price_dec),
                'exit_price': float(exit_price_dec),
                'volume': float(volume_dec),
                'commission_amount': float(commission_amount),
                'commission_rate': float(commission_rate),
                'timestamp': datetime.now()
            }
            
            # Record commission
            self._record_commission(commission_details)
            
            logger.info(f"Commission calculated: {commission_details}")
            return commission_details
        
        except Exception as e:
            logger.error(f"Commission calculation error: {e}")
            raise
    
    def _record_commission(
        self, 
        commission_details: Dict[str, Any]
    ):
        """
        Record commission in the database.
        
        :param commission_details: Commission calculation details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert commission record
                cursor.execute('''
                    INSERT INTO commissions (
                        symbol, trade_type, entry_price, 
                        exit_price, volume, commission_amount, 
                        commission_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    commission_details['symbol'],
                    commission_details['trade_type'],
                    commission_details['entry_price'],
                    commission_details['exit_price'],
                    commission_details['volume'],
                    commission_details['commission_amount'],
                    commission_details['commission_rate']
                ))
                
                conn.commit()
        
        except sqlite3.Error as e:
            logger.error(f"Commission recording error: {e}")
    
    def get_commission_summary(
        self, 
        period: str = 'all_time',
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate commission summary report.
        
        :param period: Summary period (all_time/daily/weekly/monthly)
        :param symbol: Optional specific symbol filter
        :return: Commission summary details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare base query
                query = "SELECT * FROM commissions"
                params = []
                
                # Apply period filter
                if period != 'all_time':
                    time_filter = {
                        'daily': timedelta(days=1),
                        'weekly': timedelta(weeks=1),
                        'monthly': timedelta(days=30)
                    }
                    
                    query += " WHERE timestamp >= ?"
                    params.append(
                        (datetime.now() - time_filter[period]).strftime('%Y-%m-%d %H:%M:%S')
                    )
                
                # Apply symbol filter
                if symbol:
                    query += " AND symbol = ?" if params else " WHERE symbol = ?"
                    params.append(symbol)
                
                # Read data
                df = pd.read_sql_query(query, conn, params=params)
                
                # Generate summary
                summary = {
                    'period': period,
                    'symbol': symbol or 'All Symbols',
                    'total_trades': len(df),
                    'total_commission': df['commission_amount'].sum(),
                    'avg_commission_per_trade': df['commission_amount'].mean() if len(df) > 0 else 0,
                    'commission_by_trade_type': df.groupby('trade_type')['commission_amount'].sum().to_dict(),
                    'commission_by_symbol': df.groupby('symbol')['commission_amount'].sum().to_dict()
                }
                
                # Optional: Store summary in database
                self._store_commission_summary(summary)
                
                return summary
        
        except Exception as e:
            logger.error(f"Commission summary generation error: {e}")
            raise
    
    def _store_commission_summary(
        self, 
        summary: Dict[str, Any]
    ):
        """
        Store commission summary in database.
        
        :param summary: Commission summary details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert summary record
                cursor.execute('''
                    INSERT INTO commission_summary (
                        period, total_trades, 
                        total_commission, avg_commission_per_trade
                    ) VALUES (?, ?, ?, ?)
                ''', (
                    summary['period'],
                    summary['total_trades'],
                    summary['total_commission'],
                    summary['avg_commission_per_trade']
                ))
                
                conn.commit()
        
        except sqlite3.Error as e:
            logger.error(f"Commission summary storage error: {e}")
    
    def export_commission_data(
        self, 
        export_format: str = 'csv',
        output_path: Optional[str] = None,
        period: str = 'all_time',
        symbol: Optional[str] = None
    ) -> str:
        """
        Export commission data to specified format.
        
        :param export_format: Export file format (csv/excel)
        :param output_path: Custom output path
        :param period: Data period to export
        :param symbol: Optional specific symbol filter
        :return: Path to exported file
        """
        try:
            # Fetch commission data
            with sqlite3.connect(self.db_path) as conn:
                # Prepare base query
                query = "SELECT * FROM commissions"
                params = []
                
                # Apply period filter
                if period != 'all_time':
                    time_filter = {
                        'daily': timedelta(days=1),
                        'weekly': timedelta(weeks=1),
                        'monthly': timedelta(days=30)
                    }
                    
                    query += " WHERE timestamp >= ?"
                    params.append(
                        (datetime.now() - time_filter[period]).strftime('%Y-%m-%d %H:%M:%S')
                    )
                
                # Apply symbol filter
                if symbol:
                    query += " AND symbol = ?" if params else " WHERE symbol = ?"
                    params.append(symbol)
                
                # Read data
                df = pd.read_sql_query(query, conn, params=params)
            
            # Determine output path
            if not output_path:
                export_dir = os.path.join(os.getcwd(), 'commission_exports')
                os.makedirs(export_dir, exist_ok=True)
                output_path = os.path.join(
                    export_dir, 
                    f'commission_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                )
            
            # Export based on format
            if export_format.lower() == 'csv':
                full_path = f"{output_path}.csv"
                df.to_csv(full_path, index=False)
            elif export_format.lower() == 'excel':
                full_path = f"{output_path}.xlsx"
                df.to_excel(full_path, index=False)
            else:
                raise ValueError("Supported formats: csv, excel")
            
            logger.info(f"Commission data exported to {full_path}")
            return full_path
        
        except Exception as e:
            logger.error(f"Commission data export error: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize Commission Calculator
    commission_calc = CommissionCalculator()
    
    # Calculate commission for a trade
    trade_details = commission_calc.calculate_commission(
        symbol='NZDUSD',
        trade_type='buy',
        entry_price=0.6500,
        exit_price=0.6600,
        volume=0.1
    )
    print("Trade Commission:", trade_details)
    
    # Get commission summary
    summary = commission_calc.get_commission_summary(period='daily')
    print("Commission Summary:", summary)
    
    # Export commission data
    export_path = commission_calc.export_commission_data(
        export_format='csv',
        period='weekly'
    )
    print("Exported to:", export_path)