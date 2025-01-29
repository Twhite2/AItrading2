import os
import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from decimal import Decimal, getcontext

class CommissionTracker:
    """
    Comprehensive commission tracking system for forex trading.
    Manages commission calculations, storage, and reporting.
    """
    
    def __init__(
        self, 
        db_path: str = 'commission_tracking.db',
        base_commission_rate: float = 0.0005  # 5 pips or 0.05%
    ):
        """
        Initialize Commission Tracker.
        
        :param db_path: Path to SQLite database
        :param base_commission_rate: Base commission rate per trade
        """
        # Set up precise decimal calculations
        getcontext().prec = 10
        
        # Logging setup
        logger.add("logs/commission_tracker.log", rotation="10 MB")
        
        # Database configuration
        self.db_path = db_path
        self.base_commission_rate = Decimal(str(base_commission_rate))
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._create_database()
    
    def _create_database(self):
        """
        Create SQLite database and necessary tables.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create commissions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS commissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        trade_type TEXT NOT NULL,
                        volume REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL NOT NULL,
                        commission_amount REAL NOT NULL,
                        commission_currency TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create summary table for periodic reporting
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
            logger.error(f"Database creation error: {e}")
            raise
    
    def calculate_commission(
        self, 
        symbol: str, 
        trade_type: str, 
        volume: float, 
        entry_price: float, 
        exit_price: float
    ) -> Dict[str, Any]:
        """
        Calculate commission for a trade.
        
        :param symbol: Trading symbol
        :param trade_type: Type of trade (buy/sell)
        :param volume: Trade volume
        :param entry_price: Entry price
        :param exit_price: Exit price
        :return: Commission details
        """
        try:
            # Convert to Decimal for precise calculations
            volume_dec = Decimal(str(volume))
            entry_price_dec = Decimal(str(entry_price))
            exit_price_dec = Decimal(str(exit_price))
            
            # Calculate trade value
            trade_value = volume_dec * (exit_price_dec - entry_price_dec)
            
            # Calculate commission
            commission_amount = abs(trade_value * self.base_commission_rate)
            
            # Prepare commission details
            commission_details = {
                'symbol': symbol,
                'trade_type': trade_type.upper(),
                'volume': float(volume_dec),
                'entry_price': float(entry_price_dec),
                'exit_price': float(exit_price_dec),
                'commission_amount': float(commission_amount),
                'commission_currency': 'USD',
                'timestamp': datetime.now()
            }
            
            logger.info(f"Commission calculated: {commission_details}")
            return commission_details
        
        except Exception as e:
            logger.error(f"Commission calculation error: {e}")
            raise
    
    def record_commission(
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
                        symbol, trade_type, volume, 
                        entry_price, exit_price, 
                        commission_amount, commission_currency
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    commission_details['symbol'],
                    commission_details['trade_type'],
                    commission_details['volume'],
                    commission_details['entry_price'],
                    commission_details['exit_price'],
                    commission_details['commission_amount'],
                    commission_details['commission_currency']
                ))
                
                conn.commit()
                logger.info("Commission record added successfully")
        
        except sqlite3.Error as e:
            logger.error(f"Commission recording error: {e}")
            raise
    
    def get_commission_summary(
        self, 
        period: str = 'all_time'
    ) -> Dict[str, Any]:
        """
        Generate commission summary.
        
        :param period: Summary period (all_time/daily/weekly/monthly)
        :return: Commission summary details
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Create pandas connection
                df = pd.read_sql_query("SELECT * FROM commissions", conn)
                
                # Filter by period
                if period == 'daily':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=1)]
                elif period == 'weekly':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(weeks=1)]
                elif period == 'monthly':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
                
                # Calculate summary metrics
                summary = {
                    'period': period,
                    'total_trades': len(df),
                    'total_commission': df['commission_amount'].sum(),
                    'avg_commission_per_trade': df['commission_amount'].mean() if len(df) > 0 else 0,
                    'commission_by_symbol': df.groupby('symbol')['commission_amount'].sum().to_dict(),
                    'commission_by_trade_type': df.groupby('trade_type')['commission_amount'].sum().to_dict()
                }
                
                # Optional: Store summary in database
                self._store_commission_summary(summary)
                
                logger.info(f"Commission summary generated for {period}")
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
        output_path: Optional[str] = None
    ) -> str:
        """
        Export commission data to specified format.
        
        :param export_format: Export file format (csv/excel)
        :param output_path: Custom output path
        :return: Path to exported file
        """
        try:
            # Determine output path
            if not output_path:
                output_dir = os.path.join(os.getcwd(), 'commission_exports')
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f'commission_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                )
            
            # Fetch data
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM commissions", conn)
            
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
    
    def adjust_commission_rate(
        self, 
        new_rate: float
    ):
        """
        Dynamically adjust base commission rate.
        
        :param new_rate: New commission rate
        """
        try:
            # Validate rate
            if not (0 <= new_rate <= 0.01):  # 0-1% reasonable range
                raise ValueError("Commission rate must be between 0-1%")
            
            # Update rate
            self.base_commission_rate = Decimal(str(new_rate))
            
            logger.info(f"Commission rate updated to {new_rate}")
        
        except Exception as e:
            logger.error(f"Commission rate adjustment error: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize Commission Tracker
    commission_tracker = CommissionTracker()
    
    # Example trade commission calculation
    trade_details = commission_tracker.calculate_commission(
        symbol='NZDUSD',
        trade_type='buy',
        volume=0.1,
        entry_price=0.6500,
        exit_price=0.6600
    )
    
    # Record commission
    commission_tracker.record_commission(trade_details)
    
    # Get commission summary
    summary = commission_tracker.get_commission_summary(period='all_time')
    print("Commission Summary:", summary)
    
    # Export commission data
    export_path = commission_tracker.export_commission_data(
        export_format='csv'
    )
    print("Exported to:", export_path)