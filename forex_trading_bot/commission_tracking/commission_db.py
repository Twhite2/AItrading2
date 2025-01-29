import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager
import logging

class CommissionDatabase:
    """
    Advanced SQLite database management system for commission tracking.
    Provides thread-safe, comprehensive database operations.
    """
    
    def __init__(
        self, 
        db_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the commission tracking database.
        
        :param db_path: Path to SQLite database file
        :param logger: Optional custom logger
        """
        # Determine database path
        self.db_path = db_path or os.path.join(
            os.getcwd(), 'commission_tracking', 'commission_database.sqlite'
        )
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Logging
        self.logger = logger or self._setup_default_logger()
        
        # Thread safety
        self._db_lock = threading.Lock()
        
        # Initialize database
        self._create_database()
    
    def _setup_default_logger(self) -> logging.Logger:
        """
        Create a default logger if none provided.
        
        :return: Configured logger
        """
        logger = logging.getLogger('CommissionDatabase')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('logs/commission_db.log')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    @contextmanager
    def _database_connection(self, commit: bool = False):
        """
        Provide a thread-safe database connection.
        
        :param commit: Whether to commit transactions
        :yield: Database cursor
        """
        with self._db_lock:
            try:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                
                try:
                    yield cursor
                    if commit:
                        connection.commit()
                except Exception as e:
                    connection.rollback()
                    self.logger.error(f"Database operation failed: {e}")
                    raise
                finally:
                    connection.close()
            
            except sqlite3.Error as e:
                self.logger.error(f"Database connection error: {e}")
                raise
    
    def _create_database(self):
        """
        Create initial database schema.
        """
        try:
            with self._database_connection(commit=True) as cursor:
                # Commissions table
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
                        trade_timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Indexes for performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_symbol ON commissions(symbol)
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trade_timestamp 
                    ON commissions(trade_timestamp)
                ''')
                
                # Aggregation table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS commission_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        period TEXT NOT NULL,
                        total_trades INTEGER NOT NULL,
                        total_commission REAL NOT NULL,
                        avg_commission_per_trade REAL NOT NULL,
                        min_commission REAL NOT NULL,
                        max_commission REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                self.logger.info("Database schema created successfully")
        
        except sqlite3.Error as e:
            self.logger.error(f"Database schema creation failed: {e}")
            raise
    
    def insert_commission(
        self, 
        symbol: str, 
        trade_type: str, 
        entry_price: float, 
        exit_price: float, 
        volume: float,
        commission_amount: float,
        commission_rate: float,
        trade_timestamp: Optional[datetime] = None
    ) -> int:
        """
        Insert a new commission record.
        
        :param symbol: Trading symbol
        :param trade_type: Type of trade (buy/sell)
        :param entry_price: Trade entry price
        :param exit_price: Trade exit price
        :param volume: Trading volume
        :param commission_amount: Commission charged
        :param commission_rate: Commission rate
        :param trade_timestamp: Timestamp of the trade
        :return: ID of the inserted record
        """
        try:
            trade_timestamp = trade_timestamp or datetime.now()
            
            with self._database_connection(commit=True) as cursor:
                cursor.execute('''
                    INSERT INTO commissions (
                        symbol, trade_type, entry_price, exit_price, 
                        volume, commission_amount, commission_rate, 
                        trade_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, trade_type, entry_price, exit_price, 
                    volume, commission_amount, commission_rate, 
                    trade_timestamp
                ))
                
                self.logger.info(f"Commission record inserted for {symbol}")
                return cursor.lastrowid
        
        except sqlite3.Error as e:
            self.logger.error(f"Commission insertion failed: {e}")
            raise
    
    def get_commissions(
        self, 
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve commission records with optional filtering.
        
        :param symbol: Optional symbol filter
        :param start_date: Optional start date filter
        :param end_date: Optional end date filter
        :return: DataFrame of commission records
        """
        try:
            with self._database_connection() as cursor:
                # Base query
                query = "SELECT * FROM commissions WHERE 1=1"
                params = []
                
                # Apply symbol filter
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                # Apply date filters
                if start_date:
                    query += " AND trade_timestamp >= ?"
                    params.append(start_date)
                
                if end_date:
                    query += " AND trade_timestamp <= ?"
                    params.append(end_date)
                
                # Execute query and convert to DataFrame
                df = pd.read_sql_query(query, sqlite3.connect(self.db_path), params=params)
                
                self.logger.info(f"Retrieved {len(df)} commission records")
                return df
        
        except sqlite3.Error as e:
            self.logger.error(f"Commission retrieval failed: {e}")
            raise
    
    def calculate_commission_summary(
        self, 
        symbol: Optional[str] = None,
        period: str = 'all_time'
    ) -> Dict[str, Any]:
        """
        Generate comprehensive commission summary.
        
        :param symbol: Optional symbol filter
        :param period: Summary period
        :return: Commission summary dictionary
        """
        try:
            with self._database_connection() as cursor:
                # Prepare base query
                query = """
                    SELECT 
                        symbol,
                        COUNT(*) as total_trades,
                        SUM(commission_amount) as total_commission,
                        AVG(commission_amount) as avg_commission_per_trade,
                        MIN(commission_amount) as min_commission,
                        MAX(commission_amount) as max_commission
                    FROM commissions
                    WHERE 1=1
                """
                params = []
                
                # Apply symbol filter
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                # Apply period filter
                if period != 'all_time':
                    period_filter = {
                        'daily': timedelta(days=1),
                        'weekly': timedelta(weeks=1),
                        'monthly': timedelta(days=30)
                    }
                    
                    query += " AND trade_timestamp >= ?"
                    params.append(
                        datetime.now() - period_filter[period]
                    )
                
                # Group and finalize query
                query += " GROUP BY symbol"
                
                # Execute query
                df = pd.read_sql_query(
                    query, 
                    sqlite3.connect(self.db_path), 
                    params=params
                )
                
                # Convert to dictionary for easy access
                summary = df.to_dict(orient='records')[0] if len(df) > 0 else {}
                
                self.logger.info(f"Generated commission summary for period: {period}")
                return summary
        
        except sqlite3.Error as e:
            self.logger.error(f"Commission summary calculation failed: {e}")
            raise
    
    def export_commissions(
        self, 
        export_format: str = 'csv',
        output_path: Optional[str] = None,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> str:
        """
        Export commission records to a file.
        
        :param export_format: Export file format (csv/excel)
        :param output_path: Custom output path
        :param symbol: Optional symbol filter
        :param start_date: Optional start date filter
        :param end_date: Optional end date filter
        :return: Path to exported file
        """
        try:
            # Retrieve commission data
            df = self.get_commissions(
                symbol=symbol, 
                start_date=start_date, 
                end_date=end_date
            )
            
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
            
            self.logger.info(f"Commission data exported to {full_path}")
            return full_path
        
        except Exception as e:
            self.logger.error(f"Commission export failed: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize database
    commission_db = CommissionDatabase()
    
    # Insert sample commission records
    commission_db.insert_commission(
        symbol='NZDUSD',
        trade_type='buy',
        entry_price=0.6500,
        exit_price=0.6600,
        volume=0.1,
        commission_amount=5.50,
        commission_rate=0.0005
    )
    
    # Retrieve and print commissions
    commissions = commission_db.get_commissions(symbol='NZDUSD')
    print("Commissions:", commissions)
    
    # Generate summary
    summary = commission_db.calculate_commission_summary(
        symbol='NZDUSD', 
        period='daily'
    )
    print("Commission Summary:", summary)
    
    # Export commissions
    export_path = commission_db.export_commissions(
        export_format='csv', 
        symbol='NZDUSD'
    )
    print("Exported to:", export_path)