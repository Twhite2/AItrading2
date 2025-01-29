import os
from typing import Dict, Any, Optional, List

from fastapi import (
    APIRouter, 
    HTTPException, 
    Depends, 
    UploadFile, 
    File
)
from pydantic import BaseModel, Field
import pandas as pd

# Import custom components
from utils.data_loader import DataLoader
from utils.technical_indicators import TechnicalIndicators
from models.model_utils import ModelUtils
from strategies.custom_strategy import CustomTradingStrategy
from utils.commission_tracker import CommissionTracker
from strategies.risk_management import RiskManager

# Create a router for additional endpoints
router = APIRouter()

# Request models for input validation
class SymbolAnalysisRequest(BaseModel):
    """
    Request model for symbol analysis endpoint.
    """
    symbol: str = Field(..., description="Trading symbol to analyze")
    timeframe: str = Field(default='hourly', description="Timeframe for analysis")
    indicators: Optional[List[str]] = Field(
        default=['RSI', 'MACD', 'BBANDS'], 
        description="Technical indicators to calculate"
    )

class AIInsightRequest(BaseModel):
    """
    Request model for AI-powered market insights.
    """
    symbol: str = Field(..., description="Trading symbol")
    insight_type: str = Field(
        default='trend_prediction', 
        description="Type of AI insight to generate"
    )

class BacktestConfigRequest(BaseModel):
    """
    Request model for backtesting configuration.
    """
    symbols: List[str] = Field(default=['NZDUSD'], description="Symbols to backtest")
    start_date: Optional[str] = Field(None, description="Backtest start date")
    end_date: Optional[str] = Field(None, description="Backtest end date")
    initial_capital: float = Field(10000.0, description="Initial trading capital")

@router.post("/analyze/symbol")
async def analyze_symbol(request: SymbolAnalysisRequest):
    """
    Perform comprehensive symbol analysis.
    
    :param request: Symbol analysis request parameters
    :return: Detailed symbol analysis
    """
    try:
        # Load market data
        data_loader = DataLoader(symbol=request.symbol)
        market_data = data_loader.fetch_historical_data(
            limit=500,  # Configurable limit
            timeframe=request.timeframe
        )
        
        # Initialize technical indicators
        tech_indicators = TechnicalIndicators(market_data)
        
        # Calculate requested indicators
        analysis_results = {}
        indicator_map = {
            'RSI': lambda: tech_indicators.calculate_rsi(),
            'MACD': lambda: tech_indicators.calculate_macd(),
            'BBANDS': lambda: tech_indicators.calculate_bollinger_bands(),
            'Market_Structure': lambda: tech_indicators.get_market_structure()
        }
        
        # Calculate specified indicators
        for indicator in request.indicators:
            if indicator.upper() in indicator_map:
                analysis_results[indicator] = indicator_map[indicator.upper()]()
        
        return {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "analysis": analysis_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ai/insights")
async def generate_ai_insights(request: AIInsightRequest):
    """
    Generate AI-powered market insights.
    
    :param request: AI insight generation request
    :return: AI-generated market insights
    """
    try:
        # Load market data
        data_loader = DataLoader(symbol=request.symbol)
        market_data = data_loader.fetch_historical_data(limit=200)
        
        # Initialize AI model
        model_utils = ModelUtils()
        
        # Generate insights
        insights = model_utils.generate_market_insight(
            market_data, 
            insight_type=request.insight_type
        )
        
        # Evaluate model confidence
        confidence = model_utils.evaluate_model_confidence(market_data)
        
        return {
            "symbol": request.symbol,
            "insight_type": request.insight_type,
            "insights": insights,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/backtest/configure")
async def configure_backtest(request: BacktestConfigRequest):
    """
    Configure and initiate backtesting.
    
    :param request: Backtest configuration parameters
    :return: Backtest configuration and initial results
    """
    try:
        from backtesting.lumibot_backtest import run_backtest, generate_backtest_report
        
        # Run backtest
        backtest_results = run_backtest(
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )
        
        # Generate report
        generate_backtest_report(backtest_results)
        
        return {
            "status": "Backtest completed",
            "symbols": request.symbols,
            "results": backtest_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/trade-log")
async def upload_trade_log(
    file: UploadFile = File(...),
    commission_tracker: Optional[CommissionTracker] = None
):
    """
    Upload and process trade log file.
    
    :param file: Uploaded trade log file
    :param commission_tracker: Optional commission tracker
    :return: Trade log processing results
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Determine file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.iotools.common.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(pd.iotools.common.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Optional commission tracking
        if commission_tracker:
            # Process trade log for commission calculation
            for _, row in df.iterrows():
                commission_tracker.calculate_commission(
                    symbol=row.get('symbol', 'NZDUSD'),
                    trade_type=row.get('trade_type', 'buy'),
                    entry_price=row.get('entry_price', 0),
                    exit_price=row.get('exit_price', 0),
                    volume=row.get('volume', 0.1)
                )
        
        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/portfolio-summary")
async def get_portfolio_risk_summary(
    risk_manager: RiskManager = None
):
    """
    Retrieve comprehensive portfolio risk summary.
    
    :param risk_manager: Risk management instance
    :return: Portfolio risk metrics
    """
    try:
        if not risk_manager:
            risk_manager = RiskManager()
        
        portfolio_summary = risk_manager.get_portfolio_risk_summary()
        
        return {
            "portfolio_risk": portfolio_summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional helper function to include these endpoints in main app
def include_router(app):
    """
    Include additional endpoints in the main FastAPI application.
    
    :param app: FastAPI application instance
    """
    app.include_router(
        router, 
        prefix="/api/extended", 
        tags=["Extended Trading Bot Endpoints"]
    )

# Optional: Main execution for testing
if __name__ == "__main__":
    from fastapi import FastAPI
    
    app = FastAPI()
    include_router(app)
    
    print("Additional endpoints registered successfully")