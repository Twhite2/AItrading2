# AI-Powered Forex Trading Bot

## Overview

This is an advanced, AI-driven forex trading bot that combines machine learning, technical analysis, and sophisticated risk management to generate trading signals and execute trades across multiple currency pairs.

## Features

### Key Capabilities
- 🤖 AI-Powered Trading Signals
- 📊 Multi-Timeframe Analysis
- 🧠 Machine Learning Insights
- 📈 Advanced Risk Management
- 🔍 Comprehensive Backtesting
- 🌐 Web API Integration

### Supported Functionality
- Real-time market data fetching
- AI-driven market predictions
- Technical indicator analysis
- Automated trade execution
- Detailed performance tracking
- Commission calculation

## Technology Stack

### Core Technologies
- Python 3.11+
- Hugging Face Transformers
- TA-Lib
- Backtrader & Lumibot
- FastAPI
- SQLite

### Machine Learning
- Flan-T5 trading model
- Custom AI insight generation
- Confidence-based signal evaluation

## Project Structure

```
forex_trading_bot/
├── api/               # Web service endpoints
├── backtesting/       # Backtesting implementations
├── commission_tracking/  # Commission management
├── config/            # Configuration files
├── data/              # Historical and real-time data
├── logs/              # Application logs
├── models/            # AI model utilities
├── strategies/        # Trading strategies
└── utils/             # Supporting utilities
```

## Prerequisites

### System Requirements
- Python 3.11+
- pip
- Virtual environment recommended

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/forex-trading-bot.git
cd forex-trading-bot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
- Create a `.env` file in the project root
- Add necessary API keys and configuration

## Usage

### Running the Bot

#### Web Service Mode (Default)
```bash
python main.py
# Or
python main.py --mode server
```

#### Backtesting
```bash
python main.py --mode backtest
python main.py --mode backtest --symbol NZDUSD --capital 15000
```

#### Live Trading
```bash
python main.py --mode live
python main.py --mode live --symbol EURUSD
```

### API Endpoints

- `/api/start`: Start trading
- `/api/stop`: Stop trading
- `/api/status`: Get current trading status
- `/api/extended/analyze/symbol`: Perform symbol analysis
- `/api/extended/ai/insights`: Generate AI market insights

## Configuration

### Supported Configurations
- Trading symbols
- Initial capital
- Risk management parameters
- Data sources
- Backtesting parameters

### Configuration Files
- `config/deriv_config.json`: Broker API settings
- `config/model_config.json`: AI model configuration
- `.env`: Environment-specific settings

## Logging

- Comprehensive logging across all components
- Log files stored in `logs/` directory
- Configurable log levels

## Backtesting

The bot supports two backtesting frameworks:
- Lumibot
- Backtrader

Generates detailed performance reports including:
- Total returns
- Sharpe ratio
- Maximum drawdown
- Trade analysis

## Security

- API key authentication
- Secure configuration management
- Error handling and logging

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Disclaimer

⚠️ **Risk Warning**: 
Forex trading involves significant risk. This bot is for educational and research purposes. Always conduct thorough testing and risk assessment before using with real funds.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - [Your Email]

Project Link:(https://github.com/Twhite2/forex-trading-bot)
```
