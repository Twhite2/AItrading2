import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, List, Optional, Union

class TradingVisualizer:
    """
    Comprehensive visualization utility for trading performance analysis.
    Supports both static (Matplotlib) and interactive (Plotly) visualizations.
    """
    
    def __init__(
        self, 
        output_dir: Optional[str] = None
    ):
        """
        Initialize visualization utility.
        
        :param output_dir: Directory to save visualization outputs
        """
        # Set output directory
        self.output_dir = output_dir or os.path.join(os.getcwd(), 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_equity_curve(
        self, 
        trade_log: pd.DataFrame, 
        output_type: str = 'both'
    ):
        """
        Generate equity curve visualization.
        
        :param trade_log: DataFrame with trade log data
        :param output_type: Output type (matplotlib/plotly/both)
        """
        # Ensure trade log has cumulative profit column
        if 'cumulative_profit' not in trade_log.columns:
            trade_log['cumulative_profit'] = trade_log['profit_loss'].cumsum()
        
        # Matplotlib version
        if output_type in ['matplotlib', 'both']:
            plt.figure(figsize=(12, 6))
            plt.plot(trade_log.index, trade_log['cumulative_profit'], color='blue')
            plt.title('Trading Equity Curve')
            plt.xlabel('Trade Number')
            plt.ylabel('Cumulative Profit')
            plt.grid(True)
            
            # Save matplotlib plot
            plt.savefig(os.path.join(self.output_dir, 'equity_curve_matplotlib.png'))
            plt.close()
        
        # Plotly interactive version
        if output_type in ['plotly', 'both']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trade_log.index, 
                y=trade_log['cumulative_profit'],
                mode='lines',
                name='Cumulative Profit'
            ))
            
            fig.update_layout(
                title='Trading Equity Curve (Interactive)',
                xaxis_title='Trade Number',
                yaxis_title='Cumulative Profit'
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.output_dir, 'equity_curve_plotly.html'))
    
    def analyze_trade_distribution(
        self, 
        trade_log: pd.DataFrame, 
        output_type: str = 'both'
    ):
        """
        Visualize trade profit/loss distribution.
        
        :param trade_log: DataFrame with trade log data
        :param output_type: Output type (matplotlib/plotly/both)
        """
        # Matplotlib histogram
        if output_type in ['matplotlib', 'both']:
            plt.figure(figsize=(12, 6))
            plt.hist(trade_log['profit_loss'], bins=30, edgecolor='black')
            plt.title('Trade Profit/Loss Distribution')
            plt.xlabel('Profit/Loss')
            plt.ylabel('Frequency')
            
            # Save matplotlib plot
            plt.savefig(os.path.join(self.output_dir, 'trade_distribution_matplotlib.png'))
            plt.close()
        
        # Plotly box plot
        if output_type in ['plotly', 'both']:
            fig = px.box(trade_log, y='profit_loss', title='Trade Profit/Loss Box Plot')
            
            # Save interactive plot
            fig.write_html(os.path.join(self.output_dir, 'trade_distribution_plotly.html'))
    
    def compare_trading_strategies(
        self, 
        strategy_results: Dict[str, pd.DataFrame], 
        output_type: str = 'both'
    ):
        """
        Compare performance across different trading strategies.
        
        :param strategy_results: Dictionary of strategy DataFrames
        :param output_type: Output type (matplotlib/plotly/both)
        """
        # Prepare data
        comparison_data = []
        for strategy, results in strategy_results.items():
            comparison_data.append({
                'Strategy': strategy,
                'Total Profit': results['profit_loss'].sum(),
                'Win Rate': (results['profit_loss'] > 0).mean(),
                'Max Drawdown': results['profit_loss'].min()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Matplotlib bar plot
        if output_type in ['matplotlib', 'both']:
            plt.figure(figsize=(12, 6))
            plt.bar(
                comparison_df['Strategy'], 
                comparison_df['Total Profit']
            )
            plt.title('Strategy Comparison - Total Profit')
            plt.xlabel('Strategy')
            plt.ylabel('Total Profit')
            plt.xticks(rotation=45)
            
            # Save matplotlib plot
            plt.savefig(os.path.join(self.output_dir, 'strategy_comparison_matplotlib.png'))
            plt.close()
        
        # Plotly multi-metric comparison
        if output_type in ['plotly', 'both']:
            fig = go.Figure()
            
            # Total Profit
            fig.add_trace(go.Bar(
                x=comparison_df['Strategy'],
                y=comparison_df['Total Profit'],
                name='Total Profit'
            ))
            
            # Win Rate
            fig.add_trace(go.Bar(
                x=comparison_df['Strategy'],
                y=comparison_df['Win Rate'],
                name='Win Rate'
            ))
            
            fig.update_layout(
                title='Strategy Performance Comparison',
                xaxis_title='Strategy',
                yaxis_title='Value',
                barmode='group'
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.output_dir, 'strategy_comparison_plotly.html'))
    
    def generate_commission_analysis(
        self, 
        commission_log: pd.DataFrame, 
        output_type: str = 'both'
    ):
        """
        Analyze and visualize commission data.
        
        :param commission_log: DataFrame with commission log data
        :param output_type: Output type (matplotlib/plotly/both)
        """
        # Matplotlib pie chart
        if output_type in ['matplotlib', 'both']:
            commission_by_symbol = commission_log.groupby('symbol')['commission_amount'].sum()
            
            plt.figure(figsize=(10, 6))
            plt.pie(
                commission_by_symbol.values, 
                labels=commission_by_symbol.index, 
                autopct='%1.1f%%'
            )
            plt.title('Commission Distribution by Symbol')
            
            # Save matplotlib plot
            plt.savefig(os.path.join(self.output_dir, 'commission_distribution_matplotlib.png'))
            plt.close()
        
        # Plotly interactive pie chart
        if output_type in ['plotly', 'both']:
            commission_by_symbol = commission_log.groupby('symbol')['commission_amount'].sum().reset_index()
            
            fig = px.pie(
                commission_by_symbol, 
                values='commission_amount', 
                names='symbol', 
                title='Commission Distribution by Symbol (Interactive)'
            )
            
            # Save interactive plot
            fig.write_html(os.path.join(self.output_dir, 'commission_distribution_plotly.html'))
    
    def export_performance_report(
        self, 
        trade_log: pd.DataFrame, 
        commission_log: pd.DataFrame
    ) -> str:
        """
        Generate comprehensive performance report.
        
        :param trade_log: DataFrame with trade log data
        :param commission_log: DataFrame with commission log data
        :return: Path to exported report
        """
        # Calculate performance metrics
        performance_report = {
            'Total Trades': len(trade_log),
            'Winning Trades': (trade_log['profit_loss'] > 0).sum(),
            'Losing Trades': (trade_log['profit_loss'] <= 0).sum(),
            'Total Profit': trade_log['profit_loss'].sum(),
            'Average Profit per Trade': trade_log['profit_loss'].mean(),
            'Total Commissions': commission_log['commission_amount'].sum()
        }
        
        # Export to CSV
        report_path = os.path.join(self.output_dir, 'performance_report.csv')
        pd.DataFrame.from_dict(
            performance_report, 
            orient='index', 
            columns=['Value']
        ).to_csv(report_path)
        
        return report_path

# Example usage
if __name__ == "__main__":
    # Simulate trade log data
    np.random.seed(42)
    trade_log = pd.DataFrame({
        'profit_loss': np.random.normal(50, 100, 200),
        'trade_type': np.random.choice(['buy', 'sell'], 200)
    })
    
    # Simulate commission log data
    commission_log = pd.DataFrame({
        'symbol': np.random.choice(['NZDUSD', 'EURUSD', 'GBPUSD'], 200),
        'commission_amount': np.random.uniform(5, 50, 200)
    })
    
    # Initialize visualizer
    visualizer = TradingVisualizer()
    
    # Generate visualizations
    visualizer.plot_equity_curve(trade_log)
    visualizer.analyze_trade_distribution(trade_log)
    
    # Compare multiple strategies (simulated)
    strategy_results = {
        'AI Strategy': trade_log,
        'Technical Indicator Strategy': pd.DataFrame({
            'profit_loss': np.random.normal(40, 90, 200)
        })
    }
    visualizer.compare_trading_strategies(strategy_results)
    
    # Generate commission analysis
    visualizer.generate_commission_analysis(commission_log)
    
    # Export performance report
    report_path = visualizer.export_performance_report(trade_log, commission_log)
    print(f"Performance report exported to: {report_path}")