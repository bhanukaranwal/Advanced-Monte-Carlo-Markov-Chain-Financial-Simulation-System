"""
Portfolio Optimization Example using Monte Carlo and Risk Analytics

This script demonstrates:
1. Multi-asset Monte Carlo simulation
2. Portfolio optimization
3. Risk analysis and stress testing
4. Report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# MCMF imports
from monte_carlo_engine.multi_asset import MultiAssetEngine
from analytics_engine.risk_analytics import RiskAnalytics, PortfolioRiskAnalyzer
from analytics_engine.risk_analytics import StressTestFramework
from validation.backtesting import BacktestEngine
from visualization.report_generator import PDFReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main portfolio optimization workflow"""
    
    # Portfolio configuration
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    initial_prices = [150.0, 2800.0, 300.0, 3200.0, 800.0]
    expected_returns = [0.10, 0.12, 0.08, 0.11, 0.15]  # Annual
    volatilities = [0.25, 0.28, 0.22, 0.30, 0.45]      # Annual
    
    # Correlation matrix (example)
    correlation_matrix = np.array([
        [1.00, 0.65, 0.70, 0.60, 0.45],
        [0.65, 1.00, 0.75, 0.70, 0.50],
        [0.70, 0.75, 1.00, 0.65, 0.40],
        [0.60, 0.70, 0.65, 1.00, 0.55],
        [0.45, 0.50, 0.40, 0.55, 1.00]
    ])
    
    logger.info("Starting portfolio optimization analysis...")
    logger.info(f"Assets: {assets}")
    
    # 1. Monte Carlo Simulation
    logger.info("Running Monte Carlo simulation...")
    
    mc_engine = MultiAssetEngine(
        n_simulations=50000,
        n_steps=252,
        initial_prices=initial_prices,
        drifts=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        random_seed=42
    )
    
    # Generate correlated price paths
    paths = mc_engine.simulate_correlated_paths()
    logger.info(f"Generated {paths.shape} simulation paths")
    
    # 2. Portfolio Optimization
    logger.info("Performing portfolio optimization...")
    
    # Define portfolio weights to test
    portfolio_weights = {
        'equal_weight': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        'market_cap': np.array([0.25, 0.20, 0.25, 0.20, 0.10]),
        'conservative': np.array([0.30, 0.25, 0.30, 0.15, 0.00]),
        'growth': np.array([0.15, 0.20, 0.15, 0.20, 0.30])
    }
    
    # Calculate portfolio statistics for each weighting scheme
    portfolio_results = {}
    
    for name, weights in portfolio_weights.items():
        # Calculate portfolio returns
        portfolio_paths = np.sum(paths * weights, axis=2)
        portfolio_returns = np.diff(np.log(portfolio_paths), axis=1)
        
        # Portfolio statistics
        mean_return = np.mean(portfolio_returns) * 252  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = mean_return / volatility
        
        # Risk metrics
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        portfolio_results[name] = {
            'weights': weights,
            'mean_return': mean_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'returns': portfolio_returns
        }
        
        logger.info(f"{name.title()} Portfolio:")
        logger.info(f"  Expected Return: {mean_return:.2%}")
        logger.info(f"  Volatility: {volatility:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    
    # 3. Risk Analysis
    logger.info("Conducting risk analysis...")
    
    risk_analytics = RiskAnalytics()
    portfolio_risk_analyzer = PortfolioRiskAnalyzer(risk_analytics)
    
    # Analyze best performing portfolio (highest Sharpe ratio)
    best_portfolio = max(portfolio_results.items(), key=lambda x: x[1]['sharpe_ratio'])
    best_name, best_data = best_portfolio
    
    logger.info(f"Best portfolio by Sharpe ratio: {best_name}")
    
    # Detailed risk analysis
    returns_data = pd.DataFrame({
        asset: np.diff(np.log(paths[:, :, i]), axis=1).mean(axis=0)
        for i, asset in enumerate(assets)
    })
    
    portfolio_risk = portfolio_risk_analyzer.calculate_portfolio_var(
        best_data['weights'], returns_data
    )
    
    logger.info(f"Portfolio Risk Metrics:")
    logger.info(f"  Portfolio VaR: {portfolio_risk.portfolio_var:.4f}")
    logger.info(f"  Diversification Ratio: {portfolio_risk.diversification_ratio:.3f}")
    logger.info(f"  Concentration Measure: {portfolio_risk.concentration_measure:.3f}")
    
    # 4. Stress Testing
    logger.info("Running stress tests...")
    
    stress_framework = StressTestFramework()
    
    # Add historical scenarios
    stress_scenarios = {
        '2008_financial_crisis': dict(zip(assets, [-0.35, -0.30, -0.40, -0.32, -0.50])),
        'covid_2020': dict(zip(assets, [-0.15, -0.10, -0.20, -0.08, 0.05])),
        'tech_bubble_2000': dict(zip(assets, [-0.45, -0.60, -0.30, -0.70, -0.80])),
        'interest_rate_shock': dict(zip(assets, [-0.20, -0.25, -0.18, -0.22, -0.35]))
    }
    
    for scenario_name, shocks in stress_scenarios.items():
        stress_framework.add_hypothetical_scenario(scenario_name, shocks)
    
    # Current portfolio values (assuming $1M portfolio)
    portfolio_value = 1000000
    current_prices = dict(zip(assets, initial_prices))
    portfolio_weights_dict = dict(zip(assets, best_data['weights']))
    
    stress_results = stress_framework.run_stress_tests(
        portfolio_weights_dict, current_prices
    )
    
    stress_analysis = stress_framework.analyze_stress_results(stress_results)
    
    logger.info("Stress Test Results:")
    logger.info(f"  Worst case P&L: ${stress_analysis['worst_case_pnl']:,.0f}")
    logger.info(f"  5th percentile P&L: ${stress_analysis['pnl_percentiles']['5th']:,.0f}")
    
    # 5. Visualization
    logger.info("Creating visualizations...")
    
    create_portfolio_visualizations(portfolio_results, assets, best_name)
    
    # 6. Generate Report
    logger.info("Generating portfolio report...")
    
    generate_portfolio_report(
        portfolio_results, best_name, portfolio_risk, 
        stress_analysis, assets
    )
    
    logger.info("Portfolio optimization analysis completed!")

def create_portfolio_visualizations(portfolio_results, assets, best_name):
    """Create portfolio analysis visualizations"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Risk-Return Scatter Plot
    for name, data in portfolio_results.items():
        color = 'red' if name == best_name else 'blue'
        ax1.scatter(data['volatility'], data['mean_return'], 
                   s=100, alpha=0.7, color=color, label=name.replace('_', ' ').title())
    
    ax1.set_xlabel('Volatility (Annual)')
    ax1.set_ylabel('Expected Return (Annual)')
    ax1.set_title('Portfolio Risk-Return Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format axes as percentages
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
    
    # 2. Sharpe Ratio Comparison
    names = list(portfolio_results.keys())
    sharpe_ratios = [portfolio_results[name]['sharpe_ratio'] for name in names]
    colors = ['red' if name == best_name else 'blue' for name in names]
    
    bars = ax2.bar(range(len(names)), sharpe_ratios, color=colors, alpha=0.7)
    ax2.set_xlabel('Portfolio')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Sharpe Ratio Comparison')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([name.replace('_', ' ').title() for name in names], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, sharpe_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom')
    
    # 3. Portfolio Weights (Best Portfolio)
    best_weights = portfolio_results[best_name]['weights']
    wedges, texts, autotexts = ax3.pie(best_weights, labels=assets, autopct='%1.1f%%', 
                                       startangle=90)
    ax3.set_title(f'Asset Allocation - {best_name.replace("_", " ").title()}')
    
    # 4. Return Distribution (Best Portfolio)
    best_returns = portfolio_results[best_name]['returns'].flatten()
    ax4.hist(best_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(best_returns), color='red', linestyle='--', 
                label=f'Mean: {np.mean(best_returns):.4f}')
    ax4.axvline(np.percentile(best_returns, 5), color='orange', linestyle='--',
                label=f'5th Percentile: {np.percentile(best_returns, 5):.4f}')
    ax4.set_xlabel('Daily Returns')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Return Distribution - {best_name.replace("_", " ").title()}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_portfolio_report(portfolio_results, best_name, portfolio_risk, 
                            stress_analysis, assets):
    """Generate comprehensive portfolio report"""
    
    # Initialize report generator
    report = PDFReportGenerator("Portfolio Optimization Report")
    
    # Executive Summary
    best_data = portfolio_results[best_name]
    summary_data = {
        'portfolio_name': best_name.replace('_', ' ').title(),
        'expected_return': best_data['mean_return'],
        'volatility': best_data['volatility'],
        'sharpe_ratio': best_data['sharpe_ratio'],
        'portfolio_var': portfolio_risk.portfolio_var,
        'diversification_ratio': portfolio_risk.diversification_ratio,
        'worst_stress_pnl': stress_analysis['worst_case_pnl']
    }
    
    report.add_executive_summary(summary_data)
    
    # Portfolio Analysis Section
    portfolio_analysis_data = {
        'portfolio_results': portfolio_results,
        'best_portfolio': best_name,
        'asset_weights': dict(zip(assets, best_data['weights'])),
        'risk_metrics': {
            'portfolio_var': portfolio_risk.portfolio_var,
            'diversification_ratio': portfolio_risk.diversification_ratio,
            'concentration_measure': portfolio_risk.concentration_measure,
            'component_var': portfolio_risk.component_var
        }
    }
    
    report.add_section("Portfolio Analysis", portfolio_analysis_data, 'portfolio_analysis')
    
    # Risk Analysis Section
    risk_analysis_data = {
        'var_data': {'95%': best_data['var_95'], '99%': best_data['var_99']},
        'risk_attribution': portfolio_risk.component_var,
        'stress_tests': {
            '2008 Financial Crisis': stress_analysis['worst_case_pnl'],
            'Average Scenario': stress_analysis['average_pnl'],
            '5th Percentile': stress_analysis['pnl_percentiles']['5th']
        }
    }
    
    report.add_section("Risk Analysis", risk_analysis_data, 'risk_analysis')
    
    # Generate report
    report_filename = f"portfolio_optimization_report_{datetime.now().strftime('%Y%m%d')}.pdf"
    report.generate_report(report_filename)
    
    logger.info(f"Report generated: {report_filename}")

if __name__ == "__main__":
    main()
