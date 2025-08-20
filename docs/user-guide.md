# User Guide

Comprehensive guide to using the Monte Carlo-Markov Finance System.

## Table of Contents

1. [Monte Carlo Simulations](#monte-carlo-simulations)
2. [Markov Models](#markov-models)
3. [Risk Analytics](#risk-analytics)
4. [Backtesting](#backtesting)
5. [Real-time Processing](#real-time-processing)
6. [Visualization](#visualization)

## Monte Carlo Simulations

### Geometric Brownian Motion

from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine

engine = GeometricBrownianMotionEngine(
n_simulations=10000,
n_steps=252,
initial_price=100.0,
drift=0.05, # 5% annual drift
volatility=0.2, # 20% annual volatility
random_seed=42
)

Enable variance reduction
engine.enable_antithetic_variates()
engine.enable_control_variates()

paths = engine.simulate_paths()

text

### Path-Dependent Options

from monte_carlo_engine.path_dependent import PathDependentEngine

engine = PathDependentEngine(
n_simulations=50000,
n_steps=252,
initial_price=100.0,
drift=0.05,
volatility=0.2
)

Asian option
asian_result = engine.price_asian_option(
strike=100.0,
option_type='call',
averaging_start=0,
risk_free_rate=0.03,
time_to_maturity=1.0
)

print(f"Asian option price: ${asian_result['price']:.2f}")
print(f"95% CI: [{asian_result['confidence_interval']:.2f}, {asian_result['confidence_interval']:.2f}]")

text

### Multi-Asset Simulations

from monte_carlo_engine.multi_asset import MultiAssetEngine
import numpy as np

Portfolio setup
initial_prices = [100.0, 200.0, 150.0]
drifts = [0.05, 0.06, 0.04]
volatilities = [0.2, 0.25, 0.18]

correlation_matrix = np.array([
[1.0, 0.6, 0.3],
[0.6, 1.0, 0.4],
[0.3, 0.4, 1.0]
])

engine = MultiAssetEngine(
n_simulations=10000,
n_steps=252,
initial_prices=initial_prices,
drifts=drifts,
volatilities=volatilities,
correlation_matrix=correlation_matrix
)

Generate correlated paths
paths = engine.simulate_correlated_paths()

Price basket option
basket_result = engine.price_basket_option(
weights=[0.4, 0.35, 0.25],
strike=150.0,
option_type='call',
risk_free_rate=0.03,
time_to_maturity=1.0
)

text

## Markov Models

### Hidden Markov Models

from markov_models.hidden_markov import HiddenMarkovModel
import numpy as np

Create HMM for regime detection
hmm = HiddenMarkovModel(n_states=2, n_observations=2)

Sample data: 0 = down day, 1 = up day
returns = np.random.normal(0, 0.02, 1000)
observations = (returns > 0).astype(int)

Fit model
hmm.fit(observations, max_iterations=100)

Decode states
states = hmm.viterbi_decode(observations)
state_probs = hmm.predict_states(observations)

print(f"Transition matrix:\n{hmm.transition_matrix}")
print(f"Emission matrix:\n{hmm.emission_matrix}")

text

### Regime Switching Models

from markov_models.regime_switching import RegimeSwitchingModel
import pandas as pd

Load return data
returns = pd.Series(np.random.normal(0.001, 0.02, 1000))

Create regime-switching model
model = RegimeSwitchingModel(
data=returns,
n_regimes=2,
model_type='variance_switching'
)

Fit model
model.fit(max_iterations=100)

Get regime characteristics
characteristics = model.get_regime_characteristics()
for i, regime in enumerate(characteristics):
print(f"Regime {i}: mean={regime['mean']:.4f}, vol={regime['volatility']:.4f}")

Forecast regime probabilities
forecast = model.forecast_regime_probabilities(steps=5)
print(f"5-step forecast:\n{forecast}")

text

## Risk Analytics

### Value-at-Risk Calculation

from analytics_engine.risk_analytics import RiskAnalytics
import numpy as np

risk_analytics = RiskAnalytics()

Generate sample returns
returns = np.random.normal(0.001, 0.02, 1000)

Calculate VaR using different methods
var_historical = risk_analytics.calculate_var(returns, method='historical')
var_parametric = risk_analytics.calculate_var(returns, method='parametric')
var_monte_carlo = risk_analytics.calculate_var(returns, method='monte_carlo')

print("VaR Estimates:")
print(f"Historical: {var_historical}")
print(f"Parametric: {var_parametric}")
print(f"Monte Carlo: {var_monte_carlo}")

text

### Portfolio Risk Analysis

from analytics_engine.risk_analytics import PortfolioRiskAnalyzer
import pandas as pd

Multi-asset returns
returns_data = pd.DataFrame({
'Asset_A': np.random.normal(0.001, 0.02, 252),
'Asset_B': np.random.normal(0.0008, 0.025, 252),
'Asset_C': np.random.normal(0.0012, 0.018, 252)
})

Portfolio weights
weights = np.array([0.5, 0.3, 0.2])

Risk analysis
analyzer = PortfolioRiskAnalyzer()
portfolio_risk = analyzer.calculate_portfolio_var(weights, returns_data)

print(f"Portfolio VaR: {portfolio_risk.portfolio_var:.4f}")
print(f"Diversification ratio: {portfolio_risk.diversification_ratio:.3f}")
print("Component VaR:", portfolio_risk.component_var)

text

### Stress Testing

from analytics_engine.risk_analytics import StressTestFramework

Initialize stress testing
stress_framework = StressTestFramework()

Add hypothetical scenarios
stress_framework.add_hypothetical_scenario(
'market_crash',
asset_shocks={'Asset_A': -0.20, 'Asset_B': -0.25, 'Asset_C': -0.15}
)

Add Monte Carlo scenario
distribution_params = {
'Asset_A': {'mean': -0.05, 'std': 0.15, 'distribution': 'normal'},
'Asset_B': {'mean': -0.08, 'std': 0.20, 'distribution': 'normal'},
'Asset_C': {'mean': -0.03, 'std': 0.12, 'distribution': 'normal'}
}

stress_framework.add_monte_carlo_scenario(
'recession',
distribution_params,
correlation_matrix=np.eye(3),
n_simulations=1000
)

Run stress tests
portfolio_weights = {'Asset_A': 0.5, 'Asset_B': 0.3, 'Asset_C': 0.2}
current_prices = {'Asset_A': 100, 'Asset_B': 150, 'Asset_C': 80}

results = stress_framework.run_stress_tests(portfolio_weights, current_prices)

Analyze results
analysis = stress_framework.analyze_stress_results(results)
print(f"Worst case P&L: ${analysis['worst_case_pnl']:,.2f}")

text

## Backtesting

### Basic Strategy Backtesting

from validation.backtesting import BacktestEngine
import pandas as pd

Load market data
market_data = pd.DataFrame({
'open': np.random.randn(252).cumsum() + 100,
'high': np.random.randn(252).cumsum() + 102,
'low': np.random.randn(252).cumsum() + 98,
'close': np.random.randn(252).cumsum() + 100,
'volume': np.random.randint(1000000, 10000000, 252)
}, index=pd.date_range('2023-01-01', periods=252))

Define strategy
def moving_average_strategy(data, portfolio, timestamp):
if len(data) < 50:
return {}

text
short_ma = data['close'].rolling(10).mean().iloc[-1]
long_ma = data['close'].rolling(50).mean().iloc[-1]
current_price = data['close'].iloc[-1]

if short_ma > long_ma:  # Buy signal
    return {'close': {'action': 'buy', 'quantity': 100}}
elif short_ma < long_ma:  # Sell signal
    position = portfolio.positions.get('close')
    if position and position.quantity > 0:
        return {'close': {'action': 'sell', 'quantity': position.quantity}}

return {}
Run backtest
engine = BacktestEngine(initial_capital=100000)
engine.set_market_data(market_data)
engine.set_strategy(moving_average_strategy)

result = engine.run_backtest()

print(f"Backtest Results:")
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")

text

## Real-time Processing

### Stream Processing

from real_time_engine.stream_processor import StreamProcessor, MarketTick
from datetime import datetime
import asyncio

async def main():
# Initialize stream processor
processor = StreamProcessor(buffer_size=10000)

text
# Define processing function
def process_tick(tick: MarketTick):
    print(f"Processing {tick.symbol}: ${tick.price}")
    return {'processed_price': tick.price * 1.01}

# Add processing function
processor.add_tick_processor(process_tick)

# Start processing
await processor.start()

# Simulate market data
for i in range(100):
    tick = MarketTick(
        symbol='AAPL',
        timestamp=datetime.now(),
        price=150 + np.random.normal(0, 1),
        volume=1000 + np.random.randint(0, 500)
    )
    await processor.add_tick(tick)
    await asyncio.sleep(0.01)
Run stream processing
asyncio.run(main())

text

### Kalman Filtering

from real_time_engine.kalman_filters import TrendFollowingKalman
import numpy as np

Initialize Kalman filter for price trend
kf = TrendFollowingKalman(
process_noise_std=0.01,
obs_noise_std=0.1
)

Simulate price data
prices = 100 + np.random.randn(1000).cumsum() * 0.5

Filter prices
for price in prices:
kf.predict_and_update(np.array([price]))

Get estimates
price_estimate = kf.get_price_estimate()
trend_estimate = kf.get_trend_estimate()

print(f"Price estimate: ${price_estimate:.2f}")
print(f"Trend estimate: {trend_estimate:.4f}")

text

## Visualization

### Dashboard Creation

from visualization.dashboard import FinanceDashboard
import pandas as pd

Create dashboard
dashboard = FinanceDashboard("Portfolio Dashboard")

Sample portfolio data
portfolio_data = pd.DataFrame({
'portfolio_value': np.cumprod(1 + np.random.normal(0.001, 0.02, 252)) * 100000
}, index=pd.date_range('2023-01-01', periods=252))

Create portfolio overview
fig = dashboard.create_portfolio_overview(portfolio_data)

Display (in Jupyter) or save
fig.show() # or fig.write_html("dashboard.html")

text

### Report Generation

from visualization.report_generator import PDFReportGenerator

Initialize report generator
report = PDFReportGenerator("Monthly Portfolio Report")

Add executive summary
summary_data = {
'portfolio_value': 1050000,
'total_return': 0.05,
'sharpe_ratio': 1.2,
'max_drawdown': -0.08
}

report.add_executive_summary(summary_data)

Generate report
report.generate_report("monthly_report.pdf")
print("Report generated: monthly_report.pdf")

text

## Configuration

### Environment Variables

Performance
export N_THREADS=8
export USE_GPU=true
export MEMORY_LIMIT_GB=16

Data sources
export MARKET_DATA_URL="wss://api.example.com/stream"
export REDIS_URL="redis://localhost:6379"

text

### Configuration File

import yaml

config = {
'simulation': {
'default_n_simulations': 10000,
'use_antithetic_variates': True,
'random_seed': 42
},
'risk': {
'confidence_levels': [0.95, 0.99],
'lookback_window': 252
}
}

with open('config.yaml', 'w') as f:
yaml.dump(config, f)

text

## Best Practices

1. **Memory Management**: Use chunked processing for large datasets
2. **GPU Acceleration**: Enable for simulations >100K paths
3. **Random Seeds**: Set for reproducible results
4. **Validation**: Always validate models before production use
5. **Monitoring**: Use real-time dashboards for live systems

## Troubleshooting

See [Best Practices](best-practices.md) for common issues and solutions.
