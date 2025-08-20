"""
Integration tests for the complete system
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from markov_models.regime_switching import RegimeSwitchingModel
from validation.backtesting import BacktestEngine
from analytics_engine.risk_analytics import RiskAnalytics
from visualization.dashboard import FinanceDashboard

class TestSystemIntegration:
    """Test integration between different system components"""
    
    def test_monte_carlo_to_risk_analytics_integration(self):
        """Test integration between Monte Carlo engine and risk analytics"""
        # Generate paths using Monte Carlo
        mc_engine = GeometricBrownianMotionEngine(
            n_simulations=5000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        mc_engine.set_random_seed(42)
        
        paths = mc_engine.simulate_paths()
        
        # Extract returns for risk analysis
        returns = np.diff(np.log(paths), axis=1)
        portfolio_returns = np.mean(returns, axis=0)  # Average across paths
        
        # Analyze with risk analytics
        risk_analytics = RiskAnalytics()
        
        var_estimates = risk_analytics.calculate_var(portfolio_returns, method='historical')
        es_estimates = risk_analytics.calculate_expected_shortfall(portfolio_returns)
        
        assert 'var_95' in var_estimates
        assert 'es_95' in es_estimates
        assert var_estimates['var_95'] < 0  # VaR should be negative
        assert es_estimates['es_95'] < var_estimates['var_95']  # ES should be more negative than VaR
        
    def test_regime_switching_to_backtesting_integration(self):
        """Test integration between regime switching and backtesting"""
        # Generate synthetic market data with regime changes
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        
        # Create two regimes: low vol and high vol
        regime_1_returns = np.random.normal(0.001, 0.01, 250)  # Low vol
        regime_2_returns = np.random.normal(-0.0005, 0.03, 250)  # High vol, negative drift
        
        returns = np.concatenate([regime_1_returns, regime_2_returns])
        prices = [100.0]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
            
        market_data = pd.DataFrame({
            'close': prices[1:],
            'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices[1:]],
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
            'volume': np.random.randint(1000000, 5000000, 500)
        }, index=dates)
        
        # Detect regimes
        returns_series = pd.Series(returns, index=dates)
        regime_model = RegimeSwitchingModel(
            data=returns_series,
            n_regimes=2,
            model_type='variance_switching'
        )
        regime_model.fit(max_iterations=10)
        
        regime_sequence = regime_model.get_regime_sequence()
        
        # Create regime-aware trading strategy
        def regime_strategy(data, portfolio, timestamp):
            """Strategy that uses regime information"""
            if len(data) < 20:
                return {}
                
            # Get current regime (simplified - would use actual regime detection)
            current_idx = len(data) - 1
            if current_idx < len(regime_sequence):
                current_regime = regime_sequence[current_idx]
                
                # Different strategy per regime
                if current_regime == 0:  # Low vol regime - be more aggressive
                    return {'close': {'action': 'buy', 'quantity': 100}}
                else:  # High vol regime - be defensive
                    current_position = portfolio.positions.get('close')
                    if current_position and current_position.quantity > 50:
                        return {'close': {'action': 'sell', 'quantity': 50}}
                        
            return {}
            
        # Run backtest with regime-aware strategy
        engine = BacktestEngine(initial_capital=100000)
        engine.set_market_data(market_data)
        engine.set_strategy(regime_strategy)
        
        result = engine.run_backtest()
        
        assert result.num_trades > 0
        assert result.final_capital > 0
        
    def test_full_system_workflow(self, sample_price_data):
        """Test complete workflow from data to visualization"""
        # 1. Monte Carlo simulation
        mc_engine = GeometricBrownianMotionEngine(
            n_simulations=1000,
            n_steps=100,
            initial_price=sample_price_data['close'].iloc[0],
            drift=0.05,
            volatility=0.2
        )
        mc_engine.set_random_seed(42)
        
        paths = mc_engine.simulate_paths()
        
        # 2. Risk analysis
        portfolio_returns = np.diff(np.log(paths[:, -50:]), axis=1).mean(axis=0)  # Last 50 steps
        
        risk_analytics = RiskAnalytics()
        risk_measures = risk_analytics.calculate_comprehensive_risk_measures(
            portfolio_returns, risk_free_rate=0.02
        )
        
        # 3. Backtesting
        def simple_strategy(data, portfolio, timestamp):
            if len(data) == 1:
                return {'close': {'action': 'buy', 'quantity': 1000}}
            return {}
            
        engine = BacktestEngine(initial_capital=100000)
        engine.set_market_data(sample_price_data)
        engine.set_strategy(simple_strategy)
        
        backtest_result = engine.run_backtest()
        
        # 4. Dashboard creation
        dashboard = FinanceDashboard("Integration Test Dashboard")
        
        # Create portfolio data from backtest
        portfolio_data = pd.DataFrame({
            'portfolio_value': [100000] * len(sample_price_data)  # Simplified
        }, index=sample_price_data.index)
        
        dashboard_fig = dashboard.create_portfolio_overview(portfolio_data)
        
        # 5. Verify all components worked together
        assert paths.shape[0] == 1000  # MC simulation worked
        assert hasattr(risk_measures, 'var_95')  # Risk analysis worked
        assert backtest_result.num_trades > 0  # Backtesting worked
        assert hasattr(dashboard_fig, 'data')  # Dashboard creation worked
        
        # Integration check: risk measures should be reasonable
        assert -0.1 < risk_measures.var_95 < 0  # VaR should be negative but not too extreme
        assert risk_measures.maximum_drawdown <= 0  # Max drawdown should be negative or zero
        
    def test_performance_consistency(self):
        """Test that different calculation methods give consistent results"""
        np.random.seed(42)
        
        # Generate same underlying random process
        n_sims = 5000
        n_steps = 252
        
        # Method 1: Direct GBM simulation
        returns = np.random.normal(0.0008, 0.015, (n_sims, n_steps))
        prices_method1 = np.zeros((n_sims, n_steps + 1))
        prices_method1[:, 0] = 100.0
        
        for t in range(n_steps):
            prices_method1[:, t + 1] = prices_method1[:, t] * (1 + returns[:, t])
            
        # Method 2: Using Monte Carlo engine
        mc_engine = GeometricBrownianMotionEngine(
            n_simulations=n_sims,
            n_steps=n_steps,
            initial_price=100.0,
            drift=0.0008 * 252,  # Annualized
            volatility=0.015 * np.sqrt(252)  # Annualized
        )
        mc_engine.set_random_seed(42)
        
        prices_method2 = mc_engine.simulate_paths()
        
        # Compare final values
        final_values_1 = prices_method1[:, -1]
        final_values_2 = prices_method2[:, -1]
        
        # Should be similar (within tolerance due to different parameterization)
        mean_diff = abs(np.mean(final_values_1) - np.mean(final_values_2))
        std_diff = abs(np.std(final_values_1) - np.std(final_values_2))
        
        # Allow some difference due to parameterization differences
        assert mean_diff / np.mean(final_values_1) < 0.1  # Within 10%
        assert std_diff / np.std(final_values_1) < 0.2   # Within 20%

if __name__ == "__main__":
    pytest.main([__file__])
