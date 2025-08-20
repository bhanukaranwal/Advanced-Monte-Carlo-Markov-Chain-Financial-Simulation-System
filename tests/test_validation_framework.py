"""
Tests for validation framework functionality
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from validation.backtesting import BacktestEngine, Portfolio
from validation.statistical_validation import StatisticalValidator, ModelValidator
from validation.model_verification import ModelVerifier, ConsistencyChecker

class TestBacktestEngine:
    """Test backtesting engine functionality"""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization"""
        portfolio = Portfolio(initial_capital=100000, commission_rate=0.001)
        
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.commission_rate == 0.001
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
        
    def test_trade_execution(self, sample_backtest_data):
        """Test trade execution"""
        portfolio = Portfolio(initial_capital=100000, commission_rate=0.001)
        
        # Execute buy trade
        success = portfolio.execute_trade(
            timestamp=datetime.now(),
            symbol='TEST',
            action='buy',
            quantity=100,
            price=50.0
        )
        
        assert success
        assert 'TEST' in portfolio.positions
        assert portfolio.positions['TEST'].quantity == 100
        assert portfolio.cash < 100000  # Should be reduced by trade value + commission
        
        # Execute sell trade
        success = portfolio.execute_trade(
            timestamp=datetime.now(),
            symbol='TEST',
            action='sell',
            quantity=50,
            price=55.0
        )
        
        assert success
        assert portfolio.positions['TEST'].quantity == 50
        assert len(portfolio.trades) == 2
        
    def test_backtest_execution(self, sample_backtest_data):
        """Test complete backtest execution"""
        market_data = sample_backtest_data['market_data']
        
        def simple_strategy(data, portfolio, timestamp):
            """Simple buy-and-hold strategy"""
            if len(data) == 1:  # First day
                return {'close': {'action': 'buy', 'quantity': 1000}}
            return {}
            
        engine = BacktestEngine(initial_capital=100000)
        engine.set_market_data(market_data)
        engine.set_strategy(simple_strategy)
        
        result = engine.run_backtest(
            start_date=market_data.index[0],
            end_date=market_data.index[-1]
        )
        
        assert result.initial_capital == 100000
        assert result.final_capital > 0
        assert result.start_date == market_data.index
        assert result.end_date == market_data.index[-1]
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        
    def test_performance_metrics_calculation(self, sample_backtest_data):
        """Test performance metrics calculation"""
        market_data = sample_backtest_data['market_data']
        
        def momentum_strategy(data, portfolio, timestamp):
            """Simple momentum strategy"""
            if len(data) < 20:
                return {}
                
            current_price = data['close'].iloc[-1]
            past_price = data['close'].iloc[-20]
            momentum = (current_price / past_price) - 1
            
            if momentum > 0.02:  # Buy signal
                return {'close': {'action': 'buy', 'quantity': 100}}
            elif momentum < -0.02:  # Sell signal
                current_position = portfolio.positions.get('close')
                if current_position and current_position.quantity > 0:
                    return {'close': {'action': 'sell', 'quantity': current_position.quantity}}
            
            return {}
            
        engine = BacktestEngine(initial_capital=100000)
        engine.set_market_data(market_data)
        engine.set_strategy(momentum_strategy)
        
        result = engine.run_backtest()
        
        # Check that all metrics are calculated
        assert isinstance(result.total_return, float)
        assert isinstance(result.annualized_return, float)
        assert isinstance(result.volatility, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.num_trades, int)

class TestStatisticalValidator:
    """Test statistical validation functionality"""
    
    def test_normality_tests(self, sample_returns_data):
        """Test normality tests"""
        validator = StatisticalValidator(significance_level=0.05)
        
        # Test with normal data
        normal_data = np.random.normal(0, 1, 1000)
        result = validator.normality_test(normal_data, 'jarque_bera')
        
        assert result.test_name == 'normality_jarque_bera'
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.is_significant, bool)
        assert 'sample_size' in result.additional_info
        
        # Test with non-normal data
        exponential_data = np.random.exponential(1, 1000)
        result = validator.normality_test(exponential_data, 'jarque_bera')
        
        # Should reject normality for exponential data
        assert result.is_significant  # Should be significant (reject normality)
        
    def test_stationarity_tests(self, sample_returns_data):
        """Test stationarity tests"""
        validator = StatisticalValidator()
        
        # Test with stationary data (returns should be stationary)
        result = validator.stationarity_test(sample_returns_data, 'adf')
        
        assert result.test_name == 'stationarity_adf'
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)
        
        # Test with non-stationary data (random walk)
        non_stationary = np.cumsum(sample_returns_data)
        result = validator.stationarity_test(non_stationary, 'adf')
        
        # Random walk should fail stationarity test
        assert not result.is_significant  # Should not be significant (non-stationary)
        
    def test_autocorrelation_test(self, sample_returns_data):
        """Test autocorrelation tests"""
        validator = StatisticalValidator()
        
        # Test with independent data
        result = validator.autocorrelation_test(sample_returns_data, max_lags=10)
        
        assert result.test_name == 'autocorrelation_ljungbox'
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)
        assert 'max_lags' in result.additional_info
        
    def test_heteroscedasticity_test(self, sample_returns_data):
        """Test heteroscedasticity tests"""
        validator = StatisticalValidator()
        
        # Create residuals and fitted values
        residuals = sample_returns_data.values
        fitted_values = np.random.randn(len(residuals))
        
        result = validator.heteroscedasticity_test(residuals, fitted_values, 'breusch_pagan')
        
        assert result.test_name == 'heteroscedasticity_breusch_pagan'
        assert isinstance(result.test_statistic, float)
        assert isinstance(result.p_value, float)

class TestModelValidator:
    """Test model validation functionality"""
    
    def test_monte_carlo_validation(self, sample_monte_carlo_results):
        """Test Monte Carlo model validation"""
        validator = ModelValidator()
        
        theoretical_moments = {
            'mean': 105.0,
            'variance': 400.0,
            'std': 20.0
        }
        
        simulated_data = sample_monte_carlo_results['final_values']
        
        results = validator.validate_monte_carlo_model(
            theoretical_moments, simulated_data, confidence_level=0.95
        )
        
        assert 'mean_test' in results
        assert 'variance_test' in results
        
        for test_name, result in results.items():
            assert hasattr(result, 'test_statistic')
            assert hasattr(result, 'p_value')
            assert hasattr(result, 'is_significant')
            assert hasattr(result, 'interpretation')
            
    def test_markov_chain_validation(self):
        """Test Markov chain validation"""
        validator = ModelValidator()
        
        # Create theoretical transition matrix
        transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        
        # Create observed transitions (should match theory roughly)
        observed_transitions = np.array([[70, 30], [40, 60]])
        
        # Create state sequence
        state_sequence = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0] * 10
        
        results = validator.validate_markov_chain(
            transition_matrix, observed_transitions, state_sequence
        )
        
        assert 'transition_test' in results
        
        transition_result = results['transition_test']
        assert hasattr(transition_result, 'test_statistic')
        assert hasattr(transition_result, 'p_value')

class TestModelVerifier:
    """Test model verification functionality"""
    
    def test_monte_carlo_convergence(self):
        """Test Monte Carlo convergence verification"""
        def pi_estimation_simulation(n_samples):
            """Estimate π using Monte Carlo"""
            points = np.random.uniform(-1, 1, (n_samples, 2))
            inside_circle = np.sum(points[:, 0]**2 + points[:, 1]**2 <= 1)
            return 4 * inside_circle / n_samples
            
        verifier = ModelVerifier()
        
        result = verifier.verify_monte_carlo_convergence(
            simulation_func=pi_estimation_simulation,
            true_value=np.pi,
            max_iterations=10000,
            batch_size=500,
            relative_tolerance=0.01
        )
        
        assert hasattr(result, 'converged')
        assert hasattr(result, 'final_value')
        assert hasattr(result, 'standard_error')
        assert hasattr(result, 'confidence_interval')
        assert hasattr(result, 'diagnostics')
        
        # Check if estimate is reasonable
        assert abs(result.final_value - np.pi) < 0.1
        
    def test_markov_chain_convergence(self):
        """Test Markov chain convergence verification"""
        verifier = ModelVerifier()
        
        # Simple 2-state transition matrix
        transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        results = verifier.verify_markov_chain_convergence(
            transition_matrix=transition_matrix,
            n_simulations=5000,
            burn_in=500
        )
        
        assert 'stationary_distribution' in results
        assert 'autocorrelation' in results
        
        for test_name, result in results.items():
            assert hasattr(result, 'converged')
            assert hasattr(result, 'diagnostics')

class TestConsistencyChecker:
    """Test consistency checker functionality"""
    
    def test_arbitrage_consistency(self):
        """Test arbitrage consistency checks"""
        checker = ConsistencyChecker(tolerance=1e-3)
        
        # Create consistent option prices (simplified Black-Scholes)
        spot_price = 100
        strike_prices = [90, 95, 100, 105, 110]
        risk_free_rate = 0.05
        time_to_maturity = 0.25
        
        # Generate artificial option prices
        option_prices = {}
        for strike in strike_prices:
            # Simplified intrinsic values
            call_price = max(0, spot_price - strike * np.exp(-risk_free_rate * time_to_maturity))
            put_price = max(0, strike * np.exp(-risk_free_rate * time_to_maturity) - spot_price)
            
            # Add some time value
            call_price += 5.0
            put_price += 5.0
            
            option_prices[f"call_{strike}"] = call_price
            option_prices[f"put_{strike}"] = put_price
            
        results = checker.check_arbitrage_consistency(
            option_prices=option_prices,
            strikes=strike_prices,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            time_to_maturity=time_to_maturity
        )
        
        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, 'test_name')
            assert hasattr(result, 'consistent')
            assert hasattr(result, 'discrepancy_measure')
            assert hasattr(result, 'tolerance')
            
    def test_model_consistency(self):
        """Test general model consistency checks"""
        checker = ConsistencyChecker()
        
        # Sample model outputs
        model_outputs = {
            'portfolio_value': 1000000,
            'cash': 200000,
            'total_assets': 1200000,
            'leverage': 0.5
        }
        
        # Define expected relationships
        expected_relationships = [
            {
                'name': 'balance_check',
                'type': 'equality',
                'lhs': lambda outputs: outputs['cash'] + outputs['portfolio_value'],
                'rhs': lambda outputs: outputs['total_assets']
            },
            {
                'name': 'leverage_bounds',
                'type': 'range',
                'expression': lambda outputs: outputs['leverage'],
                'min': 0.0,
                'max': 1.0
            }
        ]
        
        results = checker.check_model_consistency(
            model_outputs=model_outputs,
            expected_relationships=expected_relationships
        )
        
        assert len(results) == 2
        
        for result in results:
            assert hasattr(result, 'test_name')
            assert hasattr(result, 'consistent')

if __name__ == "__main__":
    pytest.main([__file__])
