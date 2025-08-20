"""
Comprehensive risk analytics framework
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMeasures:
    """Collection of risk measures"""
    var_95: float
    var_99: float
    var_999: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    expected_shortfall_999: float
    maximum_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    omega_ratio: float
    tail_ratio: float
    
@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    portfolio_var: float
    component_var: Dict[str, float]
    marginal_var: Dict[str, float]
    incremental_var: Dict[str, float]
    diversification_ratio: float
    concentration_measure: float
    
@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_name: str
    portfolio_pnl: float
    asset_pnls: Dict[str, float]
    risk_factor_shocks: Dict[str, float]
    probability: Optional[float] = None

class RiskAnalytics:
    """Comprehensive risk analytics framework"""
    
    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99, 0.999],
        estimation_window: int = 252,
        decay_factor: float = 0.94
    ):
        self.confidence_levels = confidence_levels
        self.estimation_window = estimation_window
        self.decay_factor = decay_factor
        
    def calculate_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        method: str = 'historical',
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk using various methods
        
        Args:
            returns: Return series
            method: VaR method ('historical', 'parametric', 'monte_carlo', 'cornish_fisher')
            **kwargs: Method-specific parameters
            
        Returns:
            VaR estimates for different confidence levels
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        else:
            returns = returns[~np.isnan(returns)]
            
        if len(returns) == 0:
            return {f'var_{int(cl*100)}': 0.0 for cl in self.confidence_levels}
            
        var_estimates = {}
        
        if method == 'historical':
            var_estimates = self._historical_var(returns)
        elif method == 'parametric':
            var_estimates = self._parametric_var(returns)
        elif method == 'monte_carlo':
            var_estimates = self._monte_carlo_var(returns, **kwargs)
        elif method == 'cornish_fisher':
            var_estimates = self._cornish_fisher_var(returns)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
        return var_estimates
        
    def _historical_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Historical simulation VaR"""
        var_estimates = {}
        
        for cl in self.confidence_levels:
            var_estimates[f'var_{int(cl*100)}'] = np.percentile(returns, (1-cl)*100)
            
        return var_estimates
        
    def _parametric_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Parametric VaR assuming normal distribution"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        var_estimates = {}
        
        for cl in self.confidence_levels:
            z_score = stats.norm.ppf(1-cl)
            var_estimates[f'var_{int(cl*100)}'] = mean_return + z_score * std_return
            
        return var_estimates
        
    def _monte_carlo_var(
        self, 
        returns: np.ndarray, 
        n_simulations: int = 10000,
        **kwargs
    ) -> Dict[str, float]:
        """Monte Carlo VaR simulation"""
        # Fit distribution to returns
        # For simplicity, using normal distribution
        # In practice, could fit t-distribution, skewed-t, etc.
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate Monte Carlo scenarios
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR from simulated returns
        var_estimates = {}
        
        for cl in self.confidence_levels:
            var_estimates[f'var_{int(cl*100)}'] = np.percentile(simulated_returns, (1-cl)*100)
            
        return var_estimates
        
    def _cornish_fisher_var(self, returns: np.ndarray) -> Dict[str, float]:
        """Cornish-Fisher expansion VaR (adjusts for skewness and kurtosis)"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis
        
        var_estimates = {}
        
        for cl in self.confidence_levels:
            z = stats.norm.ppf(1-cl)
            
            # Cornish-Fisher adjustment
            z_cf = (z + 
                   (z**2 - 1) * skewness / 6 +
                   (z**3 - 3*z) * kurtosis / 24 -
                   (2*z**3 - 5*z) * skewness**2 / 36)
                   
            var_estimates[f'var_{int(cl*100)}'] = mean_return + z_cf * std_return
            
        return var_estimates
        
    def calculate_expected_shortfall(
        self,
        returns: Union[pd.Series, np.ndarray],
        var_estimates: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Return series
            var_estimates: Pre-calculated VaR estimates
            
        Returns:
            Expected Shortfall estimates
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        else:
            returns = returns[~np.isnan(returns)]
            
        if var_estimates is None:
            var_estimates = self.calculate_var(returns)
            
        es_estimates = {}
        
        for cl in self.confidence_levels:
            var_level = var_estimates[f'var_{int(cl*100)}']
            tail_returns = returns[returns <= var_level]
            
            if len(tail_returns) > 0:
                es_estimates[f'es_{int(cl*100)}'] = np.mean(tail_returns)
            else:
                es_estimates[f'es_{int(cl*100)}'] = var_level
                
        return es_estimates
        
    def calculate_maximum_drawdown(
        self,
        prices: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate maximum drawdown statistics
        
        Args:
            prices: Price series or cumulative returns
            
        Returns:
            Drawdown statistics
        """
        if isinstance(prices, pd.Series):
            prices = prices.dropna().values
        else:
            prices = prices[~np.isnan(prices)]
            
        # Calculate running maximum (peak)
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown
        max_dd = np.min(drawdown)
        
        # Drawdown duration
        is_dd = drawdown < 0
        dd_starts = np.where(np.diff(np.concatenate(([False], is_dd))))[0]
        dd_ends = np.where(np.diff(np.concatenate((is_dd, [False]))))
        
        if len(dd_starts) > 0 and len(dd_ends) > 0:
            max_dd_duration = np.max(dd_ends - dd_starts)
        else:
            max_dd_duration = 0
            
        # Recovery time (time to recover from max drawdown)
        max_dd_idx = np.argmin(drawdown)
        recovery_idx = np.where(prices[max_dd_idx:] >= running_max[max_dd_idx])[0]
        recovery_time = recovery_idx if len(recovery_idx) > 0 else len(prices) - max_dd_idx
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'recovery_time': recovery_time,
            'drawdown_series': drawdown
        }
        
    def calculate_risk_adjusted_returns(
        self,
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted return measures
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Risk-adjusted return measures
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values
        else:
            returns = returns[~np.isnan(returns)]
            
        if len(returns) == 0:
            return {}
            
        mean_return = np.mean(returns) * 252  # Annualized
        total_volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / total_volatility if total_volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        # Need price series for max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        max_dd_stats = self.calculate_maximum_drawdown(cumulative_returns)
        calmar_ratio = mean_return / abs(max_dd_stats['max_drawdown']) if max_dd_stats['max_drawdown'] < 0 else 0
        
        # Omega ratio (gains above threshold / losses below threshold)
        threshold = risk_free_rate / 252  # Daily threshold
        gains = returns[returns > threshold]
        losses = returns[returns < threshold]
        
        gain_sum = np.sum(gains - threshold) if len(gains) > 0 else 0
        loss_sum = np.sum(threshold - losses) if len(losses) > 0 else 1e-8
        omega_ratio = gain_sum / loss_sum
        
        # Tail ratio (95th percentile / 5th percentile)
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = p95 / abs(p5) if p5 < 0 else p95
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'tail_ratio': tail_ratio
        }
        
    def calculate_comprehensive_risk_measures(
        self,
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.02
    ) -> RiskMeasures:
        """
        Calculate comprehensive set of risk measures
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate
            
        Returns:
            RiskMeasures object with all calculated measures
        """
        # VaR estimates
        var_estimates = self.calculate_var(returns)
        
        # Expected Shortfall
        es_estimates = self.calculate_expected_shortfall(returns, var_estimates)
        
        # Risk-adjusted returns
        risk_adjusted = self.calculate_risk_adjusted_returns(returns, risk_free_rate)
        
        # Maximum drawdown
        if isinstance(returns, pd.Series):
            ret_values = returns.dropna().values
        else:
            ret_values = returns[~np.isnan(returns)]
            
        cumulative_returns = np.cumprod(1 + ret_values)
        max_dd_stats = self.calculate_maximum_drawdown(cumulative_returns)
        
        return RiskMeasures(
            var_95=var_estimates.get('var_95', 0.0),
            var_99=var_estimates.get('var_99', 0.0),
            var_999=var_estimates.get('var_999', 0.0),
            expected_shortfall_95=es_estimates.get('es_95', 0.0),
            expected_shortfall_99=es_estimates.get('es_99', 0.0),
            expected_shortfall_999=es_estimates.get('es_999', 0.0),
            maximum_drawdown=max_dd_stats['max_drawdown'],
            calmar_ratio=risk_adjusted.get('calmar_ratio', 0.0),
            sortino_ratio=risk_adjusted.get('sortino_ratio', 0.0),
            omega_ratio=risk_adjusted.get('omega_ratio', 0.0),
            tail_ratio=risk_adjusted.get('tail_ratio', 0.0)
        )

class PortfolioRiskAnalyzer:
    """Portfolio-level risk analysis"""
    
    def __init__(self, risk_analytics: RiskAnalytics = None):
        self.risk_analytics = risk_analytics or RiskAnalytics()
        
    def calculate_portfolio_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        method: str = 'analytical'
    ) -> PortfolioRiskMetrics:
        """
        Calculate portfolio VaR and risk decomposition
        
        Args:
            weights: Portfolio weights
            returns: Asset return matrix
            method: Calculation method ('analytical', 'monte_carlo')
            
        Returns:
            Portfolio risk metrics
        """
        # Ensure weights sum to 1
        weights = weights / np.sum(weights)
        
        if method == 'analytical':
            return self._analytical_portfolio_var(weights, returns)
        elif method == 'monte_carlo':
            return self._monte_carlo_portfolio_var(weights, returns)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _analytical_portfolio_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame
    ) -> PortfolioRiskMetrics:
        """Analytical portfolio VaR calculation"""
        # Calculate covariance matrix
        cov_matrix = returns.cov().values
        
        # Portfolio variance
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Portfolio VaR (assuming normal distribution)
        confidence_level = 0.95
        z_score = stats.norm.ppf(1 - confidence_level)
        portfolio_var = -z_score * portfolio_vol
        
        # Component VaR (risk contribution of each asset)
        marginal_var = cov_matrix @ weights / portfolio_vol
        component_var = weights * marginal_var
        
        # Convert to dictionary
        asset_names = returns.columns.tolist()
        component_var_dict = dict(zip(asset_names, component_var))
        marginal_var_dict = dict(zip(asset_names, marginal_var))
        
        # Incremental VaR (change in portfolio VaR from small position change)
        incremental_var_dict = {}
        epsilon = 0.01  # 1% position change
        
        for i, asset in enumerate(asset_names):
            # Create modified weights
            modified_weights = weights.copy()
            modified_weights[i] += epsilon
            modified_weights = modified_weights / np.sum(modified_weights)  # Renormalize
            
            # Calculate modified portfolio VaR
            modified_variance = modified_weights.T @ cov_matrix @ modified_weights
            modified_var = -z_score * np.sqrt(modified_variance)
            
            incremental_var_dict[asset] = modified_var - portfolio_var
            
        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = weights @ individual_vols
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        # Concentration measure (Herfindahl index)
        concentration_measure = np.sum(weights**2)
        
        return PortfolioRiskMetrics(
            portfolio_var=portfolio_var,
            component_var=component_var_dict,
            marginal_var=marginal_var_dict,
            incremental_var=incremental_var_dict,
            diversification_ratio=diversification_ratio,
            concentration_measure=concentration_measure
        )
        
    def _monte_carlo_portfolio_var(
        self,
        weights: np.ndarray,
        returns: pd.DataFrame,
        n_simulations: int = 10000
    ) -> PortfolioRiskMetrics:
        """Monte Carlo portfolio VaR calculation"""
        # Estimate parameters
        mean_returns = returns.mean().values
        cov_matrix = returns.cov().values
        
        # Generate correlated random returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        # Calculate portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        # Calculate VaR
        confidence_level = 0.95
        portfolio_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Component VaR using regression approach
        component_var_dict = {}
        marginal_var_dict = {}
        
        for i, asset in enumerate(returns.columns):
            # Regression of portfolio returns on asset returns
            asset_returns = simulated_returns[:, i]
            
            # Calculate correlation-based contribution
            correlation = np.corrcoef(portfolio_returns, asset_returns)[0, 1]
            asset_vol = np.std(asset_returns)
            portfolio_vol = np.std(portfolio_returns)
            
            marginal_contribution = correlation * asset_vol / portfolio_vol
            component_contribution = weights[i] * marginal_contribution
            
            marginal_var_dict[asset] = marginal_contribution
            component_var_dict[asset] = component_contribution
            
        # Other metrics (same as analytical)
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = weights @ individual_vols
        portfolio_vol_mc = np.std(portfolio_returns)
        diversification_ratio = weighted_avg_vol / portfolio_vol_mc
        
        concentration_measure = np.sum(weights**2)
        
        # Incremental VaR (simplified)
        incremental_var_dict = {
            asset: component_var_dict[asset] * 0.1  # Approximation
            for asset in returns.columns
        }
        
        return PortfolioRiskMetrics(
            portfolio_var=portfolio_var,
            component_var=component_var_dict,
            marginal_var=marginal_var_dict,
            incremental_var=incremental_var_dict,
            diversification_ratio=diversification_ratio,
            concentration_measure=concentration_measure
        )

class StressTestFramework:
    """Comprehensive stress testing framework"""
    
    def __init__(self):
        self.stress_scenarios = {}
        
    def add_historical_scenario(
        self,
        name: str,
        start_date: datetime,
        end_date: datetime,
        market_data: pd.DataFrame
    ):
        """Add historical stress scenario"""
        scenario_data = market_data.loc[start_date:end_date]
        
        if len(scenario_data) > 0:
            # Calculate returns for the stress period
            stress_returns = scenario_data.pct_change().dropna()
            
            self.stress_scenarios[name] = {
                'type': 'historical',
                'returns': stress_returns,
                'start_date': start_date,
                'end_date': end_date,
                'probability': None  # Historical scenarios don't have explicit probabilities
            }
            
    def add_hypothetical_scenario(
        self,
        name: str,
        asset_shocks: Dict[str, float],
        probability: Optional[float] = None
    ):
        """Add hypothetical stress scenario"""
        self.stress_scenarios[name] = {
            'type': 'hypothetical',
            'asset_shocks': asset_shocks,
            'probability': probability
        }
        
    def add_monte_carlo_scenario(
        self,
        name: str,
        distribution_params: Dict[str, Dict[str, float]],
        correlation_matrix: np.ndarray,
        n_simulations: int = 1000
    ):
        """Add Monte Carlo stress scenario"""
        # Generate correlated shocks
        n_assets = len(distribution_params)
        asset_names = list(distribution_params.keys())
        
        # Generate independent normal variables
        independent_shocks = np.random.multivariate_normal(
            np.zeros(n_assets), correlation_matrix, n_simulations
        )
        
        # Transform to desired distributions
        scenario_shocks = []
        
        for i in range(n_simulations):
            shock_dict = {}
            for j, asset in enumerate(asset_names):
                params = distribution_params[asset]
                
                if params.get('distribution', 'normal') == 'normal':
                    shock = params['mean'] + params['std'] * independent_shocks[i, j]
                elif params['distribution'] == 't':
                    # Transform to t-distribution
                    normal_shock = independent_shocks[i, j]
                    t_shock = stats.t.ppf(stats.norm.cdf(normal_shock), params['df'])
                    shock = params['mean'] + params['std'] * t_shock
                else:
                    shock = params['mean'] + params['std'] * independent_shocks[i, j]
                    
                shock_dict[asset] = shock
                
            scenario_shocks.append(shock_dict)
            
        self.stress_scenarios[name] = {
            'type': 'monte_carlo',
            'scenario_shocks': scenario_shocks,
            'n_simulations': n_simulations
        }
        
    def run_stress_tests(
        self,
        portfolio_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[StressTestResult]:
        """
        Run all stress tests on portfolio
        
        Args:
            portfolio_weights: Portfolio weights by asset
            current_prices: Current asset prices
            
        Returns:
            List of stress test results
        """
        results = []
        
        for scenario_name, scenario_data in self.stress_scenarios.items():
            if scenario_data['type'] == 'historical':
                result = self._run_historical_stress(
                    scenario_name, scenario_data, portfolio_weights, current_prices
                )
                results.append(result)
                
            elif scenario_data['type'] == 'hypothetical':
                result = self._run_hypothetical_stress(
                    scenario_name, scenario_data, portfolio_weights, current_prices
                )
                results.append(result)
                
            elif scenario_data['type'] == 'monte_carlo':
                mc_results = self._run_monte_carlo_stress(
                    scenario_name, scenario_data, portfolio_weights, current_prices
                )
                results.extend(mc_results)
                
        return results
        
    def _run_historical_stress(
        self,
        scenario_name: str,
        scenario_data: Dict,
        portfolio_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> StressTestResult:
        """Run historical stress test"""
        stress_returns = scenario_data['returns']
        
        # Calculate portfolio P&L
        portfolio_pnl = 0.0
        asset_pnls = {}
        
        for asset, weight in portfolio_weights.items():
            if asset in stress_returns.columns:
                # Use worst return during stress period
                worst_return = stress_returns[asset].min()
                
                # Calculate P&L
                position_value = weight * sum(current_prices.values())  # Approximate
                asset_pnl = position_value * worst_return
                
                asset_pnls[asset] = asset_pnl
                portfolio_pnl += asset_pnl
            else:
                asset_pnls[asset] = 0.0
                
        # Risk factor shocks (simplified)
        risk_factor_shocks = {
            asset: stress_returns[asset].min() if asset in stress_returns.columns else 0.0
            for asset in portfolio_weights.keys()
        }
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_pnl=portfolio_pnl,
            asset_pnls=asset_pnls,
            risk_factor_shocks=risk_factor_shocks,
            probability=scenario_data.get('probability')
        )
        
    def _run_hypothetical_stress(
        self,
        scenario_name: str,
        scenario_data: Dict,
        portfolio_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> StressTestResult:
        """Run hypothetical stress test"""
        asset_shocks = scenario_data['asset_shocks']
        
        # Calculate portfolio P&L
        portfolio_pnl = 0.0
        asset_pnls = {}
        
        total_portfolio_value = sum(
            weight * current_prices.get(asset, 100.0) 
            for asset, weight in portfolio_weights.items()
        )
        
        for asset, weight in portfolio_weights.items():
            shock = asset_shocks.get(asset, 0.0)
            position_value = weight * total_portfolio_value
            asset_pnl = position_value * shock
            
            asset_pnls[asset] = asset_pnl
            portfolio_pnl += asset_pnl
            
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_pnl=portfolio_pnl,
            asset_pnls=asset_pnls,
            risk_factor_shocks=asset_shocks,
            probability=scenario_data.get('probability')
        )
        
    def _run_monte_carlo_stress(
        self,
        scenario_name: str,
        scenario_data: Dict,
        portfolio_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[StressTestResult]:
        """Run Monte Carlo stress test"""
        scenario_shocks = scenario_data['scenario_shocks']
        results = []
        
        total_portfolio_value = sum(
            weight * current_prices.get(asset, 100.0) 
            for asset, weight in portfolio_weights.items()
        )
        
        for i, shock_dict in enumerate(scenario_shocks):
            portfolio_pnl = 0.0
            asset_pnls = {}
            
            for asset, weight in portfolio_weights.items():
                shock = shock_dict.get(asset, 0.0)
                position_value = weight * total_portfolio_value
                asset_pnl = position_value * shock
                
                asset_pnls[asset] = asset_pnl
                portfolio_pnl += asset_pnl
                
            results.append(StressTestResult(
                scenario_name=f"{scenario_name}_sim_{i}",
                portfolio_pnl=portfolio_pnl,
                asset_pnls=asset_pnls,
                risk_factor_shocks=shock_dict,
                probability=1.0 / len(scenario_shocks)  # Equal probability
            ))
            
        return results
        
    def analyze_stress_results(
        self,
        stress_results: List[StressTestResult]
    ) -> Dict[str, Any]:
        """Analyze stress test results"""
        if not stress_results:
            return {}
            
        # Portfolio P&L statistics
        portfolio_pnls = [result.portfolio_pnl for result in stress_results]
        
        analysis = {
            'worst_case_pnl': min(portfolio_pnls),
            'best_case_pnl': max(portfolio_pnls),
            'average_pnl': np.mean(portfolio_pnls),
            'pnl_std': np.std(portfolio_pnls),
            'pnl_percentiles': {
                '1st': np.percentile(portfolio_pnls, 1),
                '5th': np.percentile(portfolio_pnls, 5),
                '10th': np.percentile(portfolio_pnls, 10),
                '90th': np.percentile(portfolio_pnls, 90),
                '95th': np.percentile(portfolio_pnls, 95),
                '99th': np.percentile(portfolio_pnls, 99)
            }
        }
        
        # Asset-level analysis
        if stress_results:
            first_result = stress_results[0]
            assets = list(first_result.asset_pnls.keys())
            
            asset_analysis = {}
            for asset in assets:
                asset_pnls = [result.asset_pnls.get(asset, 0.0) for result in stress_results]
                
                asset_analysis[asset] = {
                    'worst_pnl': min(asset_pnls),
                    'best_pnl': max(asset_pnls),
                    'avg_pnl': np.mean(asset_pnls),
                    'pnl_std': np.std(asset_pnls)
                }
                
            analysis['asset_analysis'] = asset_analysis
            
        return analysis

# Example usage and testing
if __name__ == "__main__":
    print("Testing Risk Analytics...")
    
    # Generate synthetic return data
    np.random.seed(42)
    n_obs = 1000
    n_assets = 3
    
    # Create correlated returns
    correlation_matrix = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    mean_returns = np.array([0.001, 0.0008, 0.0012])  # Daily returns
    volatilities = np.array([0.02, 0.025, 0.018])    # Daily volatilities
    
    # Generate returns
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_obs)
    
    # Create DataFrame
    return_df = pd.DataFrame(returns, columns=['Asset_A', 'Asset_B', 'Asset_C'])
    
    print(f"Generated {n_obs} observations for {n_assets} assets")
    
    # Test Risk Analytics
    print("\nTesting Risk Analytics:")
    risk_analytics = RiskAnalytics()
    
    # Test individual asset risk measures
    asset_returns = return_df['Asset_A']
    
    # VaR calculations
    var_estimates = risk_analytics.calculate_var(asset_returns, method='historical')
    print(f"Historical VaR: {var_estimates}")
    
    var_parametric = risk_analytics.calculate_var(asset_returns, method='parametric')
    print(f"Parametric VaR: {var_parametric}")
    
    # Expected Shortfall
    es_estimates = risk_analytics.calculate_expected_shortfall(asset_returns)
    print(f"Expected Shortfall: {es_estimates}")
    
    # Comprehensive risk measures
    risk_measures = risk_analytics.calculate_comprehensive_risk_measures(asset_returns)
    print(f"\nComprehensive Risk Measures for Asset_A:")
    print(f"  VaR 95%: {risk_measures.var_95:.4f}")
    print(f"  Expected Shortfall 95%: {risk_measures.expected_shortfall_95:.4f}")
    print(f"  Maximum Drawdown: {risk_measures.maximum_drawdown:.4f}")
    print(f"  Sharpe Ratio: {risk_measures.sortino_ratio:.4f}")
    
    # Test Portfolio Risk Analyzer
    print("\nTesting Portfolio Risk Analyzer:")
    portfolio_analyzer = PortfolioRiskAnalyzer(risk_analytics)
    
    # Equal weights portfolio
    weights = np.array([1/3, 1/3, 1/3])
    
    portfolio_risk = portfolio_analyzer.calculate_portfolio_var(weights, return_df)
    print(f"Portfolio VaR: {portfolio_risk.portfolio_var:.4f}")
    print(f"Diversification Ratio: {portfolio_risk.diversification_ratio:.4f}")
    print(f"Concentration Measure: {portfolio_risk.concentration_measure:.4f}")
    
    print("Component VaR:")
    for asset, comp_var in portfolio_risk.component_var.items():
        print(f"  {asset}: {comp_var:.4f}")
    
    # Test Stress Testing Framework
    print("\nTesting Stress Testing Framework:")
    stress_framework = StressTestFramework()
    
    # Add hypothetical scenario
    stress_framework.add_hypothetical_scenario(
        'market_crash',
        asset_shocks={'Asset_A': -0.20, 'Asset_B': -0.25, 'Asset_C': -0.15},
        probability=0.01
    )
    
    # Add Monte Carlo scenario
    distribution_params = {
        'Asset_A': {'mean': -0.05, 'std': 0.15, 'distribution': 'normal'},
        'Asset_B': {'mean': -0.08, 'std': 0.20, 'distribution': 'normal'},
        'Asset_C': {'mean': -0.03, 'std': 0.12, 'distribution': 'normal'}
    }
    
    stress_framework.add_monte_carlo_scenario(
        'recession_scenario',
        distribution_params,
        correlation_matrix,
        n_simulations=100
    )
    
    # Run stress tests
    portfolio_weights = {'Asset_A': 0.4, 'Asset_B': 0.35, 'Asset_C': 0.25}
    current_prices = {'Asset_A': 100.0, 'Asset_B': 150.0, 'Asset_C': 80.0}
    
    stress_results = stress_framework.run_stress_tests(portfolio_weights, current_prices)
    
    print(f"Ran {len(stress_results)} stress scenarios")
    
    # Analyze results
    stress_analysis = stress_framework.analyze_stress_results(stress_results)
    print(f"Worst case P&L: ${stress_analysis['worst_case_pnl']:,.2f}")
    print(f"Average P&L: ${stress_analysis['average_pnl']:,.2f}")
    print(f"5th percentile P&L: ${stress_analysis['pnl_percentiles']['5th']:,.2f}")
    
    print("\nRisk analytics test completed!")
