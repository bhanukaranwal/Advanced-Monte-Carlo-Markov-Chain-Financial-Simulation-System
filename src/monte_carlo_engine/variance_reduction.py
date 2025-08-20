"""
Comprehensive variance reduction techniques for Monte Carlo simulation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)

@dataclass
class VarianceReductionResult:
    """Results from variance reduction technique"""
    original_estimate: float
    reduced_estimate: float
    original_variance: float
    reduced_variance: float
    variance_reduction_ratio: float
    efficiency_gain: float

class VarianceReduction:
    """Main variance reduction orchestrator"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.enabled_methods = self.config.get('enabled_methods', [
            'antithetic', 'control_variates', 'importance_sampling'
        ])
        
    def apply_reduction(
        self,
        simulation_func: Callable,
        n_paths: int,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, VarianceReductionResult]:
        """
        Apply multiple variance reduction techniques
        
        Args:
            simulation_func: Function that generates simulation paths
            n_paths: Number of paths
            methods: List of methods to apply
            **kwargs: Additional arguments for simulation
            
        Returns:
            Dictionary of results for each method
        """
        if methods is None:
            methods = self.enabled_methods
            
        results = {}
        
        # Baseline simulation
        baseline_paths = simulation_func(n_paths, **kwargs)
        baseline_payoffs = self._calculate_payoffs(baseline_paths, kwargs.get('payoff_func'))
        baseline_estimate = np.mean(baseline_payoffs)
        baseline_variance = np.var(baseline_payoffs)
        
        # Apply each variance reduction method
        for method in methods:
            try:
                if method == 'antithetic':
                    result = self._apply_antithetic_variates(
                        simulation_func, n_paths, baseline_estimate, baseline_variance, **kwargs
                    )
                elif method == 'control_variates':
                    result = self._apply_control_variates(
                        simulation_func, n_paths, baseline_estimate, baseline_variance, **kwargs
                    )
                elif method == 'importance_sampling':
                    result = self._apply_importance_sampling(
                        simulation_func, n_paths, baseline_estimate, baseline_variance, **kwargs
                    )
                else:
                    logger.warning(f"Unknown variance reduction method: {method}")
                    continue
                    
                results[method] = result
                logger.info(f"{method}: {result.variance_reduction_ratio:.2f}x variance reduction")
                
            except Exception as e:
                logger.error(f"Error applying {method}: {e}")
                
        return results
        
    def _calculate_payoffs(self, paths: np.ndarray, payoff_func: Optional[Callable]) -> np.ndarray:
        """Calculate payoffs from simulation paths"""
        if payoff_func is None:
            # Default: final value
            return paths[:, -1] if paths.ndim > 1 else paths
        else:
            return np.array([payoff_func(path) for path in paths])
            
    def _apply_antithetic_variates(
        self,
        simulation_func: Callable,
        n_paths: int,
        baseline_estimate: float,
        baseline_variance: float,
        **kwargs
    ) -> VarianceReductionResult:
        """Apply antithetic variates technique"""
        antithetic = AntitheticVariates()
        return antithetic.apply(simulation_func, n_paths, baseline_estimate, baseline_variance, **kwargs)
        
    def _apply_control_variates(
        self,
        simulation_func: Callable,
        n_paths: int,
        baseline_estimate: float,
        baseline_variance: float,
        **kwargs
    ) -> VarianceReductionResult:
        """Apply control variates technique"""
        control_variates = ControlVariates()
        return control_variates.apply(simulation_func, n_paths, baseline_estimate, baseline_variance, **kwargs)
        
    def _apply_importance_sampling(
        self,
        simulation_func: Callable,
        n_paths: int,
        baseline_estimate: float,
        baseline_variance: float,
        **kwargs
    ) -> VarianceReductionResult:
        """Apply importance sampling technique"""
        importance_sampling = ImportanceSampling()
        return importance_sampling.apply(simulation_func, n_paths, baseline_estimate, baseline_variance, **kwargs)

class AntitheticVariates:
    """Antithetic variates variance reduction technique"""
    
    def apply(
        self,
        simulation_func: Callable,
        n_paths: int,
        baseline_estimate: float,
        baseline_variance: float,
        **kwargs
    ) -> VarianceReductionResult:
        """
        Apply antithetic variates
        
        The technique generates pairs of negatively correlated paths
        """
        # Generate half the paths normally
        half_paths = n_paths // 2
        
        # Modify simulation function to generate antithetic pairs
        def antithetic_simulation(n_half_paths, **sim_kwargs):
            # Generate normal paths
            normal_paths = simulation_func(n_half_paths, **sim_kwargs)
            
            # Generate antithetic paths by using -Z instead of Z
            # This requires modifying the random number generation
            antithetic_paths = self._generate_antithetic_paths(
                normal_paths, sim_kwargs
            )
            
            return np.vstack([normal_paths, antithetic_paths])
            
        # Generate antithetic paths
        antithetic_paths = antithetic_simulation(half_paths, **kwargs)
        
        # Calculate payoffs
        payoff_func = kwargs.get('payoff_func')
        antithetic_payoffs = self._calculate_payoffs(antithetic_paths, payoff_func)
        
        # Estimate and variance
        antithetic_estimate = np.mean(antithetic_payoffs)
        antithetic_variance = np.var(antithetic_payoffs)
        
        # Calculate variance reduction
        variance_reduction_ratio = baseline_variance / antithetic_variance if antithetic_variance > 0 else 1.0
        efficiency_gain = variance_reduction_ratio  # Same computational cost
        
        return VarianceReductionResult(
            original_estimate=baseline_estimate,
            reduced_estimate=antithetic_estimate,
            original_variance=baseline_variance,
            reduced_variance=antithetic_variance,
            variance_reduction_ratio=variance_reduction_ratio,
            efficiency_gain=efficiency_gain
        )
        
    def _generate_antithetic_paths(self, normal_paths: np.ndarray, sim_kwargs: Dict) -> np.ndarray:
        """Generate antithetic paths"""
        # This is a simplified implementation
        # In practice, would need to regenerate paths with -Z
        
        # For geometric Brownian motion: S_T = S_0 * exp((μ-σ²/2)T + σ*√T*Z)
        # Antithetic: S_T = S_0 * exp((μ-σ²/2)T + σ*√T*(-Z))
        
        # Simplified: assume final values and create antithetic based on log-returns
        if normal_paths.ndim == 1:
            # Single final values
            log_returns = np.log(normal_paths / sim_kwargs.get('initial_value', 100.0))
            antithetic_log_returns = -log_returns
            antithetic_paths = sim_kwargs.get('initial_value', 100.0) * np.exp(antithetic_log_returns)
        else:
            # Full paths - more complex antithetic generation needed
            initial_value = normal_paths[:, 0]
            returns = np.diff(np.log(normal_paths), axis=1)
            antithetic_returns = -returns
            
            # Reconstruct antithetic paths
            antithetic_paths = np.zeros_like(normal_paths)
            antithetic_paths[:, 0] = initial_value
            
            for t in range(1, normal_paths.shape[1]):
                antithetic_paths[:, t] = antithetic_paths[:, t-1] * np.exp(antithetic_returns[:, t-1])
                
        return antithetic_paths
        
    def _calculate_payoffs(self, paths: np.ndarray, payoff_func: Optional[Callable]) -> np.ndarray:
        """Calculate payoffs from paths"""
        if payoff_func is None:
            return paths[:, -1] if paths.ndim > 1 else paths
        else:
            return np.array([payoff_func(path) for path in paths])

class ControlVariates:
    """Control variates variance reduction technique"""
    
    def apply(
        self,
        simulation_func: Callable,
        n_paths: int,
        baseline_estimate: float,
        baseline_variance: float,
        **kwargs
    ) -> VarianceReductionResult:
        """
        Apply control variates technique
        
        Uses a correlated variable with known expectation
        """
        # Generate simulation paths
        paths = simulation_func(n_paths, **kwargs)
        
        # Calculate target payoffs
        payoff_func = kwargs.get('payoff_func')
        target_payoffs = self._calculate_payoffs(paths, payoff_func)
        
        # Generate control variate
        control_variates, control_mean = self._generate_control_variate(paths, kwargs)
        
        # Find optimal coefficient β
        beta_optimal = self._find_optimal_beta(target_payoffs, control_variates)
        
        # Apply control variate adjustment
        adjusted_payoffs = target_payoffs - beta_optimal * (control_variates - control_mean)
        
        # Calculate results
        reduced_estimate = np.mean(adjusted_payoffs)
        reduced_variance = np.var(adjusted_payoffs)
        
        variance_reduction_ratio = baseline_variance / reduced_variance if reduced_variance > 0 else 1.0
        efficiency_gain = variance_reduction_ratio  # Same computational cost
        
        return VarianceReductionResult(
            original_estimate=baseline_estimate,
            reduced_estimate=reduced_estimate,
            original_variance=baseline_variance,
            reduced_variance=reduced_variance,
            variance_reduction_ratio=variance_reduction_ratio,
            efficiency_gain=efficiency_gain
        )
        
    def _generate_control_variate(self, paths: np.ndarray, kwargs: Dict) -> Tuple[np.ndarray, float]:
        """Generate control variate with known expectation"""
        # Use geometric average as control variate (has known expectation for GBM)
        if paths.ndim == 1:
            # Single values - use the values themselves
            control_variates = paths
            # For GBM: E[S_T] = S_0 * exp(μT)
            mu = kwargs.get('drift', 0.05)
            T = kwargs.get('time_horizon', 1.0)
            S0 = kwargs.get('initial_value', 100.0)
            control_mean = S0 * np.exp(mu * T)
        else:
            # Full paths - use geometric average
            control_variates = stats.gmean(paths, axis=1)
            
            # Expected value of geometric average (approximation)
            mu = kwargs.get('drift', 0.05)
            sigma = kwargs.get('volatility', 0.2)
            T = kwargs.get('time_horizon', 1.0)
            n_steps = paths.shape[1] - 1
            dt = T / n_steps
            
            # Approximate expected value of geometric average
            control_mean = kwargs.get('initial_value', 100.0) * np.exp(
                (mu - sigma**2 / 2) * T / 2
            )
            
        return control_variates, control_mean
        
    def _find_optimal_beta(self, target_payoffs: np.ndarray, control_variates: np.ndarray) -> float:
        """Find optimal control variate coefficient"""
        # β* = Cov(Y, X) / Var(X)
        covariance = np.cov(target_payoffs, control_variates)[0, 1]
        control_variance = np.var(control_variates)
        
        if control_variance > 0:
            return covariance / control_variance
        else:
            return 0.0
            
    def _calculate_payoffs(self, paths: np.ndarray, payoff_func: Optional[Callable]) -> np.ndarray:
        """Calculate payoffs from paths"""
        if payoff_func is None:
            return paths[:, -1] if paths.ndim > 1 else paths
        else:
            return np.array([payoff_func(path) for path in paths])

class ImportanceSampling:
    """Importance sampling variance reduction technique"""
    
    def apply(
        self,
        simulation_func: Callable,
        n_paths: int,
        baseline_estimate: float,
        baseline_variance: float,
        **kwargs
    ) -> VarianceReductionResult:
        """
        Apply importance sampling
        
        Changes the probability distribution to sample more from important regions
        """
        # Determine importance sampling parameters
        importance_params = self._determine_importance_parameters(kwargs)
        
        # Generate paths under importance measure
        importance_paths = self._generate_importance_paths(
            simulation_func, n_paths, importance_params, **kwargs
        )
        
        # Calculate payoffs
        payoff_func = kwargs.get('payoff_func')
        payoffs = self._calculate_payoffs(importance_paths, payoff_func)
        
        # Calculate likelihood ratios
        likelihood_ratios = self._calculate_likelihood_ratios(
            importance_paths, importance_params, kwargs
        )
        
        # Apply importance sampling adjustment
        adjusted_payoffs = payoffs * likelihood_ratios
        
        # Calculate results
        reduced_estimate = np.mean(adjusted_payoffs)
        reduced_variance = np.var(adjusted_payoffs)
        
        variance_reduction_ratio = baseline_variance / reduced_variance if reduced_variance > 0 else 1.0
        efficiency_gain = variance_reduction_ratio  # Same computational cost
        
        return VarianceReductionResult(
            original_estimate=baseline_estimate,
            reduced_estimate=reduced_estimate,
            original_variance=baseline_variance,
            reduced_variance=reduced_variance,
            variance_reduction_ratio=variance_reduction_ratio,
            efficiency_gain=efficiency_gain
        )
        
    def _determine_importance_parameters(self, kwargs: Dict) -> Dict:
        """Determine optimal importance sampling parameters"""
        # For options, shift drift to make in-the-money outcomes more likely
        strike = kwargs.get('strike', 100.0)
        initial_value = kwargs.get('initial_value', 100.0)
        original_drift = kwargs.get('drift', 0.05)
        
        # Simple heuristic: shift drift towards strike
        if strike > initial_value:
            # Out-of-the-money call - increase drift
            new_drift = original_drift + 0.05
        else:
            # In-the-money call - can use original or slightly reduced drift
            new_drift = original_drift
            
        return {
            'new_drift': new_drift,
            'original_drift': original_drift
        }
        
    def _generate_importance_paths(
        self,
        simulation_func: Callable,
        n_paths: int,
        importance_params: Dict,
        **kwargs
    ) -> np.ndarray:
        """Generate paths under importance measure"""
        # Modify kwargs to use importance parameters
        modified_kwargs = kwargs.copy()
        modified_kwargs['drift'] = importance_params['new_drift']
        
        # Generate paths with modified parameters
        return simulation_func(n_paths, **modified_kwargs)
        
    def _calculate_likelihood_ratios(
        self,
        paths: np.ndarray,
        importance_params: Dict,
        kwargs: Dict
    ) -> np.ndarray:
        """Calculate likelihood ratios for importance sampling"""
        # For geometric Brownian motion, likelihood ratio involves change of drift
        new_drift = importance_params['new_drift']
        original_drift = importance_params['original_drift']
        
        if paths.ndim == 1:
            # Single final values
            T = kwargs.get('time_horizon', 1.0)
            sigma = kwargs.get('volatility', 0.2)
            S0 = kwargs.get('initial_value', 100.0)
            
            # Calculate Brownian motion increments (approximately)
            W_T = (np.log(paths / S0) - (new_drift - sigma**2/2) * T) / sigma
            
            # Likelihood ratio: exp(-θW_T - θ²T/2) where θ = (μ_new - μ_old)/σ
            theta = (new_drift - original_drift) / sigma
            likelihood_ratios = np.exp(-theta * W_T - theta**2 * T / 2)
            
        else:
            # Full paths - more complex calculation needed
            n_steps = paths.shape[1] - 1
            T = kwargs.get('time_horizon', 1.0)
            dt = T / n_steps
            sigma = kwargs.get('volatility', 0.2)
            
            # Calculate increments
            log_returns = np.diff(np.log(paths), axis=1)
            
            # Theoretical increments under new measure
            new_increments = (new_drift - sigma**2/2) * dt
            original_increments = (original_drift - sigma**2/2) * dt
            
            # Brownian increments
            dW = (log_returns - new_increments) / sigma
            
            # Likelihood ratio for each path
            theta = (new_drift - original_drift) / sigma
            likelihood_ratios = np.exp(
                -theta * np.sum(dW, axis=1) - theta**2 * T / 2
            )
            
        return likelihood_ratios
        
    def _calculate_payoffs(self, paths: np.ndarray, payoff_func: Optional[Callable]) -> np.ndarray:
        """Calculate payoffs from paths"""
        if payoff_func is None:
            return paths[:, -1] if paths.ndim > 1 else paths
        else:
            return np.array([payoff_func(path) for path in paths])

# Example usage and testing
if __name__ == "__main__":
    # Test variance reduction techniques
    def simple_gbm_simulation(n_paths, initial_value=100, drift=0.05, volatility=0.2, time_horizon=1.0, **kwargs):
        """Simple geometric Brownian motion simulation"""
        dt = time_horizon
        dW = np.random.normal(0, np.sqrt(dt), n_paths)
        final_values = initial_value * np.exp((drift - volatility**2/2) * dt + volatility * dW)
        return final_values
        
    def call_payoff(path, strike=100):
        """Call option payoff"""
        final_value = path[-1] if isinstance(path, np.ndarray) else path
        return max(final_value - strike, 0)
        
    print("Testing Variance Reduction Techniques...")
    
    # Setup simulation parameters
    sim_params = {
        'initial_value': 100,
        'drift': 0.05,
        'volatility': 0.25,
        'time_horizon': 1.0,
        'strike': 110,
        'payoff_func': lambda path: call_payoff(path, strike=110)
    }
    
    # Initialize variance reduction
    vr = VarianceReduction()
    
    # Apply variance reduction techniques
    results = vr.apply_reduction(
        simple_gbm_simulation,
        n_paths=10000,
        methods=['antithetic', 'control_variates', 'importance_sampling'],
        **sim_params
    )
    
    print("\nVariance Reduction Results:")
    print(f"{'Method':<20} {'Original Var':<12} {'Reduced Var':<12} {'VR Ratio':<10} {'Efficiency':<10}")
    print("-" * 70)
    
    for method, result in results.items():
        print(f"{method:<20} {result.original_variance:<12.6f} {result.reduced_variance:<12.6f} "
              f"{result.variance_reduction_ratio:<10.2f} {result.efficiency_gain:<10.2f}")
              
    print("\nVariance reduction techniques test completed!")
