"""
Model verification framework for consistency checking and convergence analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConvergenceResult:
    """Convergence analysis result"""
    metric_name: str
    converged: bool
    convergence_point: Optional[int]
    final_value: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    convergence_rate: float
    diagnostics: Dict[str, Any]

@dataclass
class ConsistencyResult:
    """Model consistency check result"""
    test_name: str
    consistent: bool
    discrepancy_measure: float
    tolerance: float
    details: Dict[str, Any]

class ModelVerifier:
    """Comprehensive model verification framework"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.verification_results = []
        
    def verify_monte_carlo_convergence(
        self,
        simulation_func: Callable,
        true_value: Optional[float] = None,
        max_iterations: int = 100000,
        batch_size: int = 1000,
        confidence_level: float = 0.95,
        relative_tolerance: float = 0.01
    ) -> ConvergenceResult:
        """
        Verify Monte Carlo simulation convergence
        
        Args:
            simulation_func: Function that generates simulation batch
            true_value: Known true value (if available)
            max_iterations: Maximum number of iterations
            batch_size: Size of each simulation batch
            confidence_level: Confidence level for CI
            relative_tolerance: Relative tolerance for convergence
            
        Returns:
            ConvergenceResult with convergence analysis
        """
        estimates = []
        cumulative_estimates = []
        standard_errors = []
        iterations = []
        
        cumulative_sum = 0.0
        cumulative_sum_sq = 0.0
        n_total = 0
        
        converged = False
        convergence_point = None
        
        for iteration in range(1, max_iterations // batch_size + 1):
            # Generate batch
            try:
                batch_results = simulation_func(batch_size)
                if isinstance(batch_results, (list, np.ndarray)):
                    batch_mean = np.mean(batch_results)
                else:
                    batch_mean = float(batch_results)
            except Exception as e:
                logger.error(f"Error in simulation batch {iteration}: {e}")
                continue
                
            estimates.append(batch_mean)
            
            # Update cumulative statistics
            cumulative_sum += batch_mean * batch_size
            cumulative_sum_sq += np.sum(np.array(batch_results) ** 2) if isinstance(batch_results, (list, np.ndarray)) else batch_mean ** 2 * batch_size
            n_total += batch_size
            
            # Current estimate
            current_estimate = cumulative_sum / n_total
            cumulative_estimates.append(current_estimate)
            
            # Standard error
            if n_total > 1:
                variance = (cumulative_sum_sq - cumulative_sum ** 2 / n_total) / (n_total - 1)
                std_error = np.sqrt(variance / n_total)
            else:
                std_error = 0.0
                
            standard_errors.append(std_error)
            iterations.append(n_total)
            
            # Check convergence
            if iteration > 10:  # Need some history
                # Relative change criterion
                recent_estimates = cumulative_estimates[-5:]
                relative_change = abs(recent_estimates[-1] - recent_estimates[0]) / abs(recent_estimates + 1e-8)
                
                # Standard error criterion
                se_criterion = std_error / abs(current_estimate + 1e-8) < relative_tolerance
                
                if relative_change < relative_tolerance and se_criterion and not converged:
                    converged = True
                    convergence_point = n_total
                    
            # Stop if converged and enough iterations
            if converged and iteration > 20:
                break
                
        # Final statistics
        final_estimate = cumulative_estimates[-1] if cumulative_estimates else 0.0
        final_std_error = standard_errors[-1] if standard_errors else 0.0
        
        # Confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = final_estimate - z_score * final_std_error
        ci_upper = final_estimate + z_score * final_std_error
        
        # Convergence rate estimation
        if len(cumulative_estimates) > 10:
            # Fit power law: error ~ n^(-rate)
            x = np.log(iterations[-10:])
            if true_value is not None:
                y = np.log(np.abs(np.array(cumulative_estimates[-10:]) - true_value) + 1e-10)
            else:
                # Use final estimate as proxy for true value
                y = np.log(standard_errors[-10:])
            
            try:
                slope, intercept = np.polyfit(x, y, 1)
                convergence_rate = -slope  # Rate should be positive
            except:
                convergence_rate = 0.5  # Default rate
        else:
            convergence_rate = 0.5
            
        # Diagnostics
        diagnostics = {
            'total_iterations': n_total,
            'batch_iterations': len(estimates),
            'estimates_history': cumulative_estimates,
            'standard_errors_history': standard_errors,
            'iterations_history': iterations,
            'relative_tolerance_used': relative_tolerance
        }
        
        if true_value is not None:
            diagnostics['bias'] = final_estimate - true_value
            diagnostics['absolute_error'] = abs(final_estimate - true_value)
            diagnostics['relative_error'] = abs(final_estimate - true_value) / abs(true_value + 1e-8)
            
        return ConvergenceResult(
            metric_name="monte_carlo_convergence",
            converged=converged,
            convergence_point=convergence_point,
            final_value=final_estimate,
            standard_error=final_std_error,
            confidence_interval=(ci_lower, ci_upper),
            convergence_rate=convergence_rate,
            diagnostics=diagnostics
        )
        
    def verify_markov_chain_convergence(
        self,
        transition_matrix: np.ndarray,
        n_simulations: int = 10000,
        burn_in: int = 1000,
        lag: int = 1
    ) -> Dict[str, ConvergenceResult]:
        """
        Verify Markov chain convergence to stationary distribution
        
        Args:
            transition_matrix: Transition probability matrix
            n_simulations: Number of simulation steps
            burn_in: Burn-in period
            lag: Lag for autocorrelation analysis
            
        Returns:
            Dictionary of convergence results
        """
        n_states = transition_matrix.shape[0]
        
        # Calculate theoretical stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvalues))
        theoretical_stationary = np.real(eigenvectors[:, stationary_idx])
        theoretical_stationary = theoretical_stationary / np.sum(theoretical_stationary)
        
        # Simulate Markov chain
        states = np.zeros(n_simulations, dtype=int)
        states[0] = 0  # Start from state 0
        
        for t in range(1, n_simulations):
            current_state = states[t-1]
            states[t] = np.random.choice(n_states, p=transition_matrix[current_state])
            
        # Analyze convergence
        results = {}
        
        # 1. Convergence to stationary distribution
        window_size = 500
        empirical_distributions = []
        
        for i in range(burn_in, n_simulations - window_size, window_size):
            window_states = states[i:i + window_size]
            empirical_dist = np.bincount(window_states, minlength=n_states) / window_size
            empirical_distributions.append(empirical_dist)
            
        # Calculate convergence metrics
        kl_divergences = []
        tv_distances = []  # Total variation distances
        
        for emp_dist in empirical_distributions:
            # KL divergence
            kl_div = np.sum(emp_dist * np.log((emp_dist + 1e-8) / (theoretical_stationary + 1e-8)))
            kl_divergences.append(kl_div)
            
            # Total variation distance
            tv_dist = 0.5 * np.sum(np.abs(emp_dist - theoretical_stationary))
            tv_distances.append(tv_dist)
            
        # Check convergence
        convergence_threshold = 0.01
        kl_converged = len(kl_divergences) > 5 and all(kl < convergence_threshold for kl in kl_divergences[-5:])
        tv_converged = len(tv_distances) > 5 and all(tv < convergence_threshold for tv in tv_distances[-5:])
        
        results['stationary_distribution'] = ConvergenceResult(
            metric_name="stationary_distribution_convergence",
            converged=kl_converged and tv_converged,
            convergence_point=None,  # Would need more detailed analysis
            final_value=kl_divergences[-1] if kl_divergences else 0.0,
            standard_error=np.std(kl_divergences) if len(kl_divergences) > 1 else 0.0,
            confidence_interval=(0, 0),  # Simplified
            convergence_rate=0.0,  # Would need detailed analysis
            diagnostics={
                'theoretical_stationary': theoretical_stationary.tolist(),
                'final_empirical': empirical_distributions[-1].tolist() if empirical_distributions else [],
                'kl_divergences': kl_divergences,
                'tv_distances': tv_distances
            }
        )
        
        # 2. Autocorrelation analysis
        autocorrelations = []
        for s in range(n_states):
            state_indicators = (states[burn_in:] == s).astype(int)
            if len(state_indicators) > lag:
                autocorr = np.corrcoef(state_indicators[:-lag], state_indicators[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrelations.append(autocorr)
                    
        avg_autocorr = np.mean(autocorrelations) if autocorrelations else 0.0
        autocorr_converged = abs(avg_autocorr) < 0.1  # Low autocorrelation indicates convergence
        
        results['autocorrelation'] = ConvergenceResult(
            metric_name="autocorrelation_analysis",
            converged=autocorr_converged,
            convergence_point=None,
            final_value=avg_autocorr,
            standard_error=np.std(autocorrelations) if len(autocorrelations) > 1 else 0.0,
            confidence_interval=(0, 0),
            convergence_rate=0.0,
            diagnostics={
                'lag': lag,
                'autocorrelations_by_state': autocorrelations,
                'burn_in': burn_in
            }
        )
        
        return results

class ConsistencyChecker:
    """Check consistency between different models and methods"""
    
    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance
        
    def check_arbitrage_consistency(
        self,
        option_prices: Dict[str, float],
        strikes: List[float],
        spot_price: float,
        risk_free_rate: float,
        time_to_maturity: float
    ) -> List[ConsistencyResult]:
        """
        Check for arbitrage opportunities in option prices
        
        Args:
            option_prices: Dictionary of option prices {strike: price}
            strikes: List of strike prices
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            time_to_maturity: Time to maturity
            
        Returns:
            List of consistency check results
        """
        results = []
        
        # Sort strikes and prices
        sorted_strikes = sorted(strikes)
        
        # 1. Call option lower bounds: C >= max(0, S - K*e^(-rT))
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        
        for strike in sorted_strikes:
            if f"call_{strike}" in option_prices:
                call_price = option_prices[f"call_{strike}"]
                lower_bound = max(0, spot_price - strike * discount_factor)
                
                violation = call_price < lower_bound - self.tolerance
                discrepancy = lower_bound - call_price if violation else 0
                
                results.append(ConsistencyResult(
                    test_name=f"call_lower_bound_{strike}",
                    consistent=not violation,
                    discrepancy_measure=discrepancy,
                    tolerance=self.tolerance,
                    details={
                        'call_price': call_price,
                        'lower_bound': lower_bound,
                        'strike': strike
                    }
                ))
                
        # 2. Put-call parity: C - P = S - K*e^(-rT)
        for strike in sorted_strikes:
            call_key = f"call_{strike}"
            put_key = f"put_{strike}"
            
            if call_key in option_prices and put_key in option_prices:
                call_price = option_prices[call_key]
                put_price = option_prices[put_key]
                
                lhs = call_price - put_price
                rhs = spot_price - strike * discount_factor
                
                discrepancy = abs(lhs - rhs)
                violation = discrepancy > self.tolerance
                
                results.append(ConsistencyResult(
                    test_name=f"put_call_parity_{strike}",
                    consistent=not violation,
                    discrepancy_measure=discrepancy,
                    tolerance=self.tolerance,
                    details={
                        'call_price': call_price,
                        'put_price': put_price,
                        'lhs': lhs,
                        'rhs': rhs,
                        'strike': strike
                    }
                ))
                
        # 3. Call option convexity
        call_strikes = [k for k in sorted_strikes if f"call_{k}" in option_prices]
        
        for i in range(1, len(call_strikes) - 1):
            k1, k2, k3 = call_strikes[i-1], call_strikes[i], call_strikes[i+1]
            c1 = option_prices[f"call_{k1}"]
            c2 = option_prices[f"call_{k2}"]
            c3 = option_prices[f"call_{k3}"]
            
            # Convexity condition: (c3 - c2)/(k3 - k2) <= (c2 - c1)/(k2 - k1)
            if k3 != k2 and k2 != k1:
                slope_right = (c3 - c2) / (k3 - k2)
                slope_left = (c2 - c1) / (k2 - k1)
                
                violation = slope_right > slope_left + self.tolerance
                discrepancy = max(0, slope_right - slope_left)
                
                results.append(ConsistencyResult(
                    test_name=f"call_convexity_{k1}_{k2}_{k3}",
                    consistent=not violation,
                    discrepancy_measure=discrepancy,
                    tolerance=self.tolerance,
                    details={
                        'strikes': [k1, k2, k3],
                        'prices': [c1, c2, c3],
                        'slope_left': slope_left,
                        'slope_right': slope_right
                    }
                ))
                
        return results
        
    def check_model_consistency(
        self,
        model_outputs: Dict[str, Any],
        expected_relationships: List[Dict[str, Any]]
    ) -> List[ConsistencyResult]:
        """
        Check consistency between model outputs and expected relationships
        
        Args:
            model_outputs: Dictionary of model outputs
            expected_relationships: List of expected relationships to check
            
        Returns:
            List of consistency check results
        """
        results = []
        
        for relationship in expected_relationships:
            test_name = relationship.get('name', 'unnamed_test')
            relationship_type = relationship.get('type', 'equality')
            
            try:
                if relationship_type == 'equality':
                    lhs = self._evaluate_expression(relationship['lhs'], model_outputs)
                    rhs = self._evaluate_expression(relationship['rhs'], model_outputs)
                    
                    discrepancy = abs(lhs - rhs)
                    consistent = discrepancy <= self.tolerance
                    
                elif relationship_type == 'inequality':
                    lhs = self._evaluate_expression(relationship['lhs'], model_outputs)
                    rhs = self._evaluate_expression(relationship['rhs'], model_outputs)
                    operator = relationship.get('operator', '<=')
                    
                    if operator == '<=':
                        consistent = lhs <= rhs + self.tolerance
                        discrepancy = max(0, lhs - rhs)
                    elif operator == '>=':
                        consistent = lhs >= rhs - self.tolerance
                        discrepancy = max(0, rhs - lhs)
                    elif operator == '<':
                        consistent = lhs < rhs + self.tolerance
                        discrepancy = max(0, lhs - rhs)
                    elif operator == '>':
                        consistent = lhs > rhs - self.tolerance
                        discrepancy = max(0, rhs - lhs)
                    else:
                        raise ValueError(f"Unknown operator: {operator}")
                        
                elif relationship_type == 'range':
                    value = self._evaluate_expression(relationship['expression'], model_outputs)
                    min_val = relationship.get('min', float('-inf'))
                    max_val = relationship.get('max', float('inf'))
                    
                    consistent = min_val <= value <= max_val
                    discrepancy = max(0, min_val - value, value - max_val)
                    
                else:
                    raise ValueError(f"Unknown relationship type: {relationship_type}")
                    
                results.append(ConsistencyResult(
                    test_name=test_name,
                    consistent=consistent,
                    discrepancy_measure=discrepancy,
                    tolerance=self.tolerance,
                    details=relationship
                ))
                
            except Exception as e:
                logger.error(f"Error checking relationship {test_name}: {e}")
                results.append(ConsistencyResult(
                    test_name=test_name,
                    consistent=False,
                    discrepancy_measure=float('inf'),
                    tolerance=self.tolerance,
                    details={'error': str(e)}
                ))
                
        return results
        
    def _evaluate_expression(self, expression: Union[str, float, Callable], model_outputs: Dict) -> float:
        """Evaluate expression given model outputs"""
        if isinstance(expression, (int, float)):
            return float(expression)
        elif isinstance(expression, str):
            # Simple variable lookup
            return float(model_outputs.get(expression, 0))
        elif callable(expression):
            return float(expression(model_outputs))
        else:
            raise ValueError(f"Unknown expression type: {type(expression)}")

class ConvergenceAnalyzer:
    """Advanced convergence analysis tools"""
    
    def __init__(self):
        pass
        
    def analyze_mcmc_convergence(
        self,
        chains: List[np.ndarray],
        parameter_names: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze MCMC chain convergence using multiple diagnostics
        
        Args:
            chains: List of MCMC chains (each chain is array of samples)
            parameter_names: Names of parameters
            
        Returns:
            Dictionary of convergence diagnostics
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(chains))]
            
        results = {}
        
        for i, chain in enumerate(chains):
            param_name = parameter_names[i]
            
            # 1. Geweke diagnostic
            geweke_stat = self._geweke_diagnostic(chain)
            
            # 2. Effective sample size
            eff_size = self._effective_sample_size(chain)
            
            # 3. Autocorrelation
            autocorr = self._autocorrelation_time(chain)
            
            # 4. Running mean stability
            running_mean_stable = self._running_mean_stability(chain)
            
            results[param_name] = {
                'geweke_statistic': geweke_stat,
                'effective_sample_size': eff_size,
                'autocorrelation_time': autocorr,
                'running_mean_stable': running_mean_stable,
                'chain_length': len(chain)
            }
            
        return results
        
    def _geweke_diagnostic(self, chain: np.ndarray, first_frac: float = 0.1, last_frac: float = 0.5) -> float:
        """Geweke convergence diagnostic"""
        n = len(chain)
        
        # First part of chain
        first_n = int(first_frac * n)
        first_part = chain[:first_n]
        
        # Last part of chain
        last_n = int(last_frac * n)
        last_part = chain[-last_n:]
        
        # Means
        mean_first = np.mean(first_part)
        mean_last = np.mean(last_part)
        
        # Spectral densities (simplified using standard error)
        se_first = np.std(first_part) / np.sqrt(len(first_part))
        se_last = np.std(last_part) / np.sqrt(len(last_part))
        
        # Z-score
        z_score = (mean_first - mean_last) / np.sqrt(se_first**2 + se_last**2)
        
        return z_score
        
    def _effective_sample_size(self, chain: np.ndarray) -> float:
        """Calculate effective sample size"""
        n = len(chain)
        
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, min(n//4, 100)):  # Limit lag to prevent noise
            if lag < n:
                autocorr = np.corrcoef(chain[:-lag], chain[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
                else:
                    break
            else:
                break
                
        if not autocorrs:
            return n
            
        # Find first negative autocorrelation or cutoff
        tau_int = 1.0
        for i, rho in enumerate(autocorrs):
            if rho <= 0:
                break
            tau_int += 2 * rho
            
        effective_size = n / (2 * tau_int)
        return max(1, effective_size)
        
    def _autocorrelation_time(self, chain: np.ndarray) -> float:
        """Estimate autocorrelation time"""
        # Simplified autocorrelation time estimation
        autocorrs = []
        for lag in range(1, min(len(chain)//10, 50)):
            if lag < len(chain):
                autocorr = np.corrcoef(chain[:-lag], chain[lag:])[0, 1]
                if not np.isnan(autocorr) and autocorr > 0:
                    autocorrs.append((lag, autocorr))
                else:
                    break
                    
        if not autocorrs:
            return 1.0
            
        # Find where autocorrelation drops below 1/e
        threshold = 1.0 / np.e
        for lag, rho in autocorrs:
            if rho < threshold:
                return float(lag)
                
        return float(autocorrs[-1][0])  # Use last computed lag
        
    def _running_mean_stability(self, chain: np.ndarray, window: int = None) -> bool:
        """Check if running mean is stable"""
        if window is None:
            window = max(50, len(chain) // 20)
            
        if len(chain) < window * 2:
            return True  # Too short to assess
            
        # Calculate running means
        running_means = []
        for i in range(window, len(chain), window // 2):
            running_means.append(np.mean(chain[:i]))
            
        if len(running_means) < 3:
            return True
            
        # Check if recent means are stable (low variance)
        recent_means = running_means[-3:]
        stability = np.std(recent_means) / (abs(np.mean(recent_means)) + 1e-8) < 0.01
        
        return stability

# Example usage and testing
if __name__ == "__main__":
    print("Testing Model Verification Framework...")
    
    # Test Monte Carlo convergence verification
    print("Testing Monte Carlo convergence verification:")
    
    def simple_mc_simulation(n_samples):
        """Simple Monte Carlo: estimate π using unit circle"""
        points = np.random.uniform(-1, 1, (n_samples, 2))
        inside_circle = np.sum(points[:, 0]**2 + points[:, 1]**2 <= 1)
        return 4 * inside_circle / n_samples
        
    verifier = ModelVerifier()
    
    convergence_result = verifier.verify_monte_carlo_convergence(
        simulation_func=simple_mc_simulation,
        true_value=np.pi,
        max_iterations=50000,
        batch_size=1000,
        relative_tolerance=0.01
    )
    
    print(f"MC Convergence Analysis:")
    print(f"  Converged: {convergence_result.converged}")
    print(f"  Convergence Point: {convergence_result.convergence_point}")
    print(f"  Final Estimate: {convergence_result.final_value:.6f} (true: {np.pi:.6f})")
    print(f"  Standard Error: {convergence_result.standard_error:.6f}")
    print(f"  95% CI: [{convergence_result.confidence_interval[0]:.6f}, {convergence_result.confidence_interval[1]:.6f}]")
    print(f"  Convergence Rate: {convergence_result.convergence_rate:.3f}")
    
    if 'bias' in convergence_result.diagnostics:
        print(f"  Bias: {convergence_result.diagnostics['bias']:.6f}")
        print(f"  Relative Error: {convergence_result.diagnostics['relative_error']:.4%}")
    
    # Test Markov chain convergence
    print("\nTesting Markov chain convergence:")
    
    # Create a simple 3-state Markov chain
    transition_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1], 
        [0.2, 0.3, 0.5]
    ])
    
    markov_results = verifier.verify_markov_chain_convergence(
        transition_matrix=transition_matrix,
        n_simulations=10000,
        burn_in=1000
    )
    
    for test_name, result in markov_results.items():
        print(f"\n{test_name.upper()}:")
        print(f"  Converged: {result.converged}")
        print(f"  Final Value: {result.final_value:.6f}")
        print(f"  Standard Error: {result.standard_error:.6f}")
        
        if 'theoretical_stationary' in result.diagnostics:
            theoretical = result.diagnostics['theoretical_stationary']
            empirical = result.diagnostics.get('final_empirical', [])
            print(f"  Theoretical Stationary: {[f'{x:.3f}' for x in theoretical]}")
            if empirical:
                print(f"  Empirical Stationary: {[f'{x:.3f}' for x in empirical]}")
    
    # Test Consistency Checker
    print("\nTesting Consistency Checker:")
    consistency_checker = ConsistencyChecker(tolerance=1e-3)
    
    # Test arbitrage consistency with sample option prices
    spot_price = 100
    risk_free_rate = 0.05
    time_to_maturity = 0.25  # 3 months
    
    # Generate some option prices (simplified Black-Scholes)
    from scipy.stats import norm
    
    def black_scholes_call(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        
    def black_scholes_put(S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    strikes = [90, 95, 100, 105, 110]
    volatility = 0.2
    
    option_prices = {}
    for strike in strikes:
        call_price = black_scholes_call(spot_price, strike, time_to_maturity, risk_free_rate, volatility)
        put_price = black_scholes_put(spot_price, strike, time_to_maturity, risk_free_rate, volatility)
        option_prices[f"call_{strike}"] = call_price
        option_prices[f"put_{strike}"] = put_price
    
    arbitrage_results = consistency_checker.check_arbitrage_consistency(
        option_prices=option_prices,
        strikes=strikes,
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        time_to_maturity=time_to_maturity
    )
    
    print("Arbitrage Consistency Tests:")
    consistent_count = 0
    for result in arbitrage_results:
        if result.consistent:
            consistent_count += 1
        else:
            print(f"  ❌ {result.test_name}: Discrepancy = {result.discrepancy_measure:.6f}")
            
    print(f"  ✅ {consistent_count}/{len(arbitrage_results)} tests passed")
    
    # Test Convergence Analyzer
    print("\nTesting MCMC Convergence Analyzer:")
    
    # Generate some sample MCMC chains
    np.random.seed(42)
    n_samples = 2000
    
    # Chain 1: Well-mixed chain
    chain1 = np.random.normal(0, 1, n_samples)
    for i in range(1, n_samples):
        chain1[i] = 0.9 * chain1[i-1] + 0.1 * np.random.normal(0, 1)
    
    # Chain 2: Poorly mixed chain
    chain2 = np.random.normal(0, 1, n_samples)
    for i in range(1, n_samples):
        chain2[i] = 0.99 * chain2[i-1] + 0.01 * np.random.normal(0, 1)
    
    convergence_analyzer = ConvergenceAnalyzer()
    mcmc_results = convergence_analyzer.analyze_mcmc_convergence(
        chains=[chain1, chain2],
        parameter_names=['well_mixed', 'poorly_mixed']
    )
    
    print("MCMC Convergence Analysis:")
    for param_name, diagnostics in mcmc_results.items():
        print(f"\n{param_name.upper()}:")
        print(f"  Geweke Statistic: {diagnostics['geweke_statistic']:.3f}")
        print(f"  Effective Sample Size: {diagnostics['effective_sample_size']:.1f}")
        print(f"  Autocorrelation Time: {diagnostics['autocorrelation_time']:.1f}")
        print(f"  Running Mean Stable: {diagnostics['running_mean_stable']}")
        
        # Rough convergence assessment
        geweke_ok = abs(diagnostics['geweke_statistic']) < 2  # |z| < 2
        eff_size_ok = diagnostics['effective_sample_size'] > 100
        autocorr_ok = diagnostics['autocorrelation_time'] < 50
        
        overall_ok = geweke_ok and eff_size_ok and autocorr_ok and diagnostics['running_mean_stable']
        print(f"  Overall Assessment: {'✅ Good' if overall_ok else '❌ Poor'}")
    
    print("\nModel verification framework test completed!")
