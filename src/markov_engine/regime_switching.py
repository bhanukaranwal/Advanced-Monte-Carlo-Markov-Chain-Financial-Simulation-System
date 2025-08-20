"""
Regime-switching Markov models for financial time series with structural breaks
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)

@dataclass
class RegimeParameters:
    """Parameters for a single regime"""
    regime_id: int
    mean: float
    variance: float
    persistence: float = 0.95  # Probability of staying in regime
    
@dataclass
class RegimeSwitchingResult:
    """Result of regime switching model estimation"""
    regimes: Dict[int, RegimeParameters]
    transition_matrix: np.ndarray
    regime_probabilities: np.ndarray  # Filtered probabilities
    smoothed_probabilities: np.ndarray  # Smoothed probabilities
    log_likelihood: float
    aic: float
    bic: float
    most_likely_path: List[int]  # Viterbi path

class RegimeSwitchingMarkov:
    """Regime-switching Markov model with multiple estimation methods"""
    
    def __init__(
        self,
        n_regimes: int = 2,
        estimation_method: str = 'em',  # 'em', 'mle', 'bayesian'
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        initialization: str = 'kmeans'  # 'kmeans', 'quantile', 'random'
    ):
        self.n_regimes = n_regimes
        self.estimation_method = estimation_method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.initialization = initialization
        
        # Model parameters
        self.regimes: Dict[int, RegimeParameters] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.initial_distribution: Optional[np.ndarray] = None
        
        # Results storage
        self.regime_probabilities: Optional[np.ndarray] = None
        self.smoothed_probabilities: Optional[np.ndarray] = None
        self.most_likely_path: Optional[List[int]] = None
        self.log_likelihood: Optional[float] = None
        
        # Data storage
        self.data: Optional[np.ndarray] = None
        self.fitted = False
        
    def fit(self, data: Union[pd.Series, np.ndarray]) -> 'RegimeSwitchingMarkov':
        """
        Fit regime-switching model to data
        
        Args:
            data: Time series data
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) < self.n_regimes * 10:
            raise ValueError(f"Insufficient data: need at least {self.n_regimes * 10} observations")
            
        self.data = data
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Estimate model
        if self.estimation_method == 'em':
            self._fit_em_algorithm()
        elif self.estimation_method == 'mle':
            self._fit_mle()
        elif self.estimation_method == 'bayesian':
            self._fit_bayesian()
        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")
            
        # Compute regime probabilities and most likely path
        self.regime_probabilities = self._forward_backward_algorithm()[0]
        self.smoothed_probabilities = self._forward_backward_algorithm()[1]
        self.most_likely_path = self._viterbi_algorithm()
        
        self.fitted = True
        logger.info(f"Regime-switching model fitted with {self.n_regimes} regimes")
        
        return self
        
    def _initialize_parameters(self):
        """Initialize model parameters"""
        if self.initialization == 'kmeans':
            self._initialize_kmeans()
        elif self.initialization == 'quantile':
            self._initialize_quantile()
        elif self.initialization == 'random':
            self._initialize_random()
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")
            
        # Initialize transition matrix with high persistence
        self.transition_matrix = np.eye(self.n_regimes) * 0.9
        self.transition_matrix += (1 - 0.9) / (self.n_regimes - 1) * (1 - np.eye(self.n_regimes))
        
        # Equal initial distribution
        self.initial_distribution = np.ones(self.n_regimes) / self.n_regimes
        
    def _initialize_kmeans(self):
        """Initialize using K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape data for clustering
            X = self.data.reshape(-1, 1)
            
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            self.regimes = {}
            for i in range(self.n_regimes):
                regime_data = self.data[labels == i]
                
                if len(regime_data) > 0:
                    mean = np.mean(regime_data)
                    variance = np.var(regime_data) if len(regime_data) > 1 else 0.01
                else:
                    mean = kmeans.cluster_centers_[i, 0]
                    variance = 0.01
                    
                self.regimes[i] = RegimeParameters(
                    regime_id=i,
                    mean=mean,
                    variance=max(variance, 1e-6)  # Avoid zero variance
                )
                
        except Exception as e:
            logger.warning(f"K-means initialization failed: {e}, using quantile method")
            self._initialize_quantile()
            
    def _initialize_quantile(self):
        """Initialize using quantiles"""
        quantiles = np.linspace(0.1, 0.9, self.n_regimes)
        thresholds = np.quantile(self.data, quantiles)
        
        self.regimes = {}
        for i in range(self.n_regimes):
            # Define regime based on quantile ranges
            if i == 0:
                mask = self.data <= thresholds[0]
            elif i == self.n_regimes - 1:
                mask = self.data > thresholds[-1]
            else:
                mask = (self.data > thresholds[i-1]) & (self.data <= thresholds[i])
                
            regime_data = self.data[mask]
            
            if len(regime_data) > 0:
                mean = np.mean(regime_data)
                variance = np.var(regime_data) if len(regime_data) > 1 else np.var(self.data) / 4
            else:
                mean = thresholds[min(i, len(thresholds) - 1)]
                variance = np.var(self.data) / 4
                
            self.regimes[i] = RegimeParameters(
                regime_id=i,
                mean=mean,
                variance=max(variance, 1e-6)
            )
            
    def _initialize_random(self):
        """Random initialization"""
        data_mean = np.mean(self.data)
        data_var = np.var(self.data)
        
        self.regimes = {}
        for i in range(self.n_regimes):
            # Random means around data mean
            mean = data_mean + np.random.normal(0, np.sqrt(data_var))
            variance = data_var * np.random.uniform(0.5, 2.0)
            
            self.regimes[i] = RegimeParameters(
                regime_id=i,
                mean=mean,
                variance=max(variance, 1e-6)
            )
            
    def _fit_em_algorithm(self):
        """Fit using Expectation-Maximization algorithm"""
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iterations):
            # E-step: compute regime probabilities
            gamma, xi = self._e_step()
            
            # M-step: update parameters
            self._m_step(gamma, xi)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood()
            
            # Check convergence
            if abs(log_likelihood - log_likelihood_old) < self.tolerance:
                logger.info(f"EM converged after {iteration + 1} iterations")
                break
                
            log_likelihood_old = log_likelihood
            
        self.log_likelihood = log_likelihood
        
    def _e_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """E-step of EM algorithm"""
        T = len(self.data)
        
        # Forward-backward algorithm for gamma (regime probabilities)
        alpha = self._forward_algorithm()
        beta = self._backward_algorithm()
        
        # Compute gamma (smoothed probabilities)
        log_gamma = alpha + beta
        gamma = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        
        # Compute xi (joint probabilities of consecutive states)
        xi = np.zeros((T - 1, self.n_regimes, self.n_regimes))
        
        for t in range(T - 1):
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    log_xi_t_ij = (alpha[t, i] + 
                                  np.log(self.transition_matrix[i, j]) +
                                  self._emission_log_prob(self.data[t + 1], j) +
                                  beta[t + 1, j])
                    xi[t, i, j] = log_xi_t_ij
                    
            # Normalize
            xi[t] = np.exp(xi[t] - logsumexp(xi[t]))
            
        return gamma, xi
        
    def _m_step(self, gamma: np.ndarray, xi: np.ndarray):
        """M-step of EM algorithm"""
        T = len(self.data)
        
        # Update initial distribution
        self.initial_distribution = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum(xi[:, i, j])
                denominator = np.sum(gamma[:-1, i])
                
                if denominator > 0:
                    self.transition_matrix[i, j] = numerator / denominator
                else:
                    self.transition_matrix[i, j] = 1.0 / self.n_regimes
                    
        # Ensure transition matrix rows sum to 1
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Update regime parameters
        for i in range(self.n_regimes):
            gamma_sum = np.sum(gamma[:, i])
            
            if gamma_sum > 0:
                # Update mean
                new_mean = np.sum(gamma[:, i] * self.data) / gamma_sum
                
                # Update variance
                new_variance = np.sum(gamma[:, i] * (self.data - new_mean) ** 2) / gamma_sum
                new_variance = max(new_variance, 1e-6)  # Avoid zero variance
                
                self.regimes[i] = RegimeParameters(
                    regime_id=i,
                    mean=new_mean,
                    variance=new_variance,
                    persistence=self.transition_matrix[i, i]
                )
                
    def _forward_algorithm(self) -> np.ndarray:
        """Forward algorithm for HMM"""
        T = len(self.data)
        log_alpha = np.zeros((T, self.n_regimes))
        
        # Initialize
        for i in range(self.n_regimes):
            log_alpha[0, i] = (np.log(self.initial_distribution[i]) + 
                              self._emission_log_prob(self.data[0], i))
            
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_regimes):
                log_alpha[t, j] = (logsumexp(log_alpha[t-1] + np.log(self.transition_matrix[:, j])) +
                                  self._emission_log_prob(self.data[t], j))
                                  
        return log_alpha
        
    def _backward_algorithm(self) -> np.ndarray:
        """Backward algorithm for HMM"""
        T = len(self.data)
        log_beta = np.zeros((T, self.n_regimes))
        
        # Initialize (log_beta[T-1] = 0 for all states)
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            for i in range(self.n_regimes):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_matrix[i, :]) +
                    np.array([self._emission_log_prob(self.data[t + 1], j) for j in range(self.n_regimes)]) +
                    log_beta[t + 1, :]
                )
                
        return log_beta
        
    def _emission_log_prob(self, observation: float, regime: int) -> float:
        """Log probability of observation given regime"""
        regime_params = self.regimes[regime]
        
        # Gaussian emission probability
        log_prob = -0.5 * np.log(2 * np.pi * regime_params.variance)
        log_prob -= 0.5 * ((observation - regime_params.mean) ** 2) / regime_params.variance
        
        return log_prob
        
    def _compute_log_likelihood(self) -> float:
        """Compute log-likelihood of current parameters"""
        log_alpha = self._forward_algorithm()
        return logsumexp(log_alpha[-1])
        
    def _forward_backward_algorithm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Complete forward-backward algorithm"""
        log_alpha = self._forward_algorithm()
        log_beta = self._backward_algorithm()
        
        # Filtered probabilities (forward only)
        filtered_probs = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
        
        # Smoothed probabilities (forward-backward)
        log_gamma = log_alpha + log_beta
        smoothed_probs = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True))
        
        return filtered_probs, smoothed_probs
        
    def _viterbi_algorithm(self) -> List[int]:
        """Viterbi algorithm to find most likely state sequence"""
        T = len(self.data)
        
        # Initialize
        log_delta = np.zeros((T, self.n_regimes))
        psi = np.zeros((T, self.n_regimes), dtype=int)
        
        for i in range(self.n_regimes):
            log_delta[0, i] = (np.log(self.initial_distribution[i]) + 
                              self._emission_log_prob(self.data[0], i))
            
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_regimes):
                candidates = log_delta[t-1] + np.log(self.transition_matrix[:, j])
                psi[t, j] = np.argmax(candidates)
                log_delta[t, j] = np.max(candidates) + self._emission_log_prob(self.data[t], j)
                
        # Backward pass
        path = [0] * T
        path[T-1] = np.argmax(log_delta[T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
            
        return path
        
    def _fit_mle(self):
        """Fit using Maximum Likelihood Estimation"""
        # Define objective function
        def objective(params):
            self._unpack_parameters(params)
            return -self._compute_log_likelihood()
            
        # Initial parameters
        initial_params = self._pack_parameters()
        
        # Optimize
        result = minimize(
            objective, 
            initial_params, 
            method='L-BFGS-B',
            options={'maxiter': self.max_iterations}
        )
        
        if result.success:
            self._unpack_parameters(result.x)
            self.log_likelihood = -result.fun
            logger.info("MLE optimization converged")
        else:
            logger.warning("MLE optimization did not converge")
            
    def _fit_bayesian(self):
        """Fit using Bayesian estimation (simplified MCMC)"""
        # This is a placeholder for a full Bayesian implementation
        # Would require proper MCMC sampling (e.g., using PyMC3 or Stan)
        logger.warning("Bayesian estimation not fully implemented, falling back to EM")
        self._fit_em_algorithm()
        
    def _pack_parameters(self) -> np.ndarray:
        """Pack parameters into vector for optimization"""
        params = []
        
        # Regime parameters (means and variances)
        for i in range(self.n_regimes):
            params.extend([self.regimes[i].mean, np.log(self.regimes[i].variance)])
            
        # Transition matrix (log probabilities, excluding last column)
        for i in range(self.n_regimes):
            params.extend(np.log(self.transition_matrix[i, :-1]))
            
        return np.array(params)
        
    def _unpack_parameters(self, params: np.ndarray):
        """Unpack parameters from optimization vector"""
        idx = 0
        
        # Regime parameters
        for i in range(self.n_regimes):
            mean = params[idx]
            log_var = params[idx + 1]
            
            self.regimes[i] = RegimeParameters(
                regime_id=i,
                mean=mean,
                variance=np.exp(log_var)
            )
            idx += 2
            
        # Transition matrix
        for i in range(self.n_regimes):
            log_probs = params[idx:idx + self.n_regimes - 1]
            # Convert from log probabilities and normalize
            probs = np.exp(np.concatenate([log_probs, [0]]))
            self.transition_matrix[i] = probs / np.sum(probs)
            idx += self.n_regimes - 1
            
    def predict_regime(
        self, 
        steps_ahead: int = 1,
        method: str = 'filtered'  # 'filtered' or 'smoothed'
    ) -> np.ndarray:
        """
        Predict regime probabilities steps ahead
        
        Args:
            steps_ahead: Number of steps to predict
            method: Use filtered or smoothed probabilities
            
        Returns:
            Predicted regime probabilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if method == 'filtered':
            current_probs = self.regime_probabilities[-1]
        else:
            current_probs = self.smoothed_probabilities[-1]
            
        # Evolve probabilities forward
        predicted_probs = current_probs.copy()
        for _ in range(steps_ahead):
            predicted_probs = predicted_probs @ self.transition_matrix
            
        return predicted_probs
        
    def simulate_path(
        self, 
        n_steps: int, 
        initial_regime: Optional[int] = None
    ) -> Tuple[List[int], np.ndarray]:
        """
        Simulate regime path and observations
        
        Args:
            n_steps: Number of steps to simulate
            initial_regime: Starting regime (random if None)
            
        Returns:
            Tuple of (regime_path, observations)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
            
        if initial_regime is None:
            initial_regime = np.random.choice(self.n_regimes, p=self.initial_distribution)
            
        regimes = [initial_regime]
        observations = []
        
        current_regime = initial_regime
        
        for _ in range(n_steps):
            # Generate observation from current regime
            regime_params = self.regimes[current_regime]
            obs = np.random.normal(regime_params.mean, np.sqrt(regime_params.variance))
            observations.append(obs)
            
            # Transition to next regime
            next_regime = np.random.choice(
                self.n_regimes, 
                p=self.transition_matrix[current_regime]
            )
            regimes.append(next_regime)
            current_regime = next_regime
            
        return regimes[1:], np.array(observations)  # Remove initial regime
        
    def get_regime_characteristics(self) -> Dict[int, Dict[str, float]]:
        """Get characteristics of each regime"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting characteristics")
            
        characteristics = {}
        
        for i, regime in self.regimes.items():
            # Calculate additional statistics
            regime_mask = np.array(self.most_likely_path) == i
            regime_data = self.data[regime_mask] if np.any(regime_mask) else np.array([])
            
            characteristics[i] = {
                'mean': regime.mean,
                'variance': regime.variance,
                'volatility': np.sqrt(regime.variance),
                'persistence': self.transition_matrix[i, i],
                'expected_duration': 1 / (1 - self.transition_matrix[i, i]) if self.transition_matrix[i, i] < 1 else np.inf,
                'frequency': np.mean(regime_mask) if len(regime_mask) > 0 else 0,
                'n_observations': len(regime_data)
            }
            
        return characteristics
        
    def get_model_summary(self) -> RegimeSwitchingResult:
        """Get comprehensive model results"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting summary")
            
        # Calculate information criteria
        n_params = self.n_regimes * 2 + self.n_regimes * (self.n_regimes - 1)  # means, vars, transition probs
        n_obs = len(self.data)
        
        aic = -2 * self.log_likelihood + 2 * n_params
        bic = -2 * self.log_likelihood + np.log(n_obs) * n_params
        
        return RegimeSwitchingResult(
            regimes=self.regimes.copy(),
            transition_matrix=self.transition_matrix.copy(),
            regime_probabilities=self.regime_probabilities.copy(),
            smoothed_probabilities=self.smoothed_probabilities.copy(),
            log_likelihood=self.log_likelihood,
            aic=aic,
            bic=bic,
            most_likely_path=self.most_likely_path.copy()
        )

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic regime-switching data
    np.random.seed(42)
    
    # Define true regimes
    true_regimes = [
        {'mean': 0.02, 'var': 0.01, 'duration': 50},   # Bull market
        {'mean': -0.03, 'var': 0.04, 'duration': 30}   # Bear market
    ]
    
    # Generate regime-switching data
    n_obs = 500
    true_path = []
    data = []
    
    current_regime = 0
    regime_counter = 0
    
    for t in range(n_obs):
        # Switch regimes based on duration
        if regime_counter >= true_regimes[current_regime]['duration']:
            current_regime = 1 - current_regime  # Switch regime
            regime_counter = 0
            
        true_path.append(current_regime)
        
        # Generate observation
        mean = true_regimes[current_regime]['mean']
        var = true_regimes[current_regime]['var']
        obs = np.random.normal(mean, np.sqrt(var))
        data.append(obs)
        
        regime_counter += 1
        
    data = np.array(data)
    
    print("Testing Regime-Switching Markov Model...")
    print(f"Generated {len(data)} observations with regime switches")
    
    # Test different estimation methods
    for method in ['em', 'mle']:
        print(f"\nTesting with {method} estimation:")
        
        try:
            # Fit model
            rs_model = RegimeSwitchingMarkov(
                n_regimes=2,
                estimation_method=method,
                max_iterations=100
            )
            
            rs_model.fit(data)
            
            # Get results
            results = rs_model.get_model_summary()
            
            print(f"  Log-likelihood: {results.log_likelihood:.2f}")
            print(f"  AIC: {results.aic:.2f}")
            print(f"  BIC: {results.bic:.2f}")
            
            # Show regime characteristics
            characteristics = rs_model.get_regime_characteristics()
            print("  Regime characteristics:")
            for regime_id, chars in characteristics.items():
                print(f"    Regime {regime_id}: mean={chars['mean']:.4f}, "
                      f"vol={chars['volatility']:.4f}, "
                      f"persistence={chars['persistence']:.3f}, "
                      f"duration={chars['expected_duration']:.1f}")
                      
            # Test prediction
            pred_probs = rs_model.predict_regime(steps_ahead=5)
            print(f"  5-step regime probabilities: {pred_probs}")
            
            # Test simulation
            sim_regimes, sim_data = rs_model.simulate_path(20)
            print(f"  Simulated regime path: {sim_regimes[:10]}...")
            
        except Exception as e:
            print(f"  Failed with error: {e}")
    
    print("\nRegime-switching Markov model test completed!")
