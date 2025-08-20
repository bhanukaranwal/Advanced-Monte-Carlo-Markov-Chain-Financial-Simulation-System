"""
Stochastic processes for financial modeling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from scipy.stats import norm, poisson
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProcessParameters:
    """Parameters for stochastic processes"""
    initial_value: float
    drift: float = 0.05
    volatility: float = 0.2
    time_horizon: float = 1.0
    n_steps: int = 252

class StochasticProcess(ABC):
    """Abstract base class for stochastic processes"""
    
    @abstractmethod
    def simulate_path(self, params: ProcessParameters, n_paths: int = 1) -> np.ndarray:
        """Simulate process paths"""
        pass
        
    @abstractmethod
    def get_moments(self, params: ProcessParameters, time: float) -> Dict[str, float]:
        """Get theoretical moments at given time"""
        pass

class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion: dS = μS dt + σS dW"""
    
    def simulate_path(self, params: ProcessParameters, n_paths: int = 1) -> np.ndarray:
        """
        Simulate GBM paths using exact solution
        
        S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
        """
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Generate random increments
        dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        
        # Calculate log price increments
        drift_term = (params.drift - 0.5 * params.volatility**2) * dt
        diffusion_term = params.volatility * dW
        
        log_increments = drift_term + diffusion_term
        
        # Construct paths
        log_prices = np.zeros((n_paths, params.n_steps + 1))
        log_prices[:, 0] = np.log(params.initial_value)
        log_prices[:, 1:] = np.cumsum(log_increments, axis=1)
        
        prices = np.exp(log_prices)
        
        return prices
        
    def get_moments(self, params: ProcessParameters, time: float) -> Dict[str, float]:
        """Get theoretical moments for GBM"""
        S0 = params.initial_value
        mu = params.drift
        sigma = params.volatility
        t = time
        
        # E[S(t)] = S(0) * exp(μt)
        mean = S0 * np.exp(mu * t)
        
        # Var[S(t)] = S(0)² * exp(2μt) * (exp(σ²t) - 1)
        variance = S0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }

@dataclass 
class HestonParameters(ProcessParameters):
    """Extended parameters for Heston model"""
    theta: float = 0.04      # Long-term variance
    kappa: float = 2.0       # Mean reversion speed
    sigma_v: float = 0.3     # Volatility of volatility
    rho: float = -0.7        # Correlation between price and variance
    v0: float = 0.04         # Initial variance

class HestonModel(StochasticProcess):
    """
    Heston stochastic volatility model:
    dS = μS dt + √V S dW1
    dV = κ(θ - V) dt + σ_v √V dW2
    where dW1 dW2 = ρ dt
    """
    
    def simulate_path(self, params: HestonParameters, n_paths: int = 1) -> np.ndarray:
        """Simulate Heston model paths using Euler-Maruyama scheme"""
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays
        S = np.zeros((n_paths, params.n_steps + 1))
        V = np.zeros((n_paths, params.n_steps + 1))
        
        S[:, 0] = params.initial_value
        V[:, 0] = params.v0
        
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        dW2_independent = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        
        # Apply correlation: dW2 = ρ dW1 + √(1-ρ²) dW2_independent
        dW2 = params.rho * dW1 + np.sqrt(1 - params.rho**2) * dW2_independent
        
        # Simulate paths
        for t in range(params.n_steps):
            # Ensure non-negative variance (Feller condition)
            V_current = np.maximum(V[:, t], 0)
            sqrt_V = np.sqrt(V_current)
            
            # Variance process
            dV = (params.kappa * (params.theta - V_current) * dt + 
                  params.sigma_v * sqrt_V * dW2[:, t])
            V[:, t + 1] = V_current + dV
            
            # Price process
            dS = (params.drift * S[:, t] * dt + 
                  sqrt_V * S[:, t] * dW1[:, t])
            S[:, t + 1] = S[:, t] + dS
            
        return S
        
    def get_moments(self, params: HestonParameters, time: float) -> Dict[str, float]:
        """Get theoretical moments for Heston model (approximations)"""
        # These are approximations - exact moments are complex
        S0 = params.initial_value
        mu = params.drift
        v0 = params.v0
        theta = params.theta
        kappa = params.kappa
        t = time
        
        # Mean is approximately same as GBM with average volatility
        avg_vol = np.sqrt((v0 + theta) / 2)
        mean = S0 * np.exp(mu * t)
        
        # Variance approximation
        variance = S0**2 * np.exp(2 * mu * t) * (np.exp(avg_vol**2 * t) - 1)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }

@dataclass
class JumpDiffusionParameters(ProcessParameters):
    """Parameters for jump diffusion model"""
    lambda_jump: float = 0.1      # Jump intensity (jumps per unit time)
    mu_jump: float = -0.05        # Mean jump size
    sigma_jump: float = 0.15      # Jump size volatility

class JumpDiffusionModel(StochasticProcess):
    """
    Merton jump diffusion model:
    dS = μS dt + σS dW + S(e^J - 1) dN
    where dN is Poisson process and J ~ N(μ_J, σ_J²)
    """
    
    def simulate_path(self, params: JumpDiffusionParameters, n_paths: int = 1) -> np.ndarray:
        """Simulate jump diffusion paths"""
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize price array
        S = np.zeros((n_paths, params.n_steps + 1))
        S[:, 0] = params.initial_value
        
        # Generate Brownian motion increments
        dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        
        # Generate jump times and sizes
        for path in range(n_paths):
            current_price = params.initial_value
            
            for t in range(params.n_steps):
                # Continuous part (GBM)
                drift_term = (params.drift - 0.5 * params.volatility**2) * dt
                diffusion_term = params.volatility * dW[path, t]
                
                # Jump part
                n_jumps = poisson.rvs(params.lambda_jump * dt)
                jump_contribution = 0.0
                
                if n_jumps > 0:
                    # Generate jump sizes
                    jump_sizes = np.random.normal(
                        params.mu_jump, params.sigma_jump, n_jumps
                    )
                    jump_contribution = np.sum(jump_sizes)
                
                # Update price
                log_return = drift_term + diffusion_term + jump_contribution
                current_price = current_price * np.exp(log_return)
                S[path, t + 1] = current_price
                
        return S
        
    def get_moments(self, params: JumpDiffusionParameters, time: float) -> Dict[str, float]:
        """Get theoretical moments for jump diffusion model"""
        S0 = params.initial_value
        mu = params.drift
        sigma = params.volatility
        lam = params.lambda_jump
        mu_J = params.mu_jump
        sigma_J = params.sigma_jump
        t = time
        
        # Expected value
        k = np.exp(mu_J + 0.5 * sigma_J**2) - 1  # Expected relative jump size
        adjusted_drift = mu + lam * k
        mean = S0 * np.exp(adjusted_drift * t)
        
        # Variance (approximate)
        jump_var = lam * t * (np.exp(2*mu_J + sigma_J**2) * (np.exp(sigma_J**2) - 1) + 
                             (np.exp(mu_J + 0.5*sigma_J**2) - 1)**2)
        diffusion_var = S0**2 * np.exp(2*adjusted_drift*t) * (np.exp(sigma**2 * t) - 1)
        variance = diffusion_var + S0**2 * jump_var
        
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }

class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Ornstein-Uhlenbeck process (mean-reverting):
    dX = θ(μ - X) dt + σ dW
    """
    
    def __init__(self, theta: float = 1.0, long_term_mean: float = 0.0):
        self.theta = theta           # Mean reversion speed
        self.long_term_mean = long_term_mean
        
    def simulate_path(self, params: ProcessParameters, n_paths: int = 1) -> np.ndarray:
        """Simulate O-U process paths using exact solution"""
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize array
        X = np.zeros((n_paths, params.n_steps + 1))
        X[:, 0] = params.initial_value
        
        # Generate increments using exact solution
        exp_theta_dt = np.exp(-self.theta * dt)
        variance_factor = params.volatility**2 * (1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta)
        
        for t in range(params.n_steps):
            # Exact solution increment
            mean_increment = (X[:, t] - self.long_term_mean) * exp_theta_dt + self.long_term_mean
            noise = np.random.normal(0, np.sqrt(variance_factor), n_paths)
            
            X[:, t + 1] = mean_increment + noise
            
        return X
        
    def get_moments(self, params: ProcessParameters, time: float) -> Dict[str, float]:
        """Get theoretical moments for O-U process"""
        X0 = params.initial_value
        theta = self.theta
        mu = self.long_term_mean
        sigma = params.volatility
        t = time
        
        # E[X(t)] = μ + (X0 - μ)e^(-θt)
        mean = mu + (X0 - mu) * np.exp(-theta * t)
        
        # Var[X(t)] = σ²(1 - e^(-2θt))/(2θ)
        if theta > 0:
            variance = sigma**2 * (1 - np.exp(-2 * theta * t)) / (2 * theta)
        else:
            variance = sigma**2 * t  # Brownian motion case
            
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }

class CIRProcess(StochasticProcess):
    """
    Cox-Ingersoll-Ross process (always non-negative):
    dX = κ(θ - X) dt + σ√X dW
    """
    
    def __init__(self, kappa: float = 2.0, theta: float = 0.04):
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-term level
        
    def simulate_path(self, params: ProcessParameters, n_paths: int = 1) -> np.ndarray:
        """Simulate CIR process using Euler scheme with reflection"""
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize array
        X = np.zeros((n_paths, params.n_steps + 1))
        X[:, 0] = params.initial_value
        
        # Generate random increments
        dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        
        for t in range(params.n_steps):
            X_current = np.maximum(X[:, t], 0)  # Ensure non-negative
            sqrt_X = np.sqrt(X_current)
            
            # CIR dynamics
            drift = self.kappa * (self.theta - X_current) * dt
            diffusion = params.volatility * sqrt_X * dW[:, t]
            
            X[:, t + 1] = X_current + drift + diffusion
            
            # Reflection at zero (simple approach)
            X[:, t + 1] = np.maximum(X[:, t + 1], 0)
            
        return X
        
    def get_moments(self, params: ProcessParameters, time: float) -> Dict[str, float]:
        """Get theoretical moments for CIR process"""
        X0 = params.initial_value
        kappa = self.kappa
        theta = self.theta
        sigma = params.volatility
        t = time
        
        # E[X(t)] = θ + (X0 - θ)e^(-κt)
        mean = theta + (X0 - theta) * np.exp(-kappa * t)
        
        # Var[X(t)] = X0(σ²/κ)(e^(-κt) - e^(-2κt)) + θ(σ²/2κ)(1 - e^(-κt))²
        if kappa > 0:
            term1 = X0 * (sigma**2 / kappa) * (np.exp(-kappa * t) - np.exp(-2 * kappa * t))
            term2 = theta * (sigma**2 / (2 * kappa)) * (1 - np.exp(-kappa * t))**2
            variance = term1 + term2
        else:
            variance = sigma**2 * X0 * t  # Degenerate case
            
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }

class StochasticProcesses:
    """Main orchestrator for stochastic processes"""
    
    def __init__(self):
        self.processes = {
            'gbm': GeometricBrownianMotion(),
            'heston': HestonModel(),
            'jump_diffusion': JumpDiffusionModel(),
            'ornstein_uhlenbeck': OrnsteinUhlenbeckProcess(),
            'cir': CIRProcess()
        }
        
    def simulate(
        self,
        process_type: str,
        params: Union[ProcessParameters, Dict],
        n_paths: int = 1000
    ) -> np.ndarray:
        """
        Simulate paths for specified process
        
        Args:
            process_type: Type of process to simulate
            params: Process parameters
            n_paths: Number of paths
            
        Returns:
            Simulated paths
        """
        if process_type not in self.processes:
            raise ValueError(f"Unknown process type: {process_type}")
            
        process = self.processes[process_type]
        
        # Convert dict to appropriate parameter class if needed
        if isinstance(params, dict):
            if process_type == 'heston':
                params = HestonParameters(**params)
            elif process_type == 'jump_diffusion':
                params = JumpDiffusionParameters(**params)
            else:
                params = ProcessParameters(**params)
                
        return process.simulate_path(params, n_paths)
        
    def get_process_info(self, process_type: str, params: Dict, time: float) -> Dict:
        """Get theoretical information about process"""
        if process_type not in self.processes:
            raise ValueError(f"Unknown process type: {process_type}")
            
        process = self.processes[process_type]
        
        # Convert params to appropriate type
        if process_type == 'heston':
            param_obj = HestonParameters(**params)
        elif process_type == 'jump_diffusion':
            param_obj = JumpDiffusionParameters(**params)
        else:
            param_obj = ProcessParameters(**params)
            
        moments = process.get_moments(param_obj, time)
        
        return {
            'process_type': process_type,
            'theoretical_moments': moments,
            'parameters': params
        }
        
    def compare_processes(
        self,
        process_types: List[str],
        base_params: Dict,
        n_paths: int = 1000
    ) -> Dict[str, Any]:
        """Compare multiple stochastic processes"""
        results = {}
        
        for process_type in process_types:
            try:
                # Adjust parameters for specific processes
                if process_type == 'heston' and 'theta' not in base_params:
                    params = {**base_params, 'theta': 0.04, 'kappa': 2.0, 'sigma_v': 0.3, 'rho': -0.7, 'v0': 0.04}
                elif process_type == 'jump_diffusion' and 'lambda_jump' not in base_params:
                    params = {**base_params, 'lambda_jump': 0.1, 'mu_jump': -0.05, 'sigma_jump': 0.15}
                else:
                    params = base_params
                    
                # Simulate paths
                paths = self.simulate(process_type, params, n_paths)
                
                # Calculate empirical statistics
                final_values = paths[:, -1]
                
                results[process_type] = {
                    'empirical_mean': np.mean(final_values),
                    'empirical_std': np.std(final_values),
                    'empirical_skew': self._calculate_skewness(final_values),
                    'empirical_kurtosis': self._calculate_kurtosis(final_values),
                    'theoretical_info': self.get_process_info(process_type, params, params.get('time_horizon', 1.0))
                }
                
            except Exception as e:
                logger.error(f"Error comparing process {process_type}: {e}")
                results[process_type] = {'error': str(e)}
                
        return results
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate sample kurtosis (excess)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

# Example usage and testing
if __name__ == "__main__":
    print("Testing Stochastic Processes...")
    
    # Base parameters
    base_params = {
        'initial_value': 100.0,
        'drift': 0.05,
        'volatility': 0.2,
        'time_horizon': 1.0,
        'n_steps': 252
    }
    
    # Initialize processes
    sp = StochasticProcesses()
    
    # Test individual processes
    processes_to_test = ['gbm', 'heston', 'jump_diffusion', 'ornstein_uhlenbeck', 'cir']
    
    for process_type in processes_to_test:
        print(f"\nTesting {process_type.upper()}:")
        
        try:
            # Simulate paths
            paths = sp.simulate(process_type, base_params, n_paths=100)
            print(f"  Generated {paths.shape[0]} paths with {paths.shape[1]} time steps")
            
            # Basic statistics
            final_values = paths[:, -1]
            print(f"  Final value mean: {np.mean(final_values):.3f}")
            print(f"  Final value std: {np.std(final_values):.3f}")
            
            # Theoretical info
            info = sp.get_process_info(process_type, base_params, 1.0)
            theoretical = info['theoretical_moments']
            print(f"  Theoretical mean: {theoretical['mean']:.3f}")
            print(f"  Theoretical std: {theoretical['std']:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Compare processes
    print("\nComparing Processes:")
    comparison = sp.compare_processes(['gbm', 'heston', 'jump_diffusion'], base_params, n_paths=1000)
    
    print(f"{'Process':<15} {'Emp. Mean':<10} {'Emp. Std':<10} {'Skewness':<10} {'Kurtosis':<10}")
    print("-" * 60)
    
    for process, results in comparison.items():
        if 'error' not in results:
            print(f"{process:<15} {results['empirical_mean']:<10.3f} {results['empirical_std']:<10.3f} "
                  f"{results['empirical_skew']:<10.3f} {results['empirical_kurtosis']:<10.3f}")
        else:
            print(f"{process:<15} ERROR: {results['error']}")
    
    print("\nStochastic processes test completed!")
