"""
Advanced path generation algorithms for stochastic differential equations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SDEParameters:
    """Parameters for SDE path generation"""
    initial_value: Union[float, np.ndarray]
    time_horizon: float = 1.0
    n_steps: int = 252
    drift_func: Optional[Callable] = None      # μ(t, X)
    diffusion_func: Optional[Callable] = None  # σ(t, X)
    jump_measure: Optional[Callable] = None    # For jump processes

class PathGenerationScheme(ABC):
    """Abstract base class for path generation schemes"""
    
    @abstractmethod
    def generate_paths(
        self,
        params: SDEParameters,
        n_paths: int,
        **kwargs
    ) -> np.ndarray:
        """Generate sample paths"""
        pass

class EulerMaruyamaGenerator(PathGenerationScheme):
    """Euler-Maruyama scheme for SDE path generation"""
    
    def generate_paths(
        self,
        params: SDEParameters,
        n_paths: int,
        noise_type: str = 'gaussian',
        **kwargs
    ) -> np.ndarray:
        """
        Generate paths using Euler-Maruyama scheme
        
        X_{t+dt} = X_t + μ(t, X_t)dt + σ(t, X_t)dW_t
        
        Args:
            params: SDE parameters
            n_paths: Number of paths to generate
            noise_type: Type of noise ('gaussian', 'levy')
            
        Returns:
            Array of shape (n_paths, n_steps+1, n_dim)
        """
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Determine dimensionality
        if isinstance(params.initial_value, (int, float)):
            n_dim = 1
            initial_value = np.full(n_paths, params.initial_value)
        else:
            n_dim = len(params.initial_value)
            initial_value = np.tile(params.initial_value, (n_paths, 1))
            
        # Initialize paths array
        if n_dim == 1:
            paths = np.zeros((n_paths, params.n_steps + 1))
            paths[:, 0] = initial_value
        else:
            paths = np.zeros((n_paths, params.n_steps + 1, n_dim))
            paths[:, 0, :] = initial_value
            
        # Time grid
        times = np.linspace(0, params.time_horizon, params.n_steps + 1)
        
        # Generate noise
        if noise_type == 'gaussian':
            if n_dim == 1:
                noise = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
            else:
                noise = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps, n_dim))
        elif noise_type == 'levy':
            # Simplified Lévy noise (would need more sophisticated implementation)
            noise = self._generate_levy_noise(n_paths, params.n_steps, n_dim, dt)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
            
        # Evolve paths
        for t in range(params.n_steps):
            current_time = times[t]
            
            if n_dim == 1:
                current_value = paths[:, t]
                
                # Drift term
                if params.drift_func is not None:
                    drift = params.drift_func(current_time, current_value) * dt
                else:
                    drift = 0.0
                    
                # Diffusion term
                if params.diffusion_func is not None:
                    diffusion = params.diffusion_func(current_time, current_value) * noise[:, t]
                else:
                    diffusion = noise[:, t]
                    
                paths[:, t + 1] = current_value + drift + diffusion
                
            else:
                current_value = paths[:, t, :]
                
                # Drift term
                if params.drift_func is not None:
                    drift = np.array([
                        params.drift_func(current_time, current_value[i, :])
                        for i in range(n_paths)
                    ]) * dt
                else:
                    drift = np.zeros((n_paths, n_dim))
                    
                # Diffusion term
                if params.diffusion_func is not None:
                    diffusion = np.array([
                        params.diffusion_func(current_time, current_value[i, :]) @ noise[i, t, :]
                        for i in range(n_paths)
                    ])
                else:
                    diffusion = noise[:, t, :]
                    
                paths[:, t + 1, :] = current_value + drift + diffusion
                
            # Add jumps if specified
            if params.jump_measure is not None:
                jump_contribution = self._generate_jumps(
                    current_time, current_value, dt, params.jump_measure
                )
                if n_dim == 1:
                    paths[:, t + 1] += jump_contribution
                else:
                    paths[:, t + 1, :] += jump_contribution
                    
        return paths
        
    def _generate_levy_noise(self, n_paths: int, n_steps: int, n_dim: int, dt: float) -> np.ndarray:
        """Generate Lévy process increments (simplified)"""
        # This is a placeholder - full Lévy process simulation would be more complex
        alpha = 1.8  # Stability parameter
        beta = 0.0   # Skewness parameter
        
        if n_dim == 1:
            # Approximate using mixture of normals
            noise = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            # Add some heavy-tailed component
            heavy_tail = np.random.laplace(0, dt**(1/alpha), (n_paths, n_steps)) * 0.1
            return noise + heavy_tail
        else:
            noise = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps, n_dim))
            heavy_tail = np.random.laplace(0, dt**(1/alpha), (n_paths, n_steps, n_dim)) * 0.1
            return noise + heavy_tail
            
    def _generate_jumps(
        self,
        time: float,
        current_value: np.ndarray,
        dt: float,
        jump_measure: Callable
    ) -> np.ndarray:
        """Generate jump contributions"""
        return jump_measure(time, current_value, dt)

class MilsteinGenerator(PathGenerationScheme):
    """Milstein scheme for improved accuracy"""
    
    def generate_paths(
        self,
        params: SDEParameters,
        n_paths: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate paths using Milstein scheme
        
        X_{t+dt} = X_t + μ(t,X_t)dt + σ(t,X_t)dW_t + 0.5*σ(t,X_t)*σ'(t,X_t)*(dW_t² - dt)
        
        Note: Requires derivative of diffusion function
        """
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Check if we have required derivatives
        diffusion_derivative = kwargs.get('diffusion_derivative')
        if diffusion_derivative is None:
            logger.warning("Milstein scheme requires diffusion derivative, falling back to Euler-Maruyama")
            euler_gen = EulerMaruyamaGenerator()
            return euler_gen.generate_paths(params, n_paths, **kwargs)
            
        # Determine dimensionality
        if isinstance(params.initial_value, (int, float)):
            n_dim = 1
            initial_value = np.full(n_paths, params.initial_value)
        else:
            n_dim = len(params.initial_value)
            initial_value = np.tile(params.initial_value, (n_paths, 1))
            
        # Initialize paths
        if n_dim == 1:
            paths = np.zeros((n_paths, params.n_steps + 1))
            paths[:, 0] = initial_value
        else:
            paths = np.zeros((n_paths, params.n_steps + 1, n_dim))
            paths[:, 0, :] = initial_value
            
        # Time grid
        times = np.linspace(0, params.time_horizon, params.n_steps + 1)
        
        # Generate Brownian increments
        if n_dim == 1:
            dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        else:
            dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps, n_dim))
            
        # Milstein correction terms
        if n_dim == 1:
            dW_squared = dW**2 - dt
        else:
            dW_squared = dW**2 - dt
            
        # Evolve paths
        for t in range(params.n_steps):
            current_time = times[t]
            
            if n_dim == 1:
                current_value = paths[:, t]
                
                # Drift term
                drift = params.drift_func(current_time, current_value) * dt if params.drift_func else 0.0
                
                # Diffusion term
                sigma = params.diffusion_func(current_time, current_value) if params.diffusion_func else 1.0
                diffusion = sigma * dW[:, t]
                
                # Milstein correction
                sigma_prime = diffusion_derivative(current_time, current_value)
                correction = 0.5 * sigma * sigma_prime * dW_squared[:, t]
                
                paths[:, t + 1] = current_value + drift + diffusion + correction
                
            else:
                # Multi-dimensional case is more complex
                current_value = paths[:, t, :]
                
                for i in range(n_paths):
                    # Drift
                    if params.drift_func:
                        drift = params.drift_func(current_time, current_value[i, :]) * dt
                    else:
                        drift = np.zeros(n_dim)
                        
                    # Diffusion matrix
                    if params.diffusion_func:
                        sigma_matrix = params.diffusion_func(current_time, current_value[i, :])
                        diffusion = sigma_matrix @ dW[i, t, :]
                    else:
                        diffusion = dW[i, t, :]
                        
                    # Simplified Milstein correction for multi-dimensional case
                    # (Full implementation would require all second-order derivatives)
                    correction = np.zeros(n_dim)
                    
                    paths[i, t + 1, :] = current_value[i, :] + drift + diffusion + correction
                    
        return paths

class RungeKuttaGenerator(PathGenerationScheme):
    """Runge-Kutta schemes for SDE simulation"""
    
    def __init__(self, order: int = 2):
        self.order = order
        
    def generate_paths(
        self,
        params: SDEParameters,
        n_paths: int,
        **kwargs
    ) -> np.ndarray:
        """Generate paths using stochastic Runge-Kutta method"""
        if self.order == 2:
            return self._rk2_paths(params, n_paths)
        elif self.order == 4:
            return self._rk4_paths(params, n_paths)
        else:
            raise ValueError(f"Runge-Kutta order {self.order} not implemented")
            
    def _rk2_paths(self, params: SDEParameters, n_paths: int) -> np.ndarray:
        """Second-order stochastic Runge-Kutta"""
        dt = params.time_horizon / params.n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize
        n_dim = 1 if isinstance(params.initial_value, (int, float)) else len(params.initial_value)
        
        if n_dim == 1:
            paths = np.zeros((n_paths, params.n_steps + 1))
            paths[:, 0] = params.initial_value
        else:
            paths = np.zeros((n_paths, params.n_steps + 1, n_dim))
            paths[:, 0, :] = params.initial_value
            
        times = np.linspace(0, params.time_horizon, params.n_steps + 1)
        
        if n_dim == 1:
            dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps))
        else:
            dW = np.random.normal(0, sqrt_dt, (n_paths, params.n_steps, n_dim))
            
        # RK2 scheme
        for t in range(params.n_steps):
            current_time = times[t]
            
            if n_dim == 1:
                X_current = paths[:, t]
                
                # Stage 1
                a1 = params.drift_func(current_time, X_current) * dt if params.drift_func else 0.0
                b1 = params.diffusion_func(current_time, X_current) * dW[:, t] if params.diffusion_func else dW[:, t]
                
                # Stage 2 (predictor)
                X_pred = X_current + a1 + b1
                a2 = params.drift_func(current_time + dt, X_pred) * dt if params.drift_func else 0.0
                b2 = params.diffusion_func(current_time + dt, X_pred) * dW[:, t] if params.diffusion_func else dW[:, t]
                
                # Final update
                paths[:, t + 1] = X_current + 0.5 * (a1 + a2) + 0.5 * (b1 + b2)
                
            else:
                # Multi-dimensional case
                X_current = paths[:, t, :]
                
                for i in range(n_paths):
                    # Stage 1
                    if params.drift_func:
                        a1 = params.drift_func(current_time, X_current[i, :]) * dt
                    else:
                        a1 = np.zeros(n_dim)
                        
                    if params.diffusion_func:
                        sigma = params.diffusion_func(current_time, X_current[i, :])
                        b1 = sigma @ dW[i, t, :] if sigma.ndim > 1 else sigma * dW[i, t, :]
                    else:
                        b1 = dW[i, t, :]
                        
                    # Stage 2
                    X_pred = X_current[i, :] + a1 + b1
                    
                    if params.drift_func:
                        a2 = params.drift_func(current_time + dt, X_pred) * dt
                    else:
                        a2 = np.zeros(n_dim)
                        
                    if params.diffusion_func:
                        sigma = params.diffusion_func(current_time + dt, X_pred)
                        b2 = sigma @ dW[i, t, :] if sigma.ndim > 1 else sigma * dW[i, t, :]
                    else:
                        b2 = dW[i, t, :]
                        
                    # Update
                    paths[i, t + 1, :] = X_current[i, :] + 0.5 * (a1 + a2) + 0.5 * (b1 + b2)
                    
        return paths
        
    def _rk4_paths(self, params: SDEParameters, n_paths: int) -> np.ndarray:
        """Fourth-order stochastic Runge-Kutta (simplified)"""
        # This is a placeholder - true 4th order stochastic RK is complex
        logger.warning("RK4 for SDEs not fully implemented, using RK2")
        return self._rk2_paths(params, n_paths)

class AdaptivePathGenerator(PathGenerationScheme):
    """Adaptive step size path generation"""
    
    def __init__(self, base_generator: PathGenerationScheme, tolerance: float = 1e-3):
        self.base_generator = base_generator
        self.tolerance = tolerance
        
    def generate_paths(
        self,
        params: SDEParameters,
        n_paths: int,
        **kwargs
    ) -> np.ndarray:
        """Generate paths with adaptive step sizing"""
        # Start with given step size
        current_steps = params.n_steps
        
        while True:
            # Generate paths with current step size
            current_params = SDEParameters(
                initial_value=params.initial_value,
                time_horizon=params.time_horizon,
                n_steps=current_steps,
                drift_func=params.drift_func,
                diffusion_func=params.diffusion_func,
                jump_measure=params.jump_measure
            )
            
            paths_coarse = self.base_generator.generate_paths(current_params, n_paths, **kwargs)
            
            # Generate paths with double the steps
            fine_params = SDEParameters(
                initial_value=params.initial_value,
                time_horizon=params.time_horizon,
                n_steps=current_steps * 2,
                drift_func=params.drift_func,
                diffusion_func=params.diffusion_func,
                jump_measure=params.jump_measure
            )
            
            paths_fine = self.base_generator.generate_paths(fine_params, n_paths, **kwargs)
            
            # Compare final values (simplified error estimate)
            final_coarse = paths_coarse[:, -1] if paths_coarse.ndim == 2 else paths_coarse[:, -1, :]
            final_fine = paths_fine[:, -1] if paths_fine.ndim == 2 else paths_fine[:, -1, :]
            
            # Calculate error estimate
            if paths_coarse.ndim == 2:
                error = np.mean(np.abs(final_coarse - final_fine))
            else:
                error = np.mean(np.linalg.norm(final_coarse - final_fine, axis=1))
                
            if error < self.tolerance or current_steps > 10000:
                # Return the fine grid result (subsample to original grid size)
                step_ratio = (current_steps * 2) // params.n_steps
                indices = np.arange(0, paths_fine.shape[1], step_ratio)[:params.n_steps + 1]
                
                if paths_fine.ndim == 2:
                    return paths_fine[:, indices]
                else:
                    return paths_fine[:, indices, :]
            else:
                # Increase resolution
                current_steps *= 2

class PathGenerator:
    """Main path generation orchestrator"""
    
    def __init__(self):
        self.generators = {
            'euler': EulerMaruyamaGenerator(),
            'milstein': MilsteinGenerator(),
            'rk2': RungeKuttaGenerator(order=2),
            'rk4': RungeKuttaGenerator(order=4),
            'adaptive': AdaptivePathGenerator(EulerMaruyamaGenerator())
        }
        
    def generate(
        self,
        scheme: str,
        params: Union[SDEParameters, Dict],
        n_paths: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate paths using specified scheme
        
        Args:
            scheme: Generation scheme ('euler', 'milstein', etc.)
            params: SDE parameters
            n_paths: Number of paths
            **kwargs: Additional parameters
            
        Returns:
            Generated paths
        """
        if scheme not in self.generators:
            raise ValueError(f"Unknown generation scheme: {scheme}")
            
        if isinstance(params, dict):
            params = SDEParameters(**params)
            
        generator = self.generators[scheme]
        return generator.generate_paths(params, n_paths, **kwargs)
        
    def compare_schemes(
        self,
        schemes: List[str],
        params: Union[SDEParameters, Dict],
        n_paths: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Compare different path generation schemes"""
        if isinstance(params, dict):
            params = SDEParameters(**params)
            
        results = {}
        
        for scheme in schemes:
            if scheme not in self.generators:
                results[scheme] = {'error': f'Unknown scheme: {scheme}'}
                continue
                
            try:
                import time
                start_time = time.time()
                
                paths = self.generate(scheme, params, n_paths, **kwargs)
                
                generation_time = time.time() - start_time
                
                # Calculate statistics
                if paths.ndim == 2:
                    final_values = paths[:, -1]
                else:
                    final_values = np.linalg.norm(paths[:, -1, :], axis=1)
                    
                results[scheme] = {
                    'mean_final': np.mean(final_values),
                    'std_final': np.std(final_values),
                    'generation_time': generation_time,
                    'paths_shape': paths.shape
                }
                
            except Exception as e:
                results[scheme] = {'error': str(e)}
                
        return results

# Example usage and testing
if __name__ == "__main__":
    print("Testing Path Generation Schemes...")
    
    # Define SDE functions for testing
    def gbm_drift(t, x):
        """GBM drift: μx"""
        mu = 0.05
        return mu * x
        
    def gbm_diffusion(t, x):
        """GBM diffusion: σx"""
        sigma = 0.2
        return sigma * x
        
    def gbm_diffusion_derivative(t, x):
        """Derivative of GBM diffusion: σ"""
        sigma = 0.2
        return sigma * np.ones_like(x)
        
    # Setup parameters
    sde_params = SDEParameters(
        initial_value=100.0,
        time_horizon=1.0,
        n_steps=252,
        drift_func=gbm_drift,
        diffusion_func=gbm_diffusion
    )
    
    # Initialize path generator
    path_gen = PathGenerator()
    
    # Test individual schemes
    schemes_to_test = ['euler', 'milstein', 'rk2']
    
    for scheme in schemes_to_test:
        print(f"\nTesting {scheme.upper()} scheme:")
        
        try:
            paths = path_gen.generate(
                scheme, 
                sde_params, 
                n_paths=100,
                diffusion_derivative=gbm_diffusion_derivative
            )
            
            print(f"  Generated paths shape: {paths.shape}")
            print(f"  Final value mean: {np.mean(paths[:, -1]):.3f}")
            print(f"  Final value std: {np.std(paths[:, -1]):.3f}")
            
            # Expected values for GBM: E[S_T] = S_0 * exp(μT)
            expected_mean = 100.0 * np.exp(0.05 * 1.0)
            print(f"  Expected mean: {expected_mean:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Compare schemes
    print("\nComparing Path Generation Schemes:")
    comparison = path_gen.compare_schemes(
        ['euler', 'milstein', 'rk2'], 
        sde_params, 
        n_paths=1000,
        diffusion_derivative=gbm_diffusion_derivative
    )
    
    print(f"{'Scheme':<10} {'Mean Final':<12} {'Std Final':<12} {'Time (s)':<10}")
    print("-" * 50)
    
    for scheme, results in comparison.items():
        if 'error' not in results:
            print(f"{scheme:<10} {results['mean_final']:<12.3f} {results['std_final']:<12.3f} "
                  f"{results['generation_time']:<10.4f}")
        else:
            print(f"{scheme:<10} ERROR: {results['error']}")
    
    print("\nPath generation schemes test completed!")
