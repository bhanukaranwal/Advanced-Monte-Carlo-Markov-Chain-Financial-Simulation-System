"""
Quasi-Monte Carlo methods for improved convergence in financial simulations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

class QuasiRandomSequence(ABC):
    """Abstract base class for quasi-random sequences"""
    
    @abstractmethod
    def generate(self, n_points: int, dimension: int) -> np.ndarray:
        """Generate quasi-random sequence"""
        pass

class SobolGenerator(QuasiRandomSequence):
    """Sobol sequence generator with improved uniformity"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        
    def generate(self, n_points: int, dimension: int) -> np.ndarray:
        """
        Generate Sobol sequence
        
        Args:
            n_points: Number of points to generate
            dimension: Dimensionality of sequence
            
        Returns:
            Array of shape (n_points, dimension) with values in [0,1]
        """
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dimension, seed=self.seed)
            return sampler.random(n_points)
        except ImportError:
            logger.warning("scipy.stats.qmc not available, using custom Sobol implementation")
            return self._custom_sobol(n_points, dimension)
            
    def _custom_sobol(self, n_points: int, dimension: int) -> np.ndarray:
        """Custom Sobol sequence implementation"""
        # Simplified Sobol implementation
        # In practice, would use a full Sobol implementation with direction numbers
        
        sequence = np.zeros((n_points, dimension))
        
        # Use van der Corput sequence for each dimension
        for d in range(dimension):
            base = self._get_prime(d + 1)  # Use different prime bases
            for i in range(n_points):
                sequence[i, d] = self._van_der_corput(i + 1, base)
                
        return sequence
        
    def _van_der_corput(self, n: int, base: int) -> float:
        """Van der Corput sequence in given base"""
        result = 0.0
        f = 1.0 / base
        
        while n > 0:
            result += f * (n % base)
            n //= base
            f /= base
            
        return result
        
    def _get_prime(self, n: int) -> int:
        """Get nth prime number"""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        return primes[min(n - 1, len(primes) - 1)]

class HaltonGenerator(QuasiRandomSequence):
    """Halton sequence generator"""
    
    def __init__(self, bases: Optional[List[int]] = None):
        self.bases = bases or [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
    def generate(self, n_points: int, dimension: int) -> np.ndarray:
        """Generate Halton sequence"""
        if dimension > len(self.bases):
            raise ValueError(f"Dimension {dimension} exceeds available bases {len(self.bases)}")
            
        sequence = np.zeros((n_points, dimension))
        
        for d in range(dimension):
            base = self.bases[d]
            for i in range(n_points):
                sequence[i, d] = self._halton_number(i + 1, base)
                
        return sequence
        
    def _halton_number(self, index: int, base: int) -> float:
        """Generate single Halton number"""
        result = 0.0
        f = 1.0 / base
        i = index
        
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
            
        return result

class LatinHypercubeGenerator(QuasiRandomSequence):
    """Latin Hypercube sampling"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def generate(self, n_points: int, dimension: int) -> np.ndarray:
        """Generate Latin Hypercube sample"""
        sequence = np.zeros((n_points, dimension))
        
        for d in range(dimension):
            # Create stratified samples
            intervals = np.arange(n_points)
            np.random.shuffle(intervals)
            
            # Generate uniform samples within each interval
            uniform_samples = np.random.uniform(0, 1, n_points)
            sequence[:, d] = (intervals + uniform_samples) / n_points
            
        return sequence

class QuasiMonteCarloGenerator:
    """Main quasi-Monte Carlo generator with multiple sequence types"""
    
    def __init__(self, method: str = 'sobol', **kwargs):
        self.method = method
        self.kwargs = kwargs
        
        # Initialize generator
        if method == 'sobol':
            self.generator = SobolGenerator(**kwargs)
        elif method == 'halton':
            self.generator = HaltonGenerator(**kwargs)
        elif method == 'lhs':
            self.generator = LatinHypercubeGenerator(**kwargs)
        else:
            raise ValueError(f"Unknown quasi-Monte Carlo method: {method}")
            
    def generate_uniform(self, n_points: int, dimension: int) -> np.ndarray:
        """Generate uniform quasi-random sequence"""
        return self.generator.generate(n_points, dimension)
        
    def generate_normal(self, n_points: int, dimension: int) -> np.ndarray:
        """Generate normal quasi-random sequence using inverse transform"""
        uniform_sequence = self.generate_uniform(n_points, dimension)
        return norm.ppf(uniform_sequence)
        
    def generate_correlated_normal(
        self, 
        n_points: int, 
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Generate correlated normal variables using Cholesky decomposition
        
        Args:
            n_points: Number of points
            correlation_matrix: Correlation matrix
            
        Returns:
            Correlated normal variables
        """
        dimension = correlation_matrix.shape[0]
        
        # Generate independent normal variables
        independent_normals = self.generate_normal(n_points, dimension)
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # Matrix not positive definite, use eigendecomposition
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
            
        # Apply correlation
        correlated_normals = independent_normals @ L.T
        
        return correlated_normals
        
    def generate_brownian_motion(
        self,
        n_paths: int,
        n_steps: int,
        time_horizon: float = 1.0,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generate Brownian motion paths using quasi-random numbers
        
        Args:
            n_paths: Number of paths
            n_steps: Number of time steps
            time_horizon: Total time
            correlation_matrix: Optional correlation between multiple Brownian motions
            
        Returns:
            Brownian motion paths
        """
        dt = time_horizon / n_steps
        sqrt_dt = np.sqrt(dt)
        
        if correlation_matrix is not None:
            n_assets = correlation_matrix.shape[0]
            # Generate correlated increments
            increments = self.generate_correlated_normal(n_paths * n_steps, correlation_matrix)
            increments = increments.reshape(n_paths, n_steps, n_assets) * sqrt_dt
            
            # Construct paths
            paths = np.zeros((n_paths, n_steps + 1, n_assets))
            for i in range(n_steps):
                paths[:, i + 1, :] = paths[:, i, :] + increments[:, i, :]
                
        else:
            # Single Brownian motion
            increments = self.generate_normal(n_paths, n_steps) * sqrt_dt
            
            # Construct paths
            paths = np.zeros((n_paths, n_steps + 1))
            paths[:, 1:] = np.cumsum(increments, axis=1)
            
        return paths
        
    def compare_convergence(
        self,
        target_function: callable,
        dimension: int,
        max_points: int = 10000,
        methods: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compare convergence rates of different methods
        
        Args:
            target_function: Function to integrate/estimate
            dimension: Problem dimension
            max_points: Maximum number of points to test
            methods: List of methods to compare
            
        Returns:
            Dictionary of convergence results
        """
        if methods is None:
            methods = ['sobol', 'halton', 'lhs', 'monte_carlo']
            
        results = {}
        test_points = np.logspace(2, np.log10(max_points), 20, dtype=int)
        
        for method in methods:
            estimates = []
            
            for n_points in test_points:
                if method == 'monte_carlo':
                    # Standard Monte Carlo for comparison
                    sequence = np.random.uniform(0, 1, (n_points, dimension))
                else:
                    # Quasi-Monte Carlo
                    temp_generator = QuasiMonteCarloGenerator(method)
                    sequence = temp_generator.generate_uniform(n_points, dimension)
                    
                # Evaluate function
                values = np.array([target_function(point) for point in sequence])
                estimate = np.mean(values)
                estimates.append(estimate)
                
            results[method] = {
                'points': test_points,
                'estimates': np.array(estimates)
            }
            
        return results

class BrownianBridge:
    """Brownian bridge construction for path generation"""
    
    def __init__(self, quasi_generator: QuasiMonteCarloGenerator):
        self.quasi_generator = quasi_generator
        
    def generate_paths(
        self,
        n_paths: int,
        time_points: np.ndarray,
        initial_value: float = 0.0,
        final_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate Brownian bridge paths
        
        Args:
            n_paths: Number of paths
            time_points: Array of time points
            initial_value: Starting value
            final_value: Ending value (None for free bridge)
            
        Returns:
            Brownian bridge paths
        """
        n_steps = len(time_points) - 1
        T = time_points[-1] - time_points[0]
        
        if final_value is None:
            # Free Brownian motion
            increments = self.quasi_generator.generate_normal(n_paths, n_steps)
            dt_array = np.diff(time_points)
            sqrt_dt_array = np.sqrt(dt_array)
            
            paths = np.zeros((n_paths, len(time_points)))
            paths[:, 0] = initial_value
            
            for i in range(n_steps):
                paths[:, i + 1] = paths[:, i] + increments[:, i] * sqrt_dt_array[i]
                
        else:
            # Brownian bridge to fixed endpoint
            paths = np.zeros((n_paths, len(time_points)))
            paths[:, 0] = initial_value
            paths[:, -1] = final_value
            
            # Generate bridge using binary subdivision
            self._construct_bridge(paths, time_points, 0, n_steps, n_paths)
            
        return paths
        
    def _construct_bridge(
        self,
        paths: np.ndarray,
        time_points: np.ndarray,
        left_idx: int,
        right_idx: int,
        n_paths: int
    ):
        """Recursively construct Brownian bridge"""
        if right_idx - left_idx <= 1:
            return
            
        mid_idx = (left_idx + right_idx) // 2
        t_left = time_points[left_idx]
        t_mid = time_points[mid_idx]
        t_right = time_points[right_idx]
        
        # Bridge variance
        variance = (t_mid - t_left) * (t_right - t_mid) / (t_right - t_left)
        
        # Generate normal increments
        increments = self.quasi_generator.generate_normal(n_paths, 1).flatten()
        
        # Bridge formula
        bridge_mean = (
            paths[:, left_idx] * (t_right - t_mid) + 
            paths[:, right_idx] * (t_mid - t_left)
        ) / (t_right - t_left)
        
        paths[:, mid_idx] = bridge_mean + increments * np.sqrt(variance)
        
        # Recursively fill remaining points
        self._construct_bridge(paths, time_points, left_idx, mid_idx, n_paths)
        self._construct_bridge(paths, time_points, mid_idx, right_idx, n_paths)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Quasi-Monte Carlo Generators...")
    
    # Test different generators
    methods = ['sobol', 'halton', 'lhs']
    
    for method in methods:
        print(f"\nTesting {method.upper()} generator:")
        
        qmc_gen = QuasiMonteCarloGenerator(method)
        
        # Test uniform generation
        uniform_seq = qmc_gen.generate_uniform(1000, 2)
        print(f"  Uniform sequence shape: {uniform_seq.shape}")
        print(f"  Range: [{uniform_seq.min():.3f}, {uniform_seq.max():.3f}]")
        
        # Test normal generation
        normal_seq = qmc_gen.generate_normal(1000, 2)
        print(f"  Normal sequence mean: {normal_seq.mean(axis=0)}")
        print(f"  Normal sequence std: {normal_seq.std(axis=0)}")
        
        # Test correlated normal generation
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        correlated_seq = qmc_gen.generate_correlated_normal(1000, correlation_matrix)
        empirical_corr = np.corrcoef(correlated_seq.T)
        print(f"  Target correlation: 0.5, Empirical: {empirical_corr[0,1]:.3f}")
        
    # Test Brownian motion generation
    print("\nTesting Brownian Motion Generation:")
    qmc_gen = QuasiMonteCarloGenerator('sobol')
    bm_paths = qmc_gen.generate_brownian_motion(100, 252, time_horizon=1.0)
    print(f"Brownian motion paths shape: {bm_paths.shape}")
    print(f"Final values mean: {bm_paths[:, -1].mean():.3f}")
    print(f"Final values std: {bm_paths[:, -1].std():.3f}")
    
    # Test Brownian bridge
    print("\nTesting Brownian Bridge:")
    bridge_gen = BrownianBridge(qmc_gen)
    time_points = np.linspace(0, 1, 253)
    bridge_paths = bridge_gen.generate_paths(100, time_points, initial_value=0.0, final_value=0.0)
    print(f"Bridge paths shape: {bridge_paths.shape}")
    print(f"All paths start at 0: {np.allclose(bridge_paths[:, 0], 0)}")
    print(f"All paths end at 0: {np.allclose(bridge_paths[:, -1], 0)}")
    
    # Test convergence comparison
    print("\nTesting Convergence Comparison:")
    
    def test_function(x):
        """Simple test function: integration of x^2 over unit cube"""
        return np.sum(x**2)
        
    # Only test 2D for speed
    qmc_gen = QuasiMonteCarloGenerator('sobol')
    convergence_results = qmc_gen.compare_convergence(
        test_function, dimension=2, max_points=1000, methods=['sobol', 'monte_carlo']
    )
    
    for method, result in convergence_results.items():
        final_estimate = result['estimates'][-1]
        print(f"  {method}: final estimate = {final_estimate:.4f}")
        
    print("\nQuasi-Monte Carlo test completed!")
