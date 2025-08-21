"""
Geometric Brownian Motion Monte Carlo Engine
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation"""
    paths: np.ndarray
    final_prices: np.ndarray
    statistics: Dict[str, float]
    execution_time: float
    parameters: Dict[str, Any]

class GeometricBrownianMotionEngine:
    """High-performance Geometric Brownian Motion Monte Carlo engine"""
    
    def __init__(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        drift: float,
        volatility: float,
        random_seed: Optional[int] = None,
        use_gpu: bool = True,
        antithetic_variates: bool = False
    ):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.random_seed = random_seed
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.antithetic_variates = antithetic_variates
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate simulation parameters"""
        if self.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        if self.initial_price <= 0:
            raise ValueError("Initial price must be positive")
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")
            
    def simulate_paths(self) -> SimulationResult:
        """Generate Monte Carlo paths"""
        logger.info(f"Starting GBM simulation: {self.n_simulations} paths, {self.n_steps} steps")
        start_time = time.time()
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            if self.use_gpu:
                cp.random.seed(self.random_seed)
                
        try:
            if self.use_gpu:
                paths = self._simulate_gpu()
            else:
                paths = self._simulate_cpu()
                
            execution_time = time.time() - start_time
            
            # Calculate statistics
            final_prices = paths[:, -1]
            statistics = self._calculate_statistics(final_prices)
            
            logger.info(f"Simulation completed in {execution_time:.3f}s")
            
            return SimulationResult(
                paths=paths,
                final_prices=final_prices,
                statistics=statistics,
                execution_time=execution_time,
                parameters={
                    'n_simulations': self.n_simulations,
                    'n_steps': self.n_steps,
                    'initial_price': self.initial_price,
                    'drift': self.drift,
                    'volatility': self.volatility
                }
            )
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            if self.use_gpu:
                logger.info("Falling back to CPU simulation")
                self.use_gpu = False
                return self.simulate_paths()
            raise
            
    def _simulate_cpu(self) -> np.ndarray:
        """CPU-based simulation"""
        dt = 1.0 / self.n_steps
        drift_term = (self.drift - 0.5 * self.volatility**2) * dt
        diffusion_term = self.volatility * np.sqrt(dt)
        
        # Generate random numbers
        if self.antithetic_variates:
            n_sims_half = self.n_simulations // 2
            randoms_half = np.random.normal(0, 1, (n_sims_half, self.n_steps))
            randoms = np.vstack([randoms_half, -randoms_half])
            if self.n_simulations % 2 == 1:
                extra_random = np.random.normal(0, 1, (1, self.n_steps))
                randoms = np.vstack([randoms, extra_random])
        else:
            randoms = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
            
        # Calculate log returns
        log_returns = drift_term + diffusion_term * randoms
        
        # Calculate paths
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.full((self.n_simulations, 1), np.log(self.initial_price)),
            log_prices
        ])
        
        paths = np.exp(log_prices)
        return paths
        
    def _simulate_gpu(self) -> np.ndarray:
        """GPU-accelerated simulation using CuPy"""
        dt = 1.0 / self.n_steps
        drift_term = (self.drift - 0.5 * self.volatility**2) * dt
        diffusion_term = self.volatility * cp.sqrt(dt)
        
        # Generate random numbers on GPU
        if self.antithetic_variates:
            n_sims_half = self.n_simulations // 2
            randoms_half = cp.random.normal(0, 1, (n_sims_half, self.n_steps))
            randoms = cp.vstack([randoms_half, -randoms_half])
            if self.n_simulations % 2 == 1:
                extra_random = cp.random.normal(0, 1, (1, self.n_steps))
                randoms = cp.vstack([randoms, extra_random])
        else:
            randoms = cp.random.normal(0, 1, (self.n_simulations, self.n_steps))
            
        # Calculate log returns
        log_returns = drift_term + diffusion_term * randoms
        
        # Calculate paths
        log_prices = cp.cumsum(log_returns, axis=1)
        log_prices = cp.hstack([
            cp.full((self.n_simulations, 1), cp.log(self.initial_price)),
            log_prices
        ])
        
        paths = cp.exp(log_prices)
        
        # Convert back to numpy
        return cp.asnumpy(paths)
        
    def _calculate_statistics(self, final_prices: np.ndarray) -> Dict[str, float]:
        """Calculate summary statistics"""
        return {
            'mean': float(np.mean(final_prices)),
            'std': float(np.std(final_prices)),
            'min': float(np.min(final_prices)),
            'max': float(np.max(final_prices)),
            'median': float(np.median(final_prices)),
            'percentile_5': float(np.percentile(final_prices, 5)),
            'percentile_25': float(np.percentile(final_prices, 25)),
            'percentile_75': float(np.percentile(final_prices, 75)),
            'percentile_95': float(np.percentile(final_prices, 95))
        }
