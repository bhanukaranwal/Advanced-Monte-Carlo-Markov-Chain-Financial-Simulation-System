"""
Multi-Asset Monte Carlo Engine with Correlation Modeling
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiAssetEngine:
    """Multi-asset Monte Carlo simulation with correlation modeling"""
    
    def __init__(
        self,
        n_simulations: int,
        n_steps: int,
        initial_prices: List[float],
        drifts: List[float],
        volatilities: List[float],
        correlation_matrix: np.ndarray,
        random_seed: Optional[int] = None
    ):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.initial_prices = np.array(initial_prices)
        self.drifts = np.array(drifts)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = correlation_matrix
        self.random_seed = random_seed
        self.n_assets = len(initial_prices)
        
        # Validate inputs
        self._validate_parameters()
        
        # Compute Cholesky decomposition for correlation
        self.cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        
    def _validate_parameters(self):
        """Validate multi-asset parameters"""
        if len(self.initial_prices) != len(self.drifts) != len(self.volatilities):
            raise ValueError("All asset parameter arrays must have same length")
        
        if self.correlation_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Correlation matrix dimension mismatch")
        
        # Check if correlation matrix is positive definite
        try:
            np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix is not positive definite")
            
    def simulate_correlated_paths(self) -> np.ndarray:
        """Generate correlated multi-asset Monte Carlo paths"""
        logger.info(f"Starting multi-asset simulation: {self.n_assets} assets, {self.n_simulations} paths")
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        dt = 1.0 / self.n_steps
        
        # Pre-compute drift and volatility terms
        drift_terms = (self.drifts - 0.5 * self.volatilities**2) * dt
        vol_terms = self.volatilities * np.sqrt(dt)
        
        # Generate independent random numbers
        independent_randoms = np.random.normal(
            0, 1, (self.n_simulations, self.n_steps, self.n_assets)
        )
        
        # Apply Cholesky decomposition for correlation
        correlated_randoms = np.dot(independent_randoms, self.cholesky_matrix.T)
        
        # Calculate log returns for all assets
        log_returns = (
            drift_terms[np.newaxis, np.newaxis, :] + 
            vol_terms[np.newaxis, np.newaxis, :] * correlated_randoms
        )
        
        # Calculate cumulative log prices
        log_prices = np.cumsum(log_returns, axis=1)
        
        # Add initial log prices
        initial_log_prices = np.log(self.initial_prices)
        log_prices = np.concatenate([
            np.tile(initial_log_prices, (self.n_simulations, 1, 1)),
            log_prices
        ], axis=1)
        
        # Convert to price paths
        paths = np.exp(log_prices)
        
        return paths
