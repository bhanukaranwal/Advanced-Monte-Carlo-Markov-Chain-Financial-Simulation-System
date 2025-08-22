"""
Multi-Asset Monte Carlo Engine with Correlation
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from .base_engine import BaseMonteCarloEngine

logger = logging.getLogger(__name__)

@dataclass
class MultiAssetResult:
    """Results from multi-asset simulation"""
    paths: np.ndarray
    correlation_matrix: np.ndarray
    asset_statistics: Dict[str, Dict[str, float]]
    execution_time: float

class MultiAssetEngine(BaseMonteCarloEngine):
    """
    Multi-asset Monte Carlo simulation with correlation modeling
    """
    
    def __init__(
        self,
        n_simulations: int,
        n_steps: int,
        initial_prices: List[float],
        drifts: List[float],
        volatilities: List[float],
        correlation_matrix: np.ndarray,
        asset_names: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ):
        super().__init__(n_simulations, n_steps, random_seed, False)  # Multi-asset typically CPU
        
        self.initial_prices = np.array(initial_prices)
        self.drifts = np.array(drifts)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = correlation_matrix
        self.n_assets = len(initial_prices)
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.n_assets)]
        
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
            
    def simulate(self) -> MultiAssetResult:
        """Simulate correlated multi-asset paths"""
        logger.info(f"Starting multi-asset simulation: {self.n_assets} assets, {self.n_simulations} paths")
        start_time = time.time()
        
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
        
        execution_time = time.time() - start_time
        
        # Calculate statistics for each asset
        asset_statistics = {}
        for i, name in enumerate(self.asset_names):
            final_prices = paths[:, -1, i]
            asset_statistics[name] = {
                'mean': float(np.mean(final_prices)),
                'std': float(np.std(final_prices)),
                'min': float(np.min(final_prices)),
                'max': float(np.max(final_prices))
            }
            
        logger.info(f"Multi-asset simulation completed in {execution_time:.3f}s")
        
        return MultiAssetResult(
            paths=paths,
            correlation_matrix=self.correlation_matrix,
            asset_statistics=asset_statistics,
            execution_time=execution_time
        )
