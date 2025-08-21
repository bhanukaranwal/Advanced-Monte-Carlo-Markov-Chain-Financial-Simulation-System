"""
Portfolio optimization using modern portfolio theory and advanced techniques
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Advanced portfolio optimization engine"""
    
    def __init__(self):
        self.optimized_weights = None
        self.optimization_result = None
        
    def optimize_mean_variance(self, expected_returns: np.ndarray,
                             covariance_matrix: np.ndarray,
                             target_return: Optional[float] = None,
                             risk_aversion: float = 1.0) -> Dict:
        """
        Mean-variance optimization (Markowitz)
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Asset covariance matrix
            target_return: Target portfolio return (if None, maximize Sharpe ratio)
            risk_aversion: Risk aversion parameter
        """
        n_assets = len(expected_returns)
        
        # Objective function
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            if target_return is not None:
                # Minimize variance for target return
                return portfolio_variance
            else:
                # Maximize utility (return - risk_aversion * variance)
                return -(portfolio_return - risk_aversion * portfolio_variance)
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Target return constraint (if specified)
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.dot(weights, expected_returns) - target_return
            })
        
        # Bounds (long-only portfolio)
        bounds = Bounds(0, 1)
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            self.optimized_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(result.x, expected_returns)
            portfolio_variance = np.dot(result.x.T, np.dot(covariance_matrix, result.x))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            return {
                'weights': result.x,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_return / portfolio_volatility,
                'optimization_success': True,
                'message': result.message
            }
        else:
            logger.error(f"Optimization failed: {result.message}")
            return {
                'optimization_success': False,
                'message': result.message
            }
            
    def optimize_risk_parity(self, covariance_matrix: np.ndarray) -> Dict:
        """Risk parity optimization - equal risk contribution"""
        
        n_assets = covariance_matrix.shape[0]
        
        def risk_budget_objective(weights):
            """Minimize sum of squared differences from equal risk contribution"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Marginal risk contributions
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk contribution
            target_contrib = np.ones(n_assets) / n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = Bounds(0.001, 1)  # Small minimum to avoid division by zero
        
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9}
        )
        
        if result.success:
            weights = result.x
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            return {
                'weights': weights,
                'volatility': portfolio_vol,
                'optimization_success': True
            }
        else:
            return {'optimization_success': False, 'message': result.message}
            
    def optimize_minimum_variance(self, covariance_matrix: np.ndarray) -> Dict:
        """Minimum variance portfolio optimization"""
        
        n_assets = covariance_matrix.shape[0]
        
        def objective(weights):
            return np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = Bounds(0, 1)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            min_variance = result.fun
            min_volatility = np.sqrt(min_variance)
            
            return {
                'weights': weights,
                'volatility': min_volatility,
                'variance': min_variance,
                'optimization_success': True
            }
        else:
            return {'optimization_success': False, 'message': result.message}
            
    def efficient_frontier(self, expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray,
                         n_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        results = []
        
        for target_return in target_returns:
            result = self.optimize_mean_variance(
                expected_returns,
                covariance_matrix,
                target_return=target_return
            )
            
            if result['optimization_success']:
                results.append({
                    'target_return': target_return,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'weights': result['weights']
                })
                
        return pd.DataFrame(results)
