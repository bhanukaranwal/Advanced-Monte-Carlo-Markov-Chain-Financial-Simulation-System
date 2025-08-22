"""
Attribution analytics for portfolio performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AttributionAnalytics:
    """Portfolio attribution analysis"""
    
    def sector_attribution(self, 
                          portfolio_weights: Dict[str, float],
                          portfolio_returns: Dict[str, float],
                          benchmark_weights: Dict[str, float],
                          benchmark_returns: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Calculate sector attribution using Brinson-Fachler methodology
        
        Returns:
            Dict with attribution breakdown by sector
        """
        
        attribution = {}
        
        for sector in portfolio_weights.keys():
            wp = portfolio_weights.get(sector, 0)  # Portfolio weight
            wb = benchmark_weights.get(sector, 0)  # Benchmark weight
            rp = portfolio_returns.get(sector, 0)  # Portfolio return
            rb = benchmark_returns.get(sector, 0)  # Benchmark return
            
            # Allocation effect: (wp - wb) * rb
            allocation_effect = (wp - wb) * rb
            
            # Selection effect: wb * (rp - rb)
            selection_effect = wb * (rp - rb)
            
            # Interaction effect: (wp - wb) * (rp - rb)
            interaction_effect = (wp - wb) * (rp - rb)
            
            # Total attribution
            total_attribution = allocation_effect + selection_effect + interaction_effect
            
            attribution[sector] = {
                'allocation_effect': allocation_effect,
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect,
                'total_attribution': total_attribution
            }
            
        return attribution
        
    def factor_attribution(self,
                          returns: np.ndarray,
                          factor_exposures: np.ndarray,
                          factor_returns: np.ndarray) -> Dict[str, float]:
        """
        Factor-based attribution analysis
        
        Args:
            returns: Portfolio returns
            factor_exposures: Matrix of factor exposures (n_periods x n_factors)
            factor_returns: Factor returns (n_periods x n_factors)
            
        Returns:
            Attribution to each factor
        """
        
        # Simple factor attribution: R_p = alpha + beta_1 * F_1 + ... + beta_n * F_n + epsilon
        
        # Calculate factor contributions
        factor_contributions = factor_exposures * factor_returns
        
        # Sum across time periods
        total_factor_contributions = np.sum(factor_contributions, axis=0)
        
        # Calculate residual (stock-specific return)
        explained_return = np.sum(factor_contributions, axis=1)
        residual = returns - explained_return
        total_residual = np.sum(residual)
        
        attribution = {}
        
        # Assuming factor names (in practice, these would be provided)
        factor_names = [f'Factor_{i+1}' for i in range(factor_exposures.shape[1])]
        
        for i, factor_name in enumerate(factor_names):
            attribution[factor_name] = total_factor_contributions[i]
            
        attribution['Stock_Specific'] = total_residual
        
        return attribution
        
    def risk_attribution(self,
                        portfolio_weights: np.ndarray,
                        covariance_matrix: np.ndarray) -> Dict[str, float]:
        """
        Risk attribution analysis
        
        Args:
            portfolio_weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Risk contribution by asset
        """
        
        # Portfolio variance
        portfolio_variance = np.dot(portfolio_weights.T, 
                                  np.dot(covariance_matrix, portfolio_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Marginal risk contributions
        marginal_contributions = np.dot(covariance_matrix, portfolio_weights) / portfolio_volatility
        
        # Risk contributions
        risk_contributions = portfolio_weights * marginal_contributions / portfolio_volatility
        
        # Convert to dictionary (assuming asset names)
        asset_names = [f'Asset_{i+1}' for i in range(len(portfolio_weights))]
        
        return dict(zip(asset_names, risk_contributions))
        
    def style_attribution(self,
                         portfolio_returns: np.ndarray,
                         style_factors: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Style-based attribution analysis
        
        Args:
            portfolio_returns: Portfolio returns time series
            style_factors: Dictionary of style factor returns
            
        Returns:
            Attribution to each style factor
        """
        
        # Prepare factor matrix
        factor_names = list(style_factors.keys())
        factor_matrix = np.column_stack([style_factors[name] for name in factor_names])
        
        # Add constant for alpha
        X = np.column_stack([np.ones(len(portfolio_returns)), factor_matrix])
        
        # Regression: R_p = alpha + beta_1 * Style_1 + ... + beta_n * Style_n
        coefficients = np.linalg.lstsq(X, portfolio_returns, rcond=None)[0]
        
        alpha = coefficients[0]
        factor_loadings = coefficients[1:]
        
        # Calculate factor contributions
        attribution = {'Alpha': alpha * len(portfolio_returns)}  # Total alpha contribution
        
        for i, factor_name in enumerate(factor_names):
            factor_contribution = factor_loadings[i] * np.sum(style_factors[factor_name])
            attribution[factor_name] = factor_contribution
            
        return attribution
        
    def generate_attribution_report(self,
                                   portfolio_data: Dict[str, any],
                                   benchmark_data: Dict[str, any]) -> Dict[str, any]:
        """Generate comprehensive attribution report"""
        
        report = {
            'summary': {},
            'sector_attribution': {},
            'risk_attribution': {},
            'factor_attribution': {}
        }
        
        # Calculate total portfolio vs benchmark performance
        portfolio_return = portfolio_data.get('total_return', 0)
        benchmark_return = benchmark_data.get('total_return', 0)
        active_return = portfolio_return - benchmark_return
        
        report['summary'] = {
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'active_return': active_return,
            'tracking_error': portfolio_data.get('tracking_error', 0)
        }
        
        # Sector attribution (if sector data available)
        if 'sector_weights' in portfolio_data and 'sector_weights' in benchmark_data:
            report['sector_attribution'] = self.sector_attribution(
                portfolio_data['sector_weights'],
                portfolio_data['sector_returns'],
                benchmark_data['sector_weights'],
                benchmark_data['sector_returns']
            )
            
        return report
