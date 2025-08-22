"""
Performance analytics and attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """Portfolio performance analytics"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Default 2%
        
    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """Calculate returns from price series"""
        return np.diff(prices) / prices[:-1]
        
    def calculate_cumulative_returns(self, returns: np.ndarray) -> np.ndarray:
        """Calculate cumulative returns"""
        return np.cumprod(1 + returns) - 1
        
    def calculate_annualized_return(self, returns: np.ndarray, 
                                   periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        total_return = np.prod(1 + returns) - 1
        n_periods = len(returns)
        annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
        return annualized
        
    def calculate_annualized_volatility(self, returns: np.ndarray,
                                      periods_per_year: int = 252) -> float:
        """Calculate annualized volatility"""
        return np.std(returns, ddof=1) * np.sqrt(periods_per_year)
        
    def calculate_beta(self, portfolio_returns: np.ndarray,
                      market_returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns, ddof=1)
        return covariance / market_variance if market_variance > 0 else 0
        
    def calculate_alpha(self, portfolio_returns: np.ndarray,
                       market_returns: np.ndarray,
                       risk_free_rate: Optional[float] = None) -> float:
        """Calculate Jensen's alpha"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate / 252  # Daily rate
            
        beta = self.calculate_beta(portfolio_returns, market_returns)
        
        portfolio_excess = np.mean(portfolio_returns) - risk_free_rate
        market_excess = np.mean(market_returns) - risk_free_rate
        
        expected_excess = beta * market_excess
        alpha = portfolio_excess - expected_excess
        
        return alpha * 252  # Annualized
        
    def calculate_information_ratio(self, portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns, ddof=1)
        
        if tracking_error == 0:
            return 0
            
        return np.mean(active_returns) / tracking_error * np.sqrt(252)
        
    def calculate_tracking_error(self, portfolio_returns: np.ndarray,
                                benchmark_returns: np.ndarray) -> float:
        """Calculate tracking error"""
        active_returns = portfolio_returns - benchmark_returns
        return np.std(active_returns, ddof=1) * np.sqrt(252)
        
    def performance_summary(self, returns: np.ndarray,
                          benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Generate comprehensive performance summary"""
        
        summary = {
            'total_return': np.prod(1 + returns) - 1,
            'annualized_return': self.calculate_annualized_return(returns),
            'annualized_volatility': self.calculate_annualized_volatility(returns),
            'sharpe_ratio': (np.mean(returns) - self.risk_free_rate/252) / np.std(returns, ddof=1) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'calmar_ratio': 0,  # Will calculate after max drawdown
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurtosis()),
            'best_month': np.max(returns) if len(returns) > 0 else 0,
            'worst_month': np.min(returns) if len(returns) > 0 else 0
        }
        
        # Calculate Calmar ratio
        if summary['max_drawdown'] != 0:
            summary['calmar_ratio'] = summary['annualized_return'] / abs(summary['max_drawdown'])
            
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_returns is not None:
            summary.update({
                'beta': self.calculate_beta(returns, benchmark_returns),
                'alpha': self.calculate_alpha(returns, benchmark_returns),
                'information_ratio': self.calculate_information_ratio(returns, benchmark_returns),
                'tracking_error': self.calculate_tracking_error(returns, benchmark_returns)
            })
            
        return summary
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
