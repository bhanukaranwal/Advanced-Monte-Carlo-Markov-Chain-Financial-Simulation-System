"""
Risk analytics and measurement tools
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMeasures:
    """Container for risk measurement results"""
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    maximum_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

class RiskAnalytics:
    """Risk analytics engine"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """Calculate Value at Risk"""
        if method == 'historical':
            return self._var_historical(returns, confidence_level)
        elif method == 'parametric':
            return self._var_parametric(returns, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
    def _var_historical(self, returns: np.ndarray, confidence_level: float) -> float:
        """Historical simulation VaR"""
        return np.percentile(returns, (1 - confidence_level) * 100)
        
    def _var_parametric(self, returns: np.ndarray, confidence_level: float) -> float:
        """Parametric VaR assuming normal distribution"""
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        return mu + sigma * stats.norm.ppf(1 - confidence_level)
        
    def calculate_expected_shortfall(self, returns: np.ndarray, 
                                   confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.calculate_var(returns, confidence_level, method='historical')
        return np.mean(returns[returns <= var])
        
    def calculate_maximum_drawdown(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(prices)
        drawdowns = (prices - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)
        peak_idx = np.argmax(running_max[:max_dd_idx + 1])
        
        return {
            'maximum_drawdown': max_drawdown,
            'peak_index': peak_idx,
            'trough_index': max_dd_idx,
            'duration_days': max_dd_idx - peak_idx,
            'drawdown_series': drawdowns
        }
        
    def calculate_comprehensive_risk_measures(self, returns: np.ndarray,
                                            risk_free_rate: float = 0.02) -> RiskMeasures:
        """Calculate comprehensive set of risk measures"""
        
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
        
        # VaR measures
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        
        # Expected Shortfall
        es_95 = self.calculate_expected_shortfall(returns, 0.95)
        es_99 = self.calculate_expected_shortfall(returns, 0.99)
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        mdd_results = self.calculate_maximum_drawdown(cumulative_returns)
        maximum_drawdown = mdd_results['maximum_drawdown']
        
        # Risk-adjusted ratios
        excess_return = mean_return - risk_free_rate / 252
        
        # Sharpe ratio
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        annualized_return = mean_return * 252
        calmar_ratio = annualized_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0
        
        return RiskMeasures(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            maximum_drawdown=maximum_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio
        )
