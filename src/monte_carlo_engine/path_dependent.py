"""
Path-Dependent Option Pricing Engine
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PathDependentEngine:
    """Engine for pricing path-dependent options"""
    
    def __init__(self, n_simulations: int, n_steps: int, initial_price: float,
                 drift: float, volatility: float, random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.random_seed = random_seed
        
    def price_asian_option(self, strike: float, option_type: str,
                          averaging_type: str = 'arithmetic',
                          risk_free_rate: float = 0.05,
                          time_to_maturity: float = 1.0) -> Dict[str, Any]:
        """Price Asian option with arithmetic or geometric averaging"""
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        dt = time_to_maturity / self.n_steps
        drift_term = (self.drift - 0.5 * self.volatility**2) * dt
        vol_term = self.volatility * np.sqrt(dt)
        
        # Generate paths
        randoms = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        log_returns = drift_term + vol_term * randoms
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.full((self.n_simulations, 1), np.log(self.initial_price)),
            log_prices
        ])
        paths = np.exp(log_prices)
        
        # Calculate averages
        if averaging_type == 'arithmetic':
            averages = np.mean(paths, axis=1)
        elif averaging_type == 'geometric':
            averages = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")
            
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(averages - strike, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(strike - averages, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Discount to present value
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        return {
            'option_price': option_price,
            'standard_error': discount_factor * np.std(payoffs) / np.sqrt(self.n_simulations),
            'payoffs': payoffs,
            'paths': paths
        }
        
    def price_barrier_option(self, strike: float, barrier: float,
                           option_type: str, barrier_type: str,
                           risk_free_rate: float = 0.05,
                           time_to_maturity: float = 1.0,
                           rebate: float = 0.0) -> Dict[str, Any]:
        """Price barrier options (knock-in/knock-out)"""
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            
        dt = time_to_maturity / self.n_steps
        drift_term = (self.drift - 0.5 * self.volatility**2) * dt
        vol_term = self.volatility * np.sqrt(dt)
        
        # Generate paths
        randoms = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        log_returns = drift_term + vol_term * randoms
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.full((self.n_simulations, 1), np.log(self.initial_price)),
            log_prices
        ])
        paths = np.exp(log_prices)
        
        # Check barrier conditions
        if barrier_type in ['up-and-out', 'up-and-in']:
            barrier_hit = np.any(paths >= barrier, axis=1)
        elif barrier_type in ['down-and-out', 'down-and-in']:
            barrier_hit = np.any(paths <= barrier, axis=1)
        else:
            raise ValueError("Invalid barrier_type")
            
        # Calculate vanilla option payoffs
        final_prices = paths[:, -1]
        if option_type.lower() == 'call':
            vanilla_payoffs = np.maximum(final_prices - strike, 0)
        elif option_type.lower() == 'put':
            vanilla_payoffs = np.maximum(strike - final_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
            
        # Apply barrier conditions
        if barrier_type.endswith('out'):
            # Knock-out: option worthless if barrier hit
            payoffs = np.where(barrier_hit, rebate, vanilla_payoffs)
        elif barrier_type.endswith('in'):
            # Knock-in: option only active if barrier hit
            payoffs = np.where(barrier_hit, vanilla_payoffs, rebate)
            
        # Discount to present value
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        option_price = discount_factor * np.mean(payoffs)
        
        return {
            'option_price': option_price,
            'standard_error': discount_factor * np.std(payoffs) / np.sqrt(self.n_simulations),
            'barrier_hit_rate': np.mean(barrier_hit),
            'payoffs': payoffs
        }
