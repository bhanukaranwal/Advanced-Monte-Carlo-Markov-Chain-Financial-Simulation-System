"""
Base Monte Carlo Engine
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseMonteCarloEngine(ABC):
    """Base class for all Monte Carlo engines"""
    
    def __init__(self, n_simulations: int, n_steps: int, random_seed=None, use_gpu=False):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.random_seed = random_seed
        self.use_gpu = use_gpu
        
    @abstractmethod
    def simulate(self):
        """Run the Monte Carlo simulation"""
        pass
        
    def validate_inputs(self):
        """Validate common inputs"""
        if self.n_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if self.n_steps <= 0:
            raise ValueError("Number of time steps must be positive")
