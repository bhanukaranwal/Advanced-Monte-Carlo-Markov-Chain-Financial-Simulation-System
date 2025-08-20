"""
Monte Carlo Engine for Advanced Financial Simulation

Comprehensive Monte Carlo methods with variance reduction techniques
"""

from .variance_reduction import VarianceReduction, AntitheticVariates, ControlVariates, ImportanceSampling
from .quasi_monte_carlo import QuasiMonteCarloGenerator, SobolGenerator, HaltonGenerator
from .stochastic_processes import StochasticProcesses, GeometricBrownianMotion, HestonModel, JumpDiffusionModel
from .path_generators import PathGenerator, EulerMaruyamaGenerator, MilsteinGenerator

__all__ = [
    "VarianceReduction",
    "AntitheticVariates", 
    "ControlVariates",
    "ImportanceSampling",
    "QuasiMonteCarloGenerator",
    "SobolGenerator",
    "HaltonGenerator", 
    "StochasticProcesses",
    "GeometricBrownianMotion",
    "HestonModel",
    "JumpDiffusionModel",
    "PathGenerator",
    "EulerMaruyamaGenerator",
    "MilsteinGenerator"
]

class MonteCarloEngine:
    """Main Monte Carlo simulation engine"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.variance_reduction = VarianceReduction(self.config.get('variance_reduction', {}))
        self.quasi_mc = QuasiMonteCarloGenerator()
        self.stochastic_processes = StochasticProcesses()
        self.path_generator = PathGenerator()
        
    def simulate(
        self,
        process_type: str,
        n_paths: int,
        n_steps: int,
        **kwargs
    ):
        """Main simulation interface"""
        # Implementation in main.py uses this engine
        pass
