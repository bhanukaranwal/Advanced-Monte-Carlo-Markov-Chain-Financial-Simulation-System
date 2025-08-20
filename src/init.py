"""
Advanced Monte Carlo-Markov Chain Financial Simulation System

A comprehensive framework for quantitative finance simulations combining
Monte Carlo methods with adaptive Markov chains.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Core imports
from .main import AdvancedMCMarkovSystem
from .data_engine import MarketDataIngester, FeatureEngineer
from .markov_engine import AdaptiveMarkovChain
from .monte_carlo_engine import MonteCarloEngine
from .analytics_engine import RiskAnalytics

# Version info
version_info = tuple(map(int, __version__.split('.')))

__all__ = [
    "AdvancedMCMarkovSystem",
    "MarketDataIngester", 
    "FeatureEngineer",
    "AdaptiveMarkovChain",
    "MonteCarloEngine", 
    "RiskAnalytics",
    "__version__",
    "version_info"
]
