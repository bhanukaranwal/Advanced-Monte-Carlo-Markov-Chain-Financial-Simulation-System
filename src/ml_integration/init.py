"""
Machine Learning Integration for Advanced Monte Carlo-Markov Finance System

Neural surrogates, reinforcement learning, and ensemble methods for enhanced simulation
"""

from .neural_surrogates import NeuralSurrogate, DeepPathGenerator, VariationalAutoencoder
from .rl_optimizer import RLOptimizer, PolicyGradientOptimizer, QLearningOptimizer
from .ensemble_methods import EnsemblePredictor, ModelStacking, BayesianModelAveraging

__all__ = [
    "NeuralSurrogate",
    "DeepPathGenerator", 
    "VariationalAutoencoder",
    "RLOptimizer",
    "PolicyGradientOptimizer",
    "QLearningOptimizer", 
    "EnsemblePredictor",
    "ModelStacking",
    "BayesianModelAveraging"
]
