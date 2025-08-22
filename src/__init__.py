"""
Monte Carlo-Markov Finance System
Advanced quantitative finance platform with ML integration
"""

__version__ = "2.1.0"
__author__ = "MCMF Development Team"
__license__ = "MIT"

# Core imports
from .monte_carlo_engine import *
from .analytics_engine import *
from .markov_models import *
from .optimization import *
from .ml_models import *
from .crypto_models import *
from .esg_integration import *
from .stress_testing import *
from .real_time_engine import *
from .visualization import *

# Configuration
from .config import settings

# Utilities
from .utils import *

# Version info
VERSION_INFO = {
    'version': __version__,
    'build_date': '2025-08-23',
    'python_version': '>=3.9',
    'features': [
        'Quantum Monte Carlo',
        'ML Transformers',
        'Distributed Computing',
        'Real-time API',
        'ESG Integration',
        'Crypto Models',
        'Stress Testing',
        'Mobile Dashboard'
    ]
}
