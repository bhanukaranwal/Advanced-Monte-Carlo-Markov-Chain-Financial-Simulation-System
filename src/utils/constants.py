"""
Application constants and enums
"""

from enum import Enum, IntEnum
from typing import Dict, List

# Application Constants
APP_NAME = "Monte Carlo-Markov Finance System"
APP_VERSION = "1.0.0"
DEFAULT_TIMEZONE = "UTC"

# Simulation Constants
DEFAULT_N_SIMULATIONS = 10000
DEFAULT_N_STEPS = 252
DEFAULT_RANDOM_SEED = 42
MAX_SIMULATIONS = 1000000
MAX_STEPS = 10000

# Financial Constants
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_TRADING_DAY = 6.5
RISK_FREE_RATE_DEFAULT = 0.02

# Market Data Constants
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Database Constants
DEFAULT_DB_POOL_SIZE = 10
DEFAULT_DB_MAX_OVERFLOW = 20
DEFAULT_DB_POOL_TIMEOUT = 30

# Cache Constants
DEFAULT_CACHE_TTL = 3600  # 1 hour
SHORT_CACHE_TTL = 300     # 5 minutes
LONG_CACHE_TTL = 86400    # 24 hours

# API Constants
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 1000
API_VERSION = "v1"

class SimulationType(Enum):
    """Simulation types"""
    GEOMETRIC_BROWNIAN_MOTION = "gbm"
    MULTI_ASSET = "multi_asset"
    PATH_DEPENDENT = "path_dependent"
    QUASI_MONTE_CARLO = "quasi_mc"
    HESTON = "heston"
    JUMP_DIFFUSION = "jump_diffusion"

class OptionType(Enum):
    """Option types"""
    CALL = "call"
    PUT = "put"

class AssetClass(Enum):
    """Asset classes"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"

class MarketRegime(Enum):
    """Market regimes"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class RiskMeasure(Enum):
    """Risk measures"""
    VAR = "var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"

class ValidationStatus(IntEnum):
    """Validation status codes"""
    PASSED = 0
    WARNING = 1
    FAILED = 2

# Market symbols
MAJOR_INDICES = [
    "^GSPC",  # S&P 500
    "^DJI",   # Dow Jones
    "^IXIC",  # NASDAQ
    "^RUT",   # Russell 2000
    "^VIX"    # Volatility Index
]

MAJOR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "BRK-B", "UNH", "JNJ"
]

# Configuration defaults
DEFAULT_CONFIG = {
    "simulation": {
        "n_simulations": DEFAULT_N_SIMULATIONS,
        "n_steps": DEFAULT_N_STEPS,
        "random_seed": DEFAULT_RANDOM_SEED,
        "use_gpu": False,
        "batch_size": 1000
    },
    "risk": {
        "confidence_levels": [0.95, 0.99],
        "lookback_window": 252,
        "var_method": "historical"
    },
    "api": {
        "rate_limit_per_minute": 100,
        "rate_limit_per_hour": 1000,
        "jwt_expiration_hours": 24
    }
}

# Error messages
ERROR_MESSAGES = {
    "simulation_failed": "Monte Carlo simulation failed",
    "invalid_parameters": "Invalid parameters provided",
    "insufficient_data": "Insufficient data for analysis",
    "convergence_failed": "Simulation failed to converge",
    "gpu_unavailable": "GPU acceleration not available",
    "memory_insufficient": "Insufficient memory for operation",
    "authentication_required": "Authentication required",
    "authorization_failed": "Access denied",
    "rate_limit_exceeded": "Rate limit exceeded"
}

# File extensions
SUPPORTED_DATA_FORMATS = [".csv", ".json", ".parquet", ".h5", ".pkl"]
SUPPORTED_REPORT_FORMATS = [".pdf", ".html", ".xlsx"]
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".jpeg", ".svg", ".pdf"]
