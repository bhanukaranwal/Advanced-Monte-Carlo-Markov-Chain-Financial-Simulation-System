"""
Custom exceptions for MCMF system
"""

class MCMFException(Exception):
    """Base exception for MCMF system"""
    pass

class SimulationError(MCMFException):
    """Exception raised for simulation-related errors"""
    pass

class ValidationError(MCMFException):
    """Exception raised for data validation errors"""
    pass

class ConfigurationError(MCMFException):
    """Exception raised for configuration errors"""
    pass

class DatabaseError(MCMFException):
    """Exception raised for database-related errors"""
    pass

class APIError(MCMFException):
    """Exception raised for API-related errors"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code

class AuthenticationError(APIError):
    """Exception raised for authentication errors"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401)

class AuthorizationError(APIError):
    """Exception raised for authorization errors"""
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, 403)

class RateLimitError(APIError):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429)

class GPUError(MCMFException):
    """Exception raised for GPU-related errors"""
    pass

class MemoryError(MCMFException):
    """Exception raised for memory-related errors"""
    pass

class ConvergenceError(SimulationError):
    """Exception raised when simulation doesn't converge"""
    pass

class MarketDataError(MCMFException):
    """Exception raised for market data errors"""
    pass

class BacktestError(MCMFException):
    """Exception raised for backtesting errors"""
    pass

class ReportGenerationError(MCMFException):
    """Exception raised for report generation errors"""
    pass
