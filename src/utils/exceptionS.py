"""
Custom exception classes for MCMF system
"""

class MCMFException(Exception):
    """Base exception for MCMF system"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ValidationError(MCMFException):
    """Data validation errors"""
    pass

class SimulationError(MCMFException):
    """Monte Carlo simulation errors"""
    pass

class ConvergenceError(SimulationError):
    """Convergence failure in simulations"""
    pass

class GPUError(MCMFException):
    """GPU computation errors"""
    pass

class DatabaseError(MCMFException):
    """Database operation errors"""
    pass

class AuthenticationError(MCMFException):
    """Authentication failures"""
    pass

class AuthorizationError(MCMFException):
    """Authorization failures"""
    pass

class RateLimitError(MCMFException):
    """Rate limiting errors"""
    pass

class ESGDataError(MCMFException):
    """ESG data retrieval/processing errors"""
    pass

class CryptoModelError(MCMFException):
    """Cryptocurrency modeling errors"""
    pass

class StressTestError(MCMFException):
    """Stress testing errors"""
    pass
