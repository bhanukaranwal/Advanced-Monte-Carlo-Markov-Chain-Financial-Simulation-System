"""
Logging configuration for MCMF system
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from ..config.settings import settings

def setup_logging() -> None:
    """Setup logging configuration"""
    
    log_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": settings.log_dir / "mcmf.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": settings.log_dir / "mcmf_errors.log",
                "maxBytes": 10485760,
                "backupCount": 5
            },
            "json_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": settings.log_dir / "mcmf.json.log",
                "maxBytes": 10485760,
                "backupCount": 5
            }
        },
        "loggers": {
            "mcmf": {
                "level": settings.logging_level,
                "handlers": ["console", "file", "error_file", "json_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"]
        }
    }
    
    logging.config.dictConfig(log_config)

class MCMFLogger:
    """Custom logger for MCMF with structured logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"mcmf.{name}")
        
    def log_simulation_start(self, simulation_type: str, params: Dict[str, Any]):
        """Log simulation start"""
        self.logger.info(
            "Simulation started",
            extra={
                "event_type": "simulation_start",
                "simulation_type": simulation_type,
                "parameters": params,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    def log_simulation_complete(self, simulation_type: str, duration: float, results: Dict[str, Any]):
        """Log simulation completion"""
        self.logger.info(
            "Simulation completed",
            extra={
                "event_type": "simulation_complete",
                "simulation_type": simulation_type,
                "duration_seconds": duration,
                "results_summary": results,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    def log_api_request(self, endpoint: str, method: str, user_id: str = None):
        """Log API request"""
        self.logger.info(
            f"API request: {method} {endpoint}",
            extra={
                "event_type": "api_request",
                "endpoint": endpoint,
                "method": method,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(
            f"Error occurred: {str(error)}",
            extra={
                "event_type": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            },
            exc_info=True
        )
