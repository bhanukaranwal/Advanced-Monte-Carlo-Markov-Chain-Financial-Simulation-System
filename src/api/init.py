"""
API Module for Monte Carlo-Markov Finance System

RESTful API endpoints and WebSocket interfaces for accessing all system functionality
"""

from .rest_api import app as rest_app
from .websocket_api import socketio_app
from .auth import auth_bp
from .rate_limiting import limiter

__version__ = "1.0.0"

__all__ = [
    "rest_app",
    "socketio_app", 
    "auth_bp",
    "limiter"
]
