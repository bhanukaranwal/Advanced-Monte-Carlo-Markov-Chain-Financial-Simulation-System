"""
Main application entry point for MCMF API
"""

from flask import Flask
from flask_cors import CORS
import logging
import os

from .rest_api import app as rest_app
from .websocket_api import socketio_app
from .auth import auth_bp
from .rate_limiting import limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_app(config_name='development'):
    """Application factory"""
    
    # Register authentication blueprint
    rest_app.register_blueprint(auth_bp)
    
    # Configure CORS
    CORS(rest_app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:8501"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Environment-specific configuration
    if config_name == 'production':
        rest_app.config.update({
            'DEBUG': False,
            'TESTING': False,
            'SECRET_KEY': os.environ.get('SECRET_KEY', 'prod-secret-key'),
            'SQLALCHEMY_DATABASE_URI': os.environ.get('DATABASE_URL'),
            'REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
        })
    elif config_name == 'testing':
        rest_app.config.update({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
            'REDIS_URL': 'redis://localhost:6379/1'
        })
    else:  # development
        rest_app.config.update({
            'DEBUG': True,
            'SECRET_KEY': 'dev-secret-key',
            'SQLALCHEMY_DATABASE_URI': 'postgresql://mcmf_user:password@localhost:5432/mcmf_dev',
            'REDIS_URL': 'redis://localhost:6379/0'
        })
    
    logger.info(f"Application created with config: {config_name}")
    
    return rest_app

# Create the application
app = create_app(os.environ.get('FLASK_ENV', 'development'))

if __name__ == '__main__':
    # For development
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Production deployment would use gunicorn or similar
        app.run(host='0.0.0.0', port=5000)
