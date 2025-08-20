"""
Tests for API endpoints
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from flask import Flask
import jwt
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from api.rest_api import app
from api.auth import generate_token, verify_token

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret'
    
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_token():
    """Generate test authentication token"""
    return generate_token('test_user', ['read', 'write'])

class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_login_success(self, client):
        """Test successful login"""
        response = client.post('/api/v1/auth/login', json={
            'user_id': 'demo_user',
            'password': 'demo_password'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'access_token' in data
        assert 'refresh_token' in data
        assert data['user']['user_id'] == 'demo_user'
        
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        response = client.post('/api/v1/auth/login', json={
            'user_id': 'nonexistent',
            'password': 'wrong'
        })
        
        assert response.status_code == 401
        
    def test_token_verification(self, client, auth_token):
        """Test token verification endpoint"""
        response = client.get('/api/v1/auth/verify', headers={
            'Authorization': f'Bearer {auth_token}'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['valid'] is True

class TestSimulationEndpoints:
    """Test simulation endpoints"""
    
    @patch('api.rest_api.GeometricBrownianMotionEngine')
    def test_monte_carlo_simulation(self, mock_engine, client, auth_token):
        """Test Monte Carlo simulation endpoint"""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.simulate_paths.return_value = [[100, 101, 102], [100, 99, 98]]
        mock_engine.return_value = mock_instance
        
        response = client.post('/api/v1/simulations/monte-carlo', 
                             headers={'Authorization': f'Bearer {auth_token}'},
                             json={
                                 'n_simulations': 1000,
                                 'n_steps': 10,
                                 'initial_price': 100.0,
                                 'drift': 0.05,
                                 'volatility': 0.2
                             })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'simulation_id' in data
        assert 'statistics' in data
        
    def test_monte_carlo_validation_error(self, client, auth_token):
        """Test Monte Carlo with validation errors"""
        response = client.post('/api/v1/simulations/monte-carlo',
                             headers={'Authorization': f'Bearer {auth_token}'},
                             json={
                                 'n_simulations': -1,  # Invalid
                                 'volatility': 'invalid'  # Invalid type
                             })
        
        assert response.status_code == 400

class TestRateLimiting:
    """Test rate limiting"""
    
    def test_rate_limit_exceeded(self, client, auth_token):
        """Test rate limit enforcement"""
        # This would require mocking the rate limiter
        # In a real test, you'd make many requests quickly
        pass

class TestWebSocketAPI:
    """Test WebSocket API"""
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        # WebSocket testing requires special setup
        # This is a placeholder for actual WebSocket tests
        pass
