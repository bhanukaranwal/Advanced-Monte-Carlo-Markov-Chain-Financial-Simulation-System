"""
Load testing for MCMF API using Locust
"""

from locust import HttpUser, task, between
import json
import random
import string

class MCMFAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login and get authentication token"""
        response = self.client.post("/api/v1/auth/login", json={
            "user_id": "demo_user",
            "password": "demo_password"
        })
        
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.token = None
            self.headers = {}
    
    @task(3)
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/health")
    
    @task(5)
    def monte_carlo_simulation(self):
        """Test Monte Carlo simulation endpoint"""
        if not self.token:
            return
            
        payload = {
            "n_simulations": random.randint(1000, 5000),
            "n_steps": random.randint(100, 500),
            "initial_price": random.uniform(50, 200),
            "drift": random.uniform(0.01, 0.1),
            "volatility": random.uniform(0.1, 0.5)
        }
        
        self.client.post(
            "/api/v1/simulations/monte-carlo",
            json=payload,
            headers=self.headers,
            name="monte_carlo_sim"
        )
    
    @task(2)
    def risk_analysis(self):
        """Test risk analysis endpoint"""
        if not self.token:
            return
            
        # Generate random returns data
        returns = [random.gauss(0.001, 0.02) for _ in range(252)]
        
        payload = {
            "returns_data": returns,
            "confidence_levels": [0.95, 0.99],
            "method": "historical"
        }
        
        self.client.post(
            "/api/v1/analytics/risk",
            json=payload,
            headers=self.headers,
            name="risk_analysis"
        )
    
    @task(1)
    def option_pricing(self):
        """Test option pricing endpoint"""
        if not self.token:
            return
            
        payload = {
            "monte_carlo_params": {
                "n_simulations": 10000,
                "n_steps": 252,
                "initial_price": 100.0,
                "drift": 0.05,
                "volatility": 0.2
            },
            "option_params": {
                "strike": random.uniform(80, 120),
                "option_type": random.choice(["call", "put"]),
                "risk_free_rate": 0.03,
                "time_to_maturity": random.uniform(0.1, 2.0)
            }
        }
        
        self.client.post(
            "/api/v1/pricing/options",
            json=payload,
            headers=self.headers,
            name="option_pricing"
        )

class MCMFWebSocketUser(HttpUser):
    """WebSocket load testing user"""
    wait_time = between(2, 5)
    
    def on_start(self):
        """Connect to WebSocket"""
        # WebSocket connection would be implemented here
        # This is a placeholder for actual WebSocket testing
        pass
    
    @task
    def websocket_subscribe(self):
        """Test WebSocket subscription"""
        # WebSocket subscription testing
        pass
