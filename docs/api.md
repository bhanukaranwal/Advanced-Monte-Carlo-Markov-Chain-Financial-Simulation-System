# MCMF API Documentation

## Base URL
https://api.mcmf-system.com/api/v1



## Authentication

All API endpoints require authentication using JWT tokens.

### Login
POST /auth/login
Content-Type: application/json

{
"user_id": "demo_user",
"password": "demo_password"
}



Response:
{
"access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
"refresh_token": "uuid-refresh-token",
"token_type": "Bearer",
"expires_in": 86400
}



### Using the Token
Include the token in the Authorization header:
Authorization: Bearer your_access_token_here



## Endpoints

### Monte Carlo Simulations

#### Single Asset Simulation
POST /simulations/monte-carlo
Authorization: Bearer {token}
Content-Type: application/json

{
"n_simulations": 10000,
"n_steps": 252,
"initial_price": 100.0,
"drift": 0.05,
"volatility": 0.2,
"random_seed": 42
}



#### Multi-Asset Simulation
POST /simulations/multi-asset
Authorization: Bearer {token}
Content-Type: application/json

{
"n_simulations": 5000,
"n_steps": 252,
"initial_prices": [100.0, 200.0, 150.0],
"drifts": [0.05, 0.06, 0.04],
"volatilities": [0.2, 0.25, 0.18],
"correlation_matrix": [[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]]
}



### Option Pricing

POST /pricing/options
Authorization: Bearer {token}
Content-Type: application/json

{
"monte_carlo_params": {
"n_simulations": 50000,
"n_steps": 252,
"initial_price": 100.0,
"drift": 0.05,
"volatility": 0.2
},
"option_params": {
"strike": 105.0,
"option_type": "call",
"risk_free_rate": 0.03,
"time_to_maturity": 1.0
}
}



### Risk Analytics

POST /analytics/risk
Authorization: Bearer {token}
Content-Type: application/json

{
"returns_data": [0.001, -0.002, 0.003, ...],
"confidence_levels": [0.95, 0.99],
"method": "historical",
"risk_free_rate": 0.02
}



## WebSocket API

Connect to: `wss://api.mcmf-system.com/socket.io/`

### Authentication
socket.emit('authenticate', {
token: 'your_jwt_token'
});



### Subscribe to Market Data
socket.emit('subscribe_market_data', {
symbols: ['AAPL', 'GOOGL', 'MSFT']
});



### Receive Updates
socket.on('market_data_update', (data) => {
console.log('Price update:', data);
});



## Rate Limits

- **Default**: 100 requests per minute, 1000 per hour
- **Monte Carlo**: 10 requests per minute
- **Multi-Asset**: 5 requests per minute
- **Reports**: 5 requests per minute

## Error Codes

- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `429` - Rate limit exceeded
- `500` - Internal server error
