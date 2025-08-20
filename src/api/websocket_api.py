"""
WebSocket API for real-time data streaming and live updates
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import redis
import logging

# MCMF imports
from real_time_engine.stream_processor import StreamProcessor, MarketTick
from real_time_engine.kalman_filters import TrendFollowingKalman
from analytics_engine.risk_analytics import RiskAnalytics
from .auth import token_required_ws, get_user_from_token

logger = logging.getLogger(__name__)

# Initialize Flask-SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mcmf-websocket-secret'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Redis client for pub/sub
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Global storage for active connections and subscriptions
active_connections = {}
user_subscriptions = {}
stream_processors = {}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    connection_id = str(uuid.uuid4())
    active_connections[request.sid] = {
        'connection_id': connection_id,
        'connected_at': datetime.utcnow(),
        'user_id': None,
        'subscriptions': set()
    }
    
    logger.info(f"Client connected: {request.sid}")
    emit('connection_established', {
        'connection_id': connection_id,
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'connected'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    if request.sid in active_connections:
        connection_info = active_connections[request.sid]
        
        # Leave all rooms
        for subscription in connection_info['subscriptions']:
            leave_room(subscription)
            
        # Clean up
        del active_connections[request.sid]
        
        logger.info(f"Client disconnected: {request.sid}")

@socketio.on('authenticate')
def handle_authentication(data):
    """Authenticate WebSocket connection"""
    try:
        token = data.get('token')
        if not token:
            emit('auth_error', {'error': 'Token required'})
            return
            
        user = get_user_from_token(token)
        if not user:
            emit('auth_error', {'error': 'Invalid token'})
            return
            
        # Update connection info
        if request.sid in active_connections:
            active_connections[request.sid]['user_id'] = user['user_id']
            
        emit('authenticated', {
            'user_id': user['user_id'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info(f"User {user['user_id']} authenticated on connection {request.sid}")
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        emit('auth_error', {'error': str(e)})

@socketio.on('subscribe_market_data')
def handle_market_data_subscription(data):
    """Subscribe to real-time market data"""
    try:
        symbols = data.get('symbols', [])
        if not symbols:
            emit('subscription_error', {'error': 'No symbols provided'})
            return
            
        connection_info = active_connections.get(request.sid)
        if not connection_info or not connection_info['user_id']:
            emit('subscription_error', {'error': 'Authentication required'})
            return
            
        # Join market data room for each symbol
        for symbol in symbols:
            room_name = f"market_data_{symbol}"
            join_room(room_name)
            connection_info['subscriptions'].add(room_name)
            
        emit('subscription_confirmed', {
            'type': 'market_data',
            'symbols': symbols,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info(f"User {connection_info['user_id']} subscribed to market data: {symbols}")
        
    except Exception as e:
        logger.error(f"Market data subscription error: {e}")
        emit('subscription_error', {'error': str(e)})

@socketio.on('subscribe_risk_updates')
def handle_risk_subscription(data):
    """Subscribe to real-time risk metric updates"""
    try:
        portfolio_id = data.get('portfolio_id')
        if not portfolio_id:
            emit('subscription_error', {'error': 'Portfolio ID required'})
            return
            
        connection_info = active_connections.get(request.sid)
        if not connection_info or not connection_info['user_id']:
            emit('subscription_error', {'error': 'Authentication required'})
            return
            
        # Join risk updates room
        room_name = f"risk_updates_{portfolio_id}"
        join_room(room_name)
        connection_info['subscriptions'].add(room_name)
        
        emit('subscription_confirmed', {
            'type': 'risk_updates',
            'portfolio_id': portfolio_id,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logger.info(f"User {connection_info['user_id']} subscribed to risk updates for portfolio {portfolio_id}")
        
    except Exception as e:
        logger.error(f"Risk subscription error: {e}")
        emit('subscription_error', {'error': str(e)})

@socketio.on('start_monte_carlo_stream')
def handle_monte_carlo_stream(data):
    """Start streaming Monte Carlo simulation results"""
    try:
        params = data.get('parameters', {})
        connection_info = active_connections.get(request.sid)
        
        if not connection_info or not connection_info['user_id']:
            emit('stream_error', {'error': 'Authentication required'})
            return
            
        # Start Monte Carlo streaming (simplified)
        stream_id = str(uuid.uuid4())
        
        # This would typically be handled by a background task
        socketio.start_background_task(
            monte_carlo_streaming_task, 
            request.sid, 
            stream_id, 
            params
        )
        
        emit('stream_started', {
            'stream_id': stream_id,
            'type': 'monte_carlo',
            'parameters': params,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Monte Carlo stream error: {e}")
        emit('stream_error', {'error': str(e)})

def monte_carlo_streaming_task(session_id, stream_id, params):
    """Background task for streaming Monte Carlo results"""
    try:
        from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
        
        # Simulate streaming by running smaller batches
        total_sims = params.get('n_simulations', 10000)
        batch_size = 1000
        
        engine = GeometricBrownianMotionEngine(
            n_simulations=batch_size,
            n_steps=params.get('n_steps', 252),
            initial_price=params.get('initial_price', 100),
            drift=params.get('drift', 0.05),
            volatility=params.get('volatility', 0.2)
        )
        
        completed_sims = 0
        all_final_prices = []
        
        while completed_sims < total_sims:
            # Run batch
            paths = engine.simulate_paths()
            final_prices = paths[:, -1]
            all_final_prices.extend(final_prices.tolist())
            
            completed_sims += batch_size
            progress = min(completed_sims / total_sims * 100, 100)
            
            # Calculate running statistics
            running_stats = {
                'mean': float(np.mean(all_final_prices)),
                'std': float(np.std(all_final_prices)),
                'completed_simulations': completed_sims,
                'progress': progress
            }
            
            # Emit update
            socketio.emit('monte_carlo_update', {
                'stream_id': stream_id,
                'statistics': running_stats,
                'timestamp': datetime.utcnow().isoformat()
            }, room=session_id)
            
            socketio.sleep(0.1)  # Small delay
            
        # Emit completion
        socketio.emit('monte_carlo_complete', {
            'stream_id': stream_id,
            'final_statistics': running_stats,
            'timestamp': datetime.utcnow().isoformat()
        }, room=session_id)
        
    except Exception as e:
        logger.error(f"Monte Carlo streaming task error: {e}")
        socketio.emit('stream_error', {
            'stream_id': stream_id,
            'error': str(e)
        }, room=session_id)

@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Unsubscribe from updates"""
    try:
        subscription_type = data.get('type')
        identifier = data.get('identifier')  # symbol, portfolio_id, etc.
        
        connection_info = active_connections.get(request.sid)
        if not connection_info:
            return
            
        # Determine room name based on subscription type
        if subscription_type == 'market_data':
            room_name = f"market_data_{identifier}"
        elif subscription_type == 'risk_updates':
            room_name = f"risk_updates_{identifier}"
        else:
            emit('unsubscribe_error', {'error': 'Unknown subscription type'})
            return
            
        # Leave room
        leave_room(room_name)
        connection_info['subscriptions'].discard(room_name)
        
        emit('unsubscribed', {
            'type': subscription_type,
            'identifier': identifier,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Unsubscribe error: {e}")
        emit('unsubscribe_error', {'error': str(e)})

# Background tasks for broadcasting updates
def market_data_broadcaster():
    """Broadcast market data updates to subscribed clients"""
    while True:
        try:
            # This would typically read from a message queue or database
            # For demonstration, we'll generate mock data
            
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            
            for symbol in symbols:
                # Generate mock price update
                base_price = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300, 'AMZN': 3200, 'TSLA': 800}[symbol]
                price_change = np.random.normal(0, base_price * 0.002)
                current_price = base_price + price_change
                
                market_update = {
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'change': round(price_change, 2),
                    'change_percent': round(price_change / base_price * 100, 3),
                    'volume': np.random.randint(100000, 1000000),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Broadcast to subscribed clients
                socketio.emit('market_data_update', market_update, 
                            room=f"market_data_{symbol}")
                            
            socketio.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Market data broadcaster error: {e}")
            socketio.sleep(5)

def risk_metrics_broadcaster():
    """Broadcast risk metric updates to subscribed clients"""
    while True:
        try:
            # Mock portfolio IDs
            portfolio_ids = ['portfolio_1', 'portfolio_2', 'portfolio_3']
            
            for portfolio_id in portfolio_ids:
                # Generate mock risk metrics
                risk_update = {
                    'portfolio_id': portfolio_id,
                    'var_95': round(np.random.uniform(-0.05, -0.01), 4),
                    'expected_shortfall': round(np.random.uniform(-0.08, -0.02), 4),
                    'volatility': round(np.random.uniform(0.1, 0.3), 4),
                    'sharpe_ratio': round(np.random.uniform(0.5, 2.0), 3),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Broadcast to subscribed clients
                socketio.emit('risk_update', risk_update,
                            room=f"risk_updates_{portfolio_id}")
                            
            socketio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Risk metrics broadcaster error: {e}")
            socketio.sleep(30)

# Start background tasks
socketio.start_background_task(market_data_broadcaster)
socketio.start_background_task(risk_metrics_broadcaster)

# Create the SocketIO app
socketio_app = socketio

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
