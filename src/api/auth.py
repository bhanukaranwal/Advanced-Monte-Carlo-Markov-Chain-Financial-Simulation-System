"""
Authentication and authorization for API endpoints
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, current_app, Blueprint
import redis
import uuid
import logging

logger = logging.getLogger(__name__)

# Redis client for session storage
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Blueprint for auth routes
auth_bp = Blueprint('auth', __name__)

# Configuration
JWT_SECRET_KEY = 'mcmf-jwt-secret-change-in-production'
JWT_EXPIRATION_HOURS = 24
REFRESH_TOKEN_EXPIRATION_DAYS = 30

# Mock user database (in production, use proper database)
USERS_DB = {
    'demo_user': {
        'user_id': 'demo_user',
        'password_hash': bcrypt.hashpw('demo_password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        'email': 'demo@mcmf-system.com',
        'permissions': ['read', 'write', 'admin'],
        'created_at': datetime.utcnow().isoformat()
    },
    'api_user': {
        'user_id': 'api_user', 
        'password_hash': bcrypt.hashpw('api_password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
        'email': 'api@mcmf-system.com',
        'permissions': ['read', 'write'],
        'created_at': datetime.utcnow().isoformat()
    }
}

def generate_token(user_id: str, permissions: list) -> str:
    """Generate JWT token"""
    payload = {
        'user_id': user_id,
        'permissions': permissions,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow(),
        'jti': str(uuid.uuid4())  # JWT ID for token revocation
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def generate_refresh_token(user_id: str) -> str:
    """Generate refresh token"""
    refresh_token = str(uuid.uuid4())
    
    # Store in Redis with expiration
    redis_client.setex(
        f"refresh_token:{refresh_token}",
        timedelta(days=REFRESH_TOKEN_EXPIRATION_DAYS),
        user_id
    )
    
    return refresh_token

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        
        # Check if token is blacklisted
        jti = payload.get('jti')
        if jti and redis_client.get(f"blacklist:{jti}"):
            return None
            
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None

def blacklist_token(token: str):
    """Blacklist a token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'], options={"verify_exp": False})
        jti = payload.get('jti')
        
        if jti:
            # Calculate remaining time until expiration
            exp = payload.get('exp', 0)
            exp_time = datetime.fromtimestamp(exp)
            remaining_time = max(0, (exp_time - datetime.utcnow()).total_seconds())
            
            # Add to blacklist with expiration
            redis_client.setex(f"blacklist:{jti}", int(remaining_time), "true")
            
    except Exception as e:
        logger.error(f"Error blacklisting token: {e}")

def get_user_from_token(token: str) -> dict:
    """Get user information from token"""
    payload = verify_token(token)
    if not payload:
        return None
        
    user_id = payload.get('user_id')
    if user_id in USERS_DB:
        user_info = USERS_DB[user_id].copy()
        user_info['permissions'] = payload.get('permissions', [])
        return user_info
        
    return None

# Decorators for route protection
def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({'error': 'Invalid authorization header format'}), 401
                
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
            
        # Verify token
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Token is invalid or expired'}), 401
            
        # Add user info to request context
        request.current_user = payload
        
        return f(*args, **kwargs)
        
    return decorated

def token_required_ws(f):
    """Decorator for WebSocket token verification"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # This would be implemented for WebSocket authentication
        # For now, just pass through
        return f(*args, **kwargs)
        
    return decorated

def permission_required(required_permission: str):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(request, 'current_user'):
                return jsonify({'error': 'Authentication required'}), 401
                
            user_permissions = request.current_user.get('permissions', [])
            
            if required_permission not in user_permissions and 'admin' not in user_permissions:
                return jsonify({'error': 'Insufficient permissions'}), 403
                
            return f(*args, **kwargs)
            
        return decorated
    return decorator

# Authentication routes
@auth_bp.route('/api/v1/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        
        if not data or not data.get('user_id') or not data.get('password'):
            return jsonify({'error': 'User ID and password required'}), 400
            
        user_id = data['user_id']
        password = data['password']
        
        # Check if user exists
        if user_id not in USERS_DB:
            return jsonify({'error': 'Invalid credentials'}), 401
            
        user = USERS_DB[user_id]
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            return jsonify({'error': 'Invalid credentials'}), 401
            
        # Generate tokens
        access_token = generate_token(user_id, user['permissions'])
        refresh_token = generate_refresh_token(user_id)
        
        # Log successful login
        logger.info(f"User {user_id} logged in successfully")
        
        return jsonify({
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': JWT_EXPIRATION_HOURS * 3600,
            'user': {
                'user_id': user_id,
                'email': user['email'],
                'permissions': user['permissions']
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@auth_bp.route('/api/v1/auth/refresh', methods=['POST'])
def refresh_token():
    """Refresh access token"""
    try:
        data = request.get_json()
        
        if not data or not data.get('refresh_token'):
            return jsonify({'error': 'Refresh token required'}), 400
            
        refresh_token = data['refresh_token']
        
        # Check refresh token
        user_id = redis_client.get(f"refresh_token:{refresh_token}")
        
        if not user_id or user_id not in USERS_DB:
            return jsonify({'error': 'Invalid refresh token'}), 401
            
        user = USERS_DB[user_id]
        
        # Generate new access token
        access_token = generate_token(user_id, user['permissions'])
        
        return jsonify({
            'access_token': access_token,
            'token_type': 'Bearer',
            'expires_in': JWT_EXPIRATION_HOURS * 3600
        })
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return jsonify({'error': 'Token refresh failed'}), 500

@auth_bp.route('/api/v1/auth/logout', methods=['POST'])
@token_required
def logout():
    """User logout endpoint"""
    try:
        # Get token from header
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.split(' ')[1] if len(auth_header.split(' ')) > 1 else None
        
        if token:
            # Blacklist the token
            blacklist_token(token)
            
        # Invalidate refresh token if provided
        data = request.get_json() or {}
        refresh_token = data.get('refresh_token')
        
        if refresh_token:
            redis_client.delete(f"refresh_token:{refresh_token}")
            
        logger.info(f"User {request.current_user['user_id']} logged out")
        
        return jsonify({'message': 'Logged out successfully'})
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'error': 'Logout failed'}), 500

@auth_bp.route('/api/v1/auth/verify', methods=['GET'])
@token_required
def verify_token_endpoint():
    """Verify token validity"""
    return jsonify({
        'valid': True,
        'user': {
            'user_id': request.current_user['user_id'],
            'permissions': request.current_user['permissions']
        },
        'expires_at': datetime.fromtimestamp(request.current_user['exp']).isoformat()
    })
