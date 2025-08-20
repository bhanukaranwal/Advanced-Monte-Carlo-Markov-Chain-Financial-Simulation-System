"""
Rate limiting for API endpoints
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import logging

logger = logging.getLogger(__name__)

# Redis storage for rate limiting
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_user_id_for_rate_limit():
    """Get user ID for rate limiting (if authenticated)"""
    from flask import request
    
    if hasattr(request, 'current_user') and request.current_user:
        return request.current_user.get('user_id', get_remote_address())
    
    return get_remote_address()

# Initialize rate limiter
limiter = Limiter(
    key_func=get_user_id_for_rate_limit,
    storage_uri="redis://localhost:6379",
    default_limits=["1000 per hour", "100 per minute"]
)

# Custom rate limit exceeded handler
@limiter.request_filter
def rate_limit_filter():
    """Filter requests that should not be rate limited"""
    from flask import request
    
    # Don't rate limit health checks
    if request.endpoint == 'health_check':
        return True
        
    return False

def get_rate_limit_status(user_id: str = None) -> dict:
    """Get current rate limit status for user"""
    if not user_id:
        user_id = get_remote_address()
        
    try:
        # This would query the rate limiter's storage
        # For now, return a mock status
        return {
            'user_id': user_id,
            'limits': {
                'per_minute': {'limit': 100, 'remaining': 95, 'reset_at': '2025-01-21T02:00:00Z'},
                'per_hour': {'limit': 1000, 'remaining': 850, 'reset_at': '2025-01-21T02:00:00Z'}
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}")
        return {'error': 'Unable to get rate limit status'}
