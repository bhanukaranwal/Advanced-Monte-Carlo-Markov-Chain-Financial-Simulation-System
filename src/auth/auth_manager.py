"""
Authentication and authorization manager
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config.settings import settings
from ..database.models import User, APIKey
from ..database.database import get_db_session
from ..utils.exceptions import AuthenticationError, AuthorizationError

security = HTTPBearer()

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self):
        self.jwt_secret = settings.security.jwt_secret
        self.jwt_algorithm = settings.security.jwt_algorithm
        self.jwt_expiration = settings.security.jwt_expiration
        
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(seconds=self.jwt_expiration)
            
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
        
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
            
    def get_current_user(self, credentials: HTTPAuthorizationCredentials) -> User:
        """Get current user from token"""
        payload = self.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationError("Invalid token payload")
            
        with get_db_session() as db:
            user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
            
        if not user:
            raise AuthenticationError("User not found")
            
        return user
        
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        with get_db_session() as db:
            user = db.query(User).filter(
                User.username == username,
                User.is_active == True
            ).first()
            
        if not user or not self.verify_password(password, user.password_hash):
            return None
            
        # Update last login
        with get_db_session() as db:
            db.query(User).filter(User.id == user.id).update({
                "last_login": datetime.utcnow()
            })
            
        return user
        
    def create_api_key(self, user_id: str, key_name: str, permissions: List[str] = None) -> str:
        """Create API key for user"""
        api_key = secrets.token_urlsafe(32)
        key_hash = self.hash_password(api_key)
        
        with get_db_session() as db:
            db_api_key = APIKey(
                user_id=user_id,
                key_name=key_name,
                key_hash=key_hash,
                permissions=permissions or []
            )
            db.add(db_api_key)
            
        return api_key
        
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """Verify API key and return user"""
        with get_db_session() as db:
            # Get all active API keys
            api_keys = db.query(APIKey).filter(APIKey.is_active == True).all()
            
        for db_key in api_keys:
            if self.verify_password(api_key, db_key.key_hash):
                # Update last used
                with get_db_session() as db:
                    db.query(APIKey).filter(APIKey.id == db_key.id).update({
                        "last_used": datetime.utcnow()
                    })
                    
                # Get user
                with get_db_session() as db:
                    user = db.query(User).filter(
                        User.id == db_key.user_id,
                        User.is_active == True
                    ).first()
                    
                return user
                
        return None
        
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        if user.is_admin:
            return True
            
        # Check API key permissions if applicable
        # This would need to be enhanced based on your permission model
        return True
        
    def require_permission(self, user: User, permission: str):
        """Require specific permission or raise exception"""
        if not self.check_permission(user, permission):
            raise AuthorizationError(f"Permission required: {permission}")

# Global auth manager instance
auth_manager = AuthManager()
