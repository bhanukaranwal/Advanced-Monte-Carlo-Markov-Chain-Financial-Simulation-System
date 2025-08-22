"""
Configuration management for MCMF system
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import BaseSettings, Field, validator
from pathlib import Path

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="mcmf_db", env="DB_NAME")
    user: str = Field(default="mcmf_user", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

class RedisSettings(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

class APISettings(BaseSettings):
    """API configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=4, env="API_WORKERS")
    max_request_size: int = Field(default=50 * 1024 * 1024, env="API_MAX_REQUEST_SIZE")  # 50MB
    cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")
    rate_limit: str = Field(default="1000/minute", env="API_RATE_LIMIT")

class SecuritySettings(BaseSettings):
    """Security configuration"""
    jwt_secret: str = Field(default="", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration: int = Field(default=3600, env="JWT_EXPIRATION")  # 1 hour
    encryption_key: str = Field(default="", env="ENCRYPTION_KEY")
    session_timeout: int = Field(default=1800, env="SESSION_TIMEOUT")  # 30 minutes
    max_login_attempts: int = Field(default=5, env="MAX_LOGIN_ATTEMPTS")
    password_min_length: int = Field(default=8, env="PASSWORD_MIN_LENGTH")
    
    @validator('jwt_secret')
    def validate_jwt_secret(cls, v):
        if not v:
            raise ValueError("JWT_SECRET must be set")
        return v

class MCMFSettings(BaseSettings):
    """Main MCMF system settings"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Project info
    project_name: str = "Monte Carlo-Markov Finance System"
    version: str = "2.1.0"
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    log_dir: Path = base_dir / "logs"
    data_dir: Path = base_dir / "data"
    temp_dir: Path = base_dir / "temp"
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    
    # Monte Carlo settings
    default_simulations: int = Field(default=10000, env="DEFAULT_SIMULATIONS")
    max_simulations: int = Field(default=10000000, env="MAX_SIMULATIONS")
    gpu_enabled: bool = Field(default=True, env="GPU_ENABLED")
    distributed_enabled: bool = Field(default=True, env="DISTRIBUTED_ENABLED")
    
    # Monitoring
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    tracing_enabled: bool = Field(default=True, env="TRACING_ENABLED")
    logging_level: str = Field(default="INFO", env="LOGGING_LEVEL")
    
    # External services
    email_backend: str = Field(default="smtp", env="EMAIL_BACKEND")
    smtp_host: str = Field(default="localhost", env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: str = Field(default="", env="SMTP_USER")
    smtp_password: str = Field(default="", env="SMTP_PASSWORD")
    
    # Cloud settings
    cloud_provider: str = Field(default="aws", env="CLOUD_PROVIDER")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    s3_bucket: str = Field(default="", env="S3_BUCKET")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories
        self.log_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

# Global settings instance
settings = MCMFSettings()
