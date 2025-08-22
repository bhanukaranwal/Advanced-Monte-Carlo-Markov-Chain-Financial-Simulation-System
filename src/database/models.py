"""
Database models for MCMF system
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

Base = declarative_base()

class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user")
    simulations = relationship("Simulation", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")

class Portfolio(Base):
    """Portfolio model"""
    __tablename__ = "portfolios"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    total_value = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    positions = relationship("Position", back_populates="portfolio")
    risk_metrics = relationship("RiskMetric", back_populates="portfolio")

class Position(Base):
    """Portfolio position model"""
    __tablename__ = "positions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    asset_type = Column(String(20), nullable=False)  # equity, bond, crypto, etc.
    quantity = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    weight = Column(Float)  # Portfolio weight
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")

class Simulation(Base):
    """Monte Carlo simulation model"""
    __tablename__ = "simulations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(String, ForeignKey("portfolios.id"))
    simulation_type = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    n_simulations = Column(Integer, nullable=False)
    n_steps = Column(Integer, nullable=False)
    results = Column(JSON)
    execution_time = Column(Float)
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="simulations")

class RiskMetric(Base):
    """Risk metrics model"""
    __tablename__ = "risk_metrics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    calculation_date = Column(DateTime, default=func.now())
    var_95 = Column(Float)
    var_99 = Column(Float)
    expected_shortfall_95 = Column(Float)
    expected_shortfall_99 = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    beta = Column(Float)
    alpha = Column(Float)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="risk_metrics")

class ESGScore(Base):
    """ESG scores model"""
    __tablename__ = "esg_scores"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False)
    provider = Column(String(50), nullable=False)  # MSCI, Sustainalytics, etc.
    environmental_score = Column(Float)
    social_score = Column(Float)
    governance_score = Column(Float)
    overall_score = Column(Float)
    carbon_intensity = Column(Float)
    data_quality = Column(Float)
    score_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())

class StressTest(Base):
    """Stress test results model"""
    __tablename__ = "stress_tests"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    scenario_name = Column(String(100), nullable=False)
    scenario_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    portfolio_impact = Column(Float)
    max_drawdown = Column(Float)
    recovery_time = Column(Integer)
    results = Column(JSON)
    created_at = Column(DateTime, default=func.now())

class APIKey(Base):
    """API keys model"""
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False)
    permissions = Column(JSON, default=list)  # List of permissions
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="api_keys")

class AuditLog(Base):
    """Audit log model"""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    created_at = Column(DateTime, default=func.now())

class MarketData(Base):
    """Market data model"""
    __tablename__ = "market_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    adjusted_close = Column(Float)
    
    # Indexes
    __table_args__ = (
        {"extend_existing": True}
    )
