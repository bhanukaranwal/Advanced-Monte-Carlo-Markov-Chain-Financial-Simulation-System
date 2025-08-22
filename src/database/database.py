"""
Database connection and session management
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from ..config.settings import settings
from .models import Base

logger = logging.getLogger(__name__)

# Database engine
engine = create_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,
    echo=settings.debug
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database session"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.engine = engine
        
    def health_check(self) -> bool:
        """Check database health"""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
            
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        return {
            "url": settings.database.url.replace(settings.database.password, "***"),
            "pool_size": settings.database.pool_size,
            "max_overflow": settings.database.max_overflow
        }

# Global database manager instance
db_manager = DatabaseManager()
