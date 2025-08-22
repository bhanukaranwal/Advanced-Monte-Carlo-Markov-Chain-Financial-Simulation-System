"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import redis
import time
from typing import Dict, Any

from ..database.database import get_db, db_manager
from ..config.settings import settings
from ..utils.exceptions import MCMFException

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.version,
        "environment": settings.environment
    }

@router.get("/detailed")
async def detailed_health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Detailed health check including dependencies"""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.version,
        "environment": settings.environment,
        "checks": {}
    }
    
    # Database check
    try:
        db_healthy = db_manager.health_check()
        health_status["checks"]["database"] = {
            "status": "healthy" if db_healthy else "unhealthy",
            "details": db_manager.get_connection_info() if db_healthy else "Connection failed"
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        
    # Redis check
    try:
        r = redis.from_url(settings.redis.url)
        r.ping()
        health_status["checks"]["redis"] = {
            "status": "healthy",
            "details": {
                "host": settings.redis.host,
                "port": settings.redis.port
            }
        }
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        
    # GPU check (if enabled)
    if settings.gpu_enabled:
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            health_status["checks"]["gpu"] = {
                "status": "healthy",
                "details": {
                    "device_count": cp.cuda.runtime.getDeviceCount(),
                    "current_device": cp.cuda.Device().id
                }
            }
        except Exception as e:
            health_status["checks"]["gpu"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    
    # Check if any component is unhealthy
    unhealthy_components = [
        name for name, check in health_status["checks"].items()
        if check["status"] == "unhealthy"
    ]
    
    if unhealthy_components:
        health_status["status"] = "degraded"
        health_status["unhealthy_components"] = unhealthy_components
        
    return health_status

@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Readiness check for Kubernetes"""
    
    # Check critical dependencies
    try:
        # Database must be available
        if not db_manager.health_check():
            raise MCMFException("Database not available")
            
        # Redis should be available
        r = redis.from_url(settings.redis.url)
        r.ping()
        
        return {"status": "ready"}
        
    except Exception as e:
        return {"status": "not ready", "error": str(e)}

@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Liveness check for Kubernetes"""
    return {"status": "alive"}
