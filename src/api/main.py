"""
Main FastAPI application
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from ..config.settings import settings
from ..database.database import create_tables
from ..utils.logging_config import setup_logging
from .middleware import TimingMiddleware, RequestIDMiddleware, ErrorHandlingMiddleware
from .routes import portfolio, simulations, analytics
from .health import router as health_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Monte Carlo-Markov Finance API",
    description="Advanced quantitative finance platform with Monte Carlo simulations",
    version=settings.version,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(portfolio.router, prefix="/api/v1")
app.include_router(simulations.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(f"Starting MCMF API v{settings.version}")
    
    # Create database tables
    try:
        create_tables()
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        
@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down MCMF API")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Monte Carlo-Markov Finance API",
        "version": settings.version,
        "docs": "/api/docs"
    }

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Path {request.url.path} not found",
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers
    )
