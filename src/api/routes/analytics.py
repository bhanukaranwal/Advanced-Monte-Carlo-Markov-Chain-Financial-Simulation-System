"""
Analytics API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import numpy as np
import logging

from ...database.database import get_db
from ...database.models import Portfolio, Position, User, RiskMetric
from ...auth.auth_manager import auth_manager, security
from ...analytics_engine.risk_analytics import RiskAnalytics
from ...analytics_engine.performance_analytics import PerformanceAnalytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Pydantic models
class RiskAnalysisRequest(BaseModel):
    portfolio_id: str
    confidence_levels: List[float] = [0.95, 0.99]
    risk_free_rate: float = 0.02
    lookback_days: int = 252

class PerformanceAnalysisRequest(BaseModel):
    portfolio_id: str
    benchmark_symbol: str = "SPY"
    start_date: str
    end_date: str

@router.post("/risk/{portfolio_id}", response_model=Dict[str, Any])
async def calculate_portfolio_risk(
    portfolio_id: str,
    request: RiskAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Calculate comprehensive portfolio risk metrics"""
    
    try:
        # Verify portfolio ownership
        portfolio = db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id,
            Portfolio.is_active == True
        ).first()
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        # Get portfolio positions
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        
        if not positions:
            raise HTTPException(status_code=400, detail="Portfolio has no positions")
            
        # Mock historical returns for demonstration
        # In practice, you would fetch real market data
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.001, 0.02, request.lookback_days)  # Daily returns
        
        # Calculate risk metrics
        risk_analytics = RiskAnalytics(confidence_levels=request.confidence_levels)
        risk_measures = risk_analytics.calculate_comprehensive_risk_measures(
            returns, request.risk_free_rate
        )
        
        # Store in database
        risk_metric = RiskMetric(
            portfolio_id=portfolio_id,
            var_95=risk_measures.var_95,
            var_99=risk_measures.var_99,
            expected_shortfall_95=risk_measures.expected_shortfall_95,
            expected_shortfall_99=risk_measures.expected_shortfall_99,
            volatility=risk_measures.volatility,
            sharpe_ratio=risk_measures.sharpe_ratio,
            max_drawdown=risk_measures.maximum_drawdown
        )
        
        db.add(risk_metric)
        db.commit()
        
        return {
            "portfolio_id": portfolio_id,
            "risk_metrics": {
                "var_95": risk_measures.var_95,
                "var_99": risk_measures.var_99,
                "expected_shortfall_95": risk_measures.expected_shortfall_95,
                "expected_shortfall_99": risk_measures.expected_shortfall_99,
                "volatility": risk_measures.volatility,
                "sharpe_ratio": risk_measures.sharpe_ratio,
                "sortino_ratio": risk_measures.sortino_ratio,
                "calmar_ratio": risk_measures.calmar_ratio,
                "maximum_drawdown": risk_measures.maximum_drawdown
            },
            "calculation_date": risk_metric.calculation_date,
            "parameters": {
                "lookback_days": request.lookback_days,
                "risk_free_rate": request.risk_free_rate,
                "confidence_levels": request.confidence_levels
            }
        }
        
    except Exception as e:
        logger.error(f"Risk analysis failed for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Risk analysis failed")

@router.post("/performance/{portfolio_id}", response_model=Dict[str, Any])
async def calculate_portfolio_performance(
    portfolio_id: str,
    request: PerformanceAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Calculate portfolio performance metrics"""
    
    try:
        # Verify portfolio ownership
        portfolio = db.query(Portfolio).filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == current_user.id,
            Portfolio.is_active == True
        ).first()
        
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
            
        # Mock portfolio and benchmark returns
        np.random.seed(42)
        n_days = 252  # 1 year of daily returns
        portfolio_returns = np.random.normal(0.0008, 0.015, n_days)  # Portfolio returns
        benchmark_returns = np.random.normal(0.0005, 0.012, n_days)   # Benchmark returns
        
        # Calculate performance metrics
        perf_analytics = PerformanceAnalytics()
        performance_summary = perf_analytics.performance_summary(
            portfolio_returns, 
            benchmark_returns
        )
        
        return {
            "portfolio_id": portfolio_id,
            "performance_metrics": performance_summary,
            "benchmark_symbol": request.benchmark_symbol,
            "period": {
                "start_date": request.start_date,
                "end_date": request.end_date
            },
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance analysis failed for portfolio {portfolio_id}: {e}")
        raise HTTPException(status_code=500, detail="Performance analysis failed")

@router.get("/risk-history/{portfolio_id}", response_model=List[Dict[str, Any]])
async def get_risk_history(
    portfolio_id: str,
    limit: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Get historical risk metrics for portfolio"""
    
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
        Portfolio.is_active == True
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
        
    # Get risk metric history
    risk_metrics = db.query(RiskMetric).filter(
        RiskMetric.portfolio_id == portfolio_id
    ).order_by(RiskMetric.calculation_date.desc()).limit(limit).all()
    
    return [
        {
            "calculation_date": rm.calculation_date,
            "var_95": rm.var_95,
            "var_99": rm.var_99,
            "expected_shortfall_95": rm.expected_shortfall_95,
            "expected_shortfall_99": rm.expected_shortfall_99,
            "volatility": rm.volatility,
            "sharpe_ratio": rm.sharpe_ratio,
            "max_drawdown": rm.max_drawdown,
            "beta": rm.beta,
            "alpha": rm.alpha
        }
        for rm in risk_metrics
    ]

@router.get("/market-summary", response_model=Dict[str, Any])
async def get_market_summary():
    """Get current market summary and indicators"""
    
    # Mock market data - in practice would fetch from real data sources
    return {
        "indices": {
            "S&P 500": {
                "value": 4450.12,
                "change": 15.67,
                "change_percent": 0.35
            },
            "NASDAQ": {
                "value": 13789.45,
                "change": -23.12,
                "change_percent": -0.17
            },
            "Dow Jones": {
                "value": 34567.89,
                "change": 45.23,
                "change_percent": 0.13
            }
        },
        "market_indicators": {
            "vix": 18.45,
            "10y_treasury_yield": 4.25,
            "dollar_index": 103.45,
            "gold_price": 1987.50,
            "oil_price": 82.15
        },
        "sector_performance": {
            "technology": 0.85,
            "healthcare": 0.45,
            "financials": 0.32,
            "energy": -0.23,
            "utilities": -0.15
        },
        "market_sentiment": "Neutral",
        "last_updated": datetime.utcnow().isoformat()
    }
