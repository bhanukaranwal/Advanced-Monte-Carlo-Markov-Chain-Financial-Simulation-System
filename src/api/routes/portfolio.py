"""
Portfolio management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

from ...database.database import get_db
from ...database.models import Portfolio, Position, User
from ...auth.auth_manager import auth_manager, security
from ...utils.exceptions import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolios", tags=["portfolios"])

# Pydantic models
class PortfolioCreate(BaseModel):
    name: str
    description: str = ""
    currency: str = "USD"

class PortfolioUpdate(BaseModel):
    name: str = None
    description: str = None
    currency: str = None

class PositionCreate(BaseModel):
    symbol: str
    asset_type: str
    quantity: float
    average_cost: float

class PositionUpdate(BaseModel):
    quantity: float = None
    average_cost: float = None

@router.post("/", response_model=Dict[str, Any])
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Create a new portfolio"""
    
    try:
        # Create portfolio
        portfolio = Portfolio(
            name=portfolio_data.name,
            description=portfolio_data.description,
            user_id=current_user.id,
            currency=portfolio_data.currency
        )
        
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
        
        logger.info(f"Portfolio created: {portfolio.id} for user {current_user.id}")
        
        return {
            "id": portfolio.id,
            "name": portfolio.name,
            "description": portfolio.description,
            "currency": portfolio.currency,
            "created_at": portfolio.created_at
        }
        
    except Exception as e:
        logger.error(f"Error creating portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to create portfolio")

@router.get("/", response_model=List[Dict[str, Any]])
async def get_portfolios(
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Get user's portfolios"""
    
    portfolios = db.query(Portfolio).filter(
        Portfolio.user_id == current_user.id,
        Portfolio.is_active == True
    ).all()
    
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "total_value": p.total_value,
            "currency": p.currency,
            "created_at": p.created_at,
            "updated_at": p.updated_at
        }
        for p in portfolios
    ]

@router.get("/{portfolio_id}", response_model=Dict[str, Any])
async def get_portfolio(
    portfolio_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Get specific portfolio with positions"""
    
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
        Portfolio.is_active == True
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
        
    # Get positions
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "description": portfolio.description,
        "total_value": portfolio.total_value,
        "currency": portfolio.currency,
        "positions": [
            {
                "id": pos.id,
                "symbol": pos.symbol,
                "asset_type": pos.asset_type,
                "quantity": pos.quantity,
                "average_cost": pos.average_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "weight": pos.weight
            }
            for pos in positions
        ],
        "created_at": portfolio.created_at,
        "updated_at": portfolio.updated_at
    }

@router.put("/{portfolio_id}", response_model=Dict[str, Any])
async def update_portfolio(
    portfolio_id: str,
    portfolio_data: PortfolioUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Update portfolio"""
    
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
        Portfolio.is_active == True
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
        
    # Update fields
    update_data = portfolio_data.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(portfolio, field, value)
        
    db.commit()
    db.refresh(portfolio)
    
    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "description": portfolio.description,
        "currency": portfolio.currency,
        "updated_at": portfolio.updated_at
    }

@router.post("/{portfolio_id}/positions", response_model=Dict[str, Any])
async def add_position(
    portfolio_id: str,
    position_data: PositionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Add position to portfolio"""
    
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
        Portfolio.is_active == True
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
        
    # Create position
    position = Position(
        portfolio_id=portfolio_id,
        symbol=position_data.symbol,
        asset_type=position_data.asset_type,
        quantity=position_data.quantity,
        average_cost=position_data.average_cost,
        market_value=position_data.quantity * position_data.average_cost
    )
    
    db.add(position)
    db.commit()
    db.refresh(position)
    
    return {
        "id": position.id,
        "symbol": position.symbol,
        "asset_type": position.asset_type,
        "quantity": position.quantity,
        "average_cost": position.average_cost,
        "market_value": position.market_value
    }

@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Delete portfolio (soft delete)"""
    
    portfolio = db.query(Portfolio).filter(
        Portfolio.id == portfolio_id,
        Portfolio.user_id == current_user.id,
        Portfolio.is_active == True
    ).first()
    
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
        
    portfolio.is_active = False
    db.commit()
    
    logger.info(f"Portfolio deleted: {portfolio_id}")
