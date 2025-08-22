"""
Monte Carlo simulation API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from pydantic import BaseModel, validator
import logging

from ...database.database import get_db
from ...database.models import Simulation, User
from ...auth.auth_manager import auth_manager, security
from ...monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from ...monte_carlo_engine.multi_asset import MultiAssetEngine
from ...cache.cache_manager import simulation_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulations", tags=["simulations"])

# Pydantic models
class GBMSimulationRequest(BaseModel):
    n_simulations: int
    n_steps: int
    initial_price: float
    drift: float
    volatility: float
    use_gpu: bool = True
    antithetic_variates: bool = False
    random_seed: int = None
    
    @validator('n_simulations')
    def validate_simulations(cls, v):
        if not 1000 <= v <= 10000000:
            raise ValueError('n_simulations must be between 1,000 and 10,000,000')
        return v

class MultiAssetSimulationRequest(BaseModel):
    n_simulations: int
    n_steps: int
    initial_prices: List[float]
    drifts: List[float]
    volatilities: List[float]
    correlation_matrix: List[List[float]]
    asset_names: List[str] = None
    random_seed: int = None

@router.post("/gbm", response_model=Dict[str, Any])
async def run_gbm_simulation(
    request: GBMSimulationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Run Geometric Brownian Motion simulation"""
    
    try:
        # Check cache first
        params = request.dict()
        cached_result = simulation_cache.get_simulation_result("gbm", params)
        
        if cached_result:
            logger.info(f"Returning cached GBM simulation for user {current_user.id}")
            return {
                "simulation_id": "cached",
                "status": "completed",
                "results": cached_result,
                "cached": True
            }
        
        # Create simulation record
        simulation = Simulation(
            user_id=current_user.id,
            simulation_type="gbm",
            parameters=params,
            status="running",
            n_simulations=request.n_simulations,
            n_steps=request.n_steps
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        # Run simulation in background
        background_tasks.add_task(
            _run_gbm_background,
            simulation.id,
            request
        )
        
        return {
            "simulation_id": simulation.id,
            "status": "running",
            "message": "Simulation started. Check status endpoint for results."
        }
        
    except Exception as e:
        logger.error(f"Error starting GBM simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to start simulation")

@router.post("/multi-asset", response_model=Dict[str, Any])
async def run_multi_asset_simulation(
    request: MultiAssetSimulationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Run multi-asset Monte Carlo simulation"""
    
    try:
        # Validation
        n_assets = len(request.initial_prices)
        if not (len(request.drifts) == len(request.volatilities) == n_assets):
            raise ValueError("All asset parameter arrays must have same length")
            
        if len(request.correlation_matrix) != n_assets:
            raise ValueError("Correlation matrix dimension mismatch")
            
        # Create simulation record
        params = request.dict()
        simulation = Simulation(
            user_id=current_user.id,
            simulation_type="multi_asset",
            parameters=params,
            status="running",
            n_simulations=request.n_simulations,
            n_steps=request.n_steps
        )
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        # Run simulation in background
        background_tasks.add_task(
            _run_multi_asset_background,
            simulation.id,
            request
        )
        
        return {
            "simulation_id": simulation.id,
            "status": "running",
            "message": "Multi-asset simulation started"
        }
        
    except Exception as e:
        logger.error(f"Error starting multi-asset simulation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{simulation_id}", response_model=Dict[str, Any])
async def get_simulation_status(
    simulation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Get simulation status and results"""
    
    simulation = db.query(Simulation).filter(
        Simulation.id == simulation_id,
        Simulation.user_id == current_user.id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
        
    response = {
        "simulation_id": simulation.id,
        "simulation_type": simulation.simulation_type,
        "status": simulation.status,
        "parameters": simulation.parameters,
        "created_at": simulation.created_at
    }
    
    if simulation.status == "completed":
        response["results"] = simulation.results
        response["execution_time"] = simulation.execution_time
        response["completed_at"] = simulation.completed_at
    elif simulation.status == "failed":
        response["error"] = "Simulation failed"
        
    return response

@router.get("/", response_model=List[Dict[str, Any]])
async def get_user_simulations(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: auth_manager.get_current_user(Depends(security)))
):
    """Get user's simulations"""
    
    simulations = db.query(Simulation).filter(
        Simulation.user_id == current_user.id
    ).order_by(Simulation.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        {
            "simulation_id": sim.id,
            "simulation_type": sim.simulation_type,
            "status": sim.status,
            "n_simulations": sim.n_simulations,
            "n_steps": sim.n_steps,
            "created_at": sim.created_at,
            "completed_at": sim.completed_at,
            "execution_time": sim.execution_time
        }
        for sim in simulations
    ]

async def _run_gbm_background(simulation_id: str, request: GBMSimulationRequest):
    """Run GBM simulation in background"""
    
    with get_db() as db:
        try:
            # Create engine
            engine = GeometricBrownianMotionEngine(
                n_simulations=request.n_simulations,
                n_steps=request.n_steps,
                initial_price=request.initial_price,
                drift=request.drift,
                volatility=request.volatility,
                random_seed=request.random_seed,
                use_gpu=request.use_gpu,
                antithetic_variates=request.antithetic_variates
            )
            
            # Run simulation
            result = engine.simulate()
            
            # Store results
            simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            simulation.status = "completed"
            simulation.results = {
                "statistics": result.statistics,
                "execution_time": result.execution_time,
                "parameters": result.parameters,
                "final_prices_sample": result.final_prices[:100].tolist()  # Store sample
            }
            simulation.execution_time = result.execution_time
            simulation.completed_at = datetime.utcnow()
            
            db.commit()
            
            # Cache results
            simulation_cache.cache_simulation_result("gbm", request.dict(), simulation.results)
            
            logger.info(f"GBM simulation {simulation_id} completed")
            
        except Exception as e:
            logger.error(f"GBM simulation {simulation_id} failed: {e}")
            
            simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            simulation.status = "failed"
            db.commit()

async def _run_multi_asset_background(simulation_id: str, request: MultiAssetSimulationRequest):
    """Run multi-asset simulation in background"""
    
    with get_db() as db:
        try:
            import numpy as np
            
            # Create engine
            engine = MultiAssetEngine(
                n_simulations=request.n_simulations,
                n_steps=request.n_steps,
                initial_prices=request.initial_prices,
                drifts=request.drifts,
                volatilities=request.volatilities,
                correlation_matrix=np.array(request.correlation_matrix),
                asset_names=request.asset_names,
                random_seed=request.random_seed
            )
            
            # Run simulation
            result = engine.simulate()
            
            # Store results
            simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            simulation.status = "completed"
            simulation.results = {
                "asset_statistics": result.asset_statistics,
                "execution_time": result.execution_time,
                "correlation_matrix": result.correlation_matrix.tolist()
            }
            simulation.execution_time = result.execution_time
            simulation.completed_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Multi-asset simulation {simulation_id} completed")
            
        except Exception as e:
            logger.error(f"Multi-asset simulation {simulation_id} failed: {e}")
            
            simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
            simulation.status = "failed"
            db.commit()
