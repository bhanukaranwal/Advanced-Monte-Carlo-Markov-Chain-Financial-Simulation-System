"""
Real-time options pricing API with WebSocket support
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import redis.asyncio as redis
from contextlib import asynccontextmanager

from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from monte_carlo_engine.path_dependent import PathDependentEngine
from analytics_engine.risk_analytics import RiskAnalytics

logger = logging.getLogger(__name__)

# Pydantic models
class OptionParameters(BaseModel):
    underlying_price: float
    strike_price: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    option_type: str = "call"
    
    @validator('option_type')
    def validate_option_type(cls, v):
        if v.lower() not in ['call', 'put']:
            raise ValueError('option_type must be "call" or "put"')
        return v.lower()
    
    @validator('time_to_expiry')
    def validate_time_to_expiry(cls, v):
        if v <= 0:
            raise ValueError('time_to_expiry must be positive')
        return v

class BarrierOptionParameters(OptionParameters):
    barrier_level: float
    barrier_type: str = "up-and-out"
    rebate: float = 0.0
    
    @validator('barrier_type')
    def validate_barrier_type(cls, v):
        valid_types = ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in']
        if v.lower() not in valid_types:
            raise ValueError(f'barrier_type must be one of {valid_types}')
        return v.lower()

class AsianOptionParameters(OptionParameters):
    averaging_type: str = "arithmetic"
    
    @validator('averaging_type')
    def validate_averaging_type(cls, v):
        if v.lower() not in ['arithmetic', 'geometric']:
            raise ValueError('averaging_type must be "arithmetic" or "geometric"')
        return v.lower()

class SimulationParameters(BaseModel):
    n_simulations: int = 50000
    n_steps: int = 252
    use_gpu: bool = False
    antithetic_variates: bool = True
    random_seed: Optional[int] = None
    
    @validator('n_simulations')
    def validate_n_simulations(cls, v):
        if v < 1000 or v > 10000000:
            raise ValueError('n_simulations must be between 1,000 and 10,000,000')
        return v

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        # Remove from subscriptions
        for symbol, connections in self.subscriptions.items():
            if websocket in connections:
                connections.remove(websocket)
                
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Connection might be closed
                pass
                
    async def send_to_subscribers(self, symbol: str, message: str):
        if symbol in self.subscriptions:
            for connection in self.subscriptions[symbol][:]:  # Copy to avoid modification during iteration
                try:
                    await connection.send_text(message)
                except:
                    self.subscriptions[symbol].remove(connection)
                    
    def subscribe_to_symbol(self, websocket: WebSocket, symbol: str):
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        self.subscriptions[symbol].append(websocket)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Real-time Options Pricing API")
    app.redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)
    app.connection_manager = ConnectionManager()
    
    yield
    
    # Shutdown
    logger.info("Shutting down API")
    await app.redis_client.close()

# FastAPI app
app = FastAPI(
    title="Real-time Options Pricing API",
    description="High-performance real-time options pricing with Monte Carlo simulations",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v2/options/vanilla/price")
async def price_vanilla_option(
    option_params: OptionParameters,
    sim_params: SimulationParameters = SimulationParameters()
):
    """Price vanilla European option using Monte Carlo"""
    
    try:
        # Create engine
        engine = GeometricBrownianMotionEngine(
            n_simulations=sim_params.n_simulations,
            n_steps=sim_params.n_steps,
            initial_price=option_params.underlying_price,
            drift=option_params.risk_free_rate,
            volatility=option_params.volatility,
            random_seed=sim_params.random_seed,
            use_gpu=sim_params.use_gpu,
            antithetic_variates=sim_params.antithetic_variates
        )
        
        # Simulate paths
        result = engine.simulate_paths()
        
        # Calculate option payoff
        final_prices = result.final_prices
        
        if option_params.option_type == "call":
            payoffs = np.maximum(final_prices - option_params.strike_price, 0)
        else:
            payoffs = np.maximum(option_params.strike_price - final_prices, 0)
            
        # Discount to present value
        discount_factor = np.exp(-option_params.risk_free_rate * option_params.time_to_expiry)
        option_price = discount_factor * np.mean(payoffs)
        
        # Calculate Greeks (finite difference approximation)
        greeks = await calculate_greeks(option_params, sim_params)
        
        # Cache result
        cache_key = f"option:{hash(str(option_params.dict()))}"
        await app.redis_client.setex(
            cache_key, 
            60,  # 60 seconds TTL
            json.dumps({
                'price': float(option_price),
                'timestamp': datetime.now().isoformat()
            })
        )
        
        return {
            'option_price': float(option_price),
            'standard_error': float(discount_factor * np.std(payoffs) / np.sqrt(sim_params.n_simulations)),
            'greeks': greeks,
            'simulation_stats': {
                'execution_time': result.execution_time,
                'n_simulations': sim_params.n_simulations,
                'convergence_ratio': float(np.std(payoffs[-1000:]) / np.std(payoffs))
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error pricing vanilla option: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/options/barrier/price")
async def price_barrier_option(
    option_params: BarrierOptionParameters,
    sim_params: SimulationParameters = SimulationParameters()
):
    """Price barrier option using Monte Carlo"""
    
    try:
        engine = PathDependentEngine(
            n_simulations=sim_params.n_simulations,
            n_steps=sim_params.n_steps,
            initial_price=option_params.underlying_price,
            drift=option_params.risk_free_rate,
            volatility=option_params.volatility,
            random_seed=sim_params.random_seed
        )
        
        result = engine.price_barrier_option(
            strike=option_params.strike_price,
            barrier=option_params.barrier_level,
            option_type=option_params.option_type,
            barrier_type=option_params.barrier_type,
            risk_free_rate=option_params.risk_free_rate,
            time_to_maturity=option_params.time_to_expiry,
            rebate=option_params.rebate
        )
        
        return {
            'option_price': result['option_price'],
            'standard_error': result['standard_error'],
            'barrier_hit_probability': result['barrier_hit_rate'],
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error pricing barrier option: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/options/asian/price")
async def price_asian_option(
    option_params: AsianOptionParameters,
    sim_params: SimulationParameters = SimulationParameters()
):
    """Price Asian option using Monte Carlo"""
    
    try:
        engine = PathDependentEngine(
            n_simulations=sim_params.n_simulations,
            n_steps=sim_params.n_steps,
            initial_price=option_params.underlying_price,
            drift=option_params.risk_free_rate,
            volatility=option_params.volatility,
            random_seed=sim_params.random_seed
        )
        
        result = engine.price_asian_option(
            strike=option_params.strike_price,
            option_type=option_params.option_type,
            averaging_type=option_params.averaging_type,
            risk_free_rate=option_params.risk_free_rate,
            time_to_maturity=option_params.time_to_expiry
        )
        
        return {
            'option_price': result['option_price'],
            'standard_error': result['standard_error'],
            'averaging_type': option_params.averaging_type,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error pricing Asian option: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/api/v2/options/realtime")
async def realtime_pricing_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time option pricing"""
    
    await app.connection_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message['type'] == 'subscribe':
                # Subscribe to real-time pricing for specific options
                symbol = message.get('symbol')
                if symbol:
                    app.connection_manager.subscribe_to_symbol(websocket, symbol)
                    await app.connection_manager.send_personal_message(
                        json.dumps({
                            'type': 'subscription_confirmed',
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat()
                        }),
                        websocket
                    )
                    
            elif message['type'] == 'price_request':
                # Handle real-time pricing request
                try:
                    option_params = OptionParameters(**message['option_params'])
                    sim_params = SimulationParameters(**message.get('sim_params', {}))
                    
                    # Quick pricing (reduced simulations for real-time)
                    sim_params.n_simulations = min(10000, sim_params.n_simulations)
                    
                    # Price option (reuse the vanilla option pricing logic)
                    engine = GeometricBrownianMotionEngine(
                        n_simulations=sim_params.n_simulations,
                        n_steps=sim_params.n_steps,
                        initial_price=option_params.underlying_price,
                        drift=option_params.risk_free_rate,
                        volatility=option_params.volatility,
                        use_gpu=sim_params.use_gpu,
                        antithetic_variates=sim_params.antithetic_variates
                    )
                    
                    result = engine.simulate_paths()
                    final_prices = result.final_prices
                    
                    if option_params.option_type == "call":
                        payoffs = np.maximum(final_prices - option_params.strike_price, 0)
                    else:
                        payoffs = np.maximum(option_params.strike_price - final_prices, 0)
                        
                    discount_factor = np.exp(-option_params.risk_free_rate * option_params.time_to_expiry)
                    option_price = discount_factor * np.mean(payoffs)
                    
                    # Send response
                    response = {
                        'type': 'price_update',
                        'option_price': float(option_price),
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': result.execution_time
                    }
                    
                    await app.connection_manager.send_personal_message(
                        json.dumps(response), websocket
                    )
                    
                except Exception as e:
                    await app.connection_manager.send_personal_message(
                        json.dumps({
                            'type': 'error',
                            'message': str(e),
                            'timestamp': datetime.now().isoformat()
                        }),
                        websocket
                    )
                    
    except WebSocketDisconnect:
        app.connection_manager.disconnect(websocket)

async def calculate_greeks(option_params: OptionParameters, sim_params: SimulationParameters) -> Dict[str, float]:
    """Calculate option Greeks using finite differences"""
    
    # Delta (sensitivity to underlying price)
    bump = 0.01 * option_params.underlying_price
    
    # Price with bumped underlying
    option_params_up = option_params.copy()
    option_params_up.underlying_price += bump
    
    # Simplified pricing for Greeks calculation
    engine_up = GeometricBrownianMotionEngine(
        n_simulations=min(10000, sim_params.n_simulations),
        n_steps=sim_params.n_steps,
        initial_price=option_params_up.underlying_price,
        drift=option_params.risk_free_rate,
        volatility=option_params.volatility
    )
    
    result_up = engine_up.simulate_paths()
    if option_params.option_type == "call":
        payoffs_up = np.maximum(result_up.final_prices - option_params.strike_price, 0)
    else:
        payoffs_up = np.maximum(option_params.strike_price - result_up.final_prices, 0)
        
    discount_factor = np.exp(-option_params.risk_free_rate * option_params.time_to_expiry)
    price_up = discount_factor * np.mean(payoffs_up)
    
    # Price with original parameters (already calculated)
    engine = GeometricBrownianMotionEngine(
        n_simulations=min(10000, sim_params.n_simulations),
        n_steps=sim_params.n_steps,
        initial_price=option_params.underlying_price,
        drift=option_params.risk_free_rate,
        volatility=option_params.volatility
    )
    
    result = engine.simulate_paths()
    if option_params.option_type == "call":
        payoffs = np.maximum(result.final_prices - option_params.strike_price, 0)
    else:
        payoffs = np.maximum(option_params.strike_price - result.final_prices, 0)
        
    price = discount_factor * np.mean(payoffs)
    
    # Calculate Delta
    delta = (price_up - price) / bump
    
    # Simplified Greeks (in practice, would calculate all Greeks)
    return {
        'delta': float(delta),
        'gamma': 0.0,  # Would need second-order finite difference
        'theta': 0.0,  # Would need time bump
        'vega': 0.0,   # Would need volatility bump
        'rho': 0.0     # Would need rate bump
    }

@app.get("/api/v2/options/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.1.0'
    }

# Background task for market data updates
async def market_data_updater():
    """Background task to update market data and send to subscribers"""
    while True:
        try:
            # Simulate market data update
            await asyncio.sleep(1)
            
            # Send updates to subscribers
            update_message = json.dumps({
                'type': 'market_update',
                'data': {
                    'SPY': np.random.normal(400, 2),
                    'QQQ': np.random.normal(350, 3),
                    'timestamp': datetime.now().isoformat()
                }
            })
            
            await app.connection_manager.broadcast(update_message)
            
        except Exception as e:
            logger.error(f"Error in market data updater: {e}")
            await asyncio.sleep(5)

# Start background tasks
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(market_data_updater())
