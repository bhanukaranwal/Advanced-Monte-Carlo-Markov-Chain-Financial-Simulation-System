"""
Cryptocurrency market models with advanced volatility and correlation dynamics
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t, skew, kurtosis
from scipy.optimize import minimize
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import aiohttp
from abc import ABC, abstractmethod

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    import web3
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CryptoModelResult:
    """Results from cryptocurrency model simulation"""
    prices: np.ndarray
    returns: np.ndarray
    volatility: np.ndarray
    regime_probabilities: Optional[np.ndarray]
    correlation_matrix: Optional[np.ndarray]
    on_chain_metrics: Optional[Dict[str, Any]]
    defi_metrics: Optional[Dict[str, Any]]
    execution_time: float

class BaseCryptoModel(ABC):
    """Abstract base class for cryptocurrency models"""
    
    def __init__(self, symbol: str, initial_price: float):
        self.symbol = symbol
        self.initial_price = initial_price
        
    @abstractmethod
    def simulate(self, n_steps: int, n_simulations: int) -> CryptoModelResult:
        """Simulate cryptocurrency price paths"""
        pass
        
    @abstractmethod
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate model parameters from historical data"""
        pass

class BitcoinJumpDiffusionModel(BaseCryptoModel):
    """
    Bitcoin-specific jump-diffusion model with regime switching
    Accounts for high volatility, fat tails, and sudden price jumps
    """
    
    def __init__(self, symbol: str = "BTC", initial_price: float = 50000.0):
        super().__init__(symbol, initial_price)
        
        # Model parameters
        self.drift = 0.5  # Annual drift
        self.volatility = 0.8  # Base volatility (high for crypto)
        self.jump_intensity = 30.0  # Jumps per year
        self.jump_size_mean = 0.0  # Mean jump size
        self.jump_size_std = 0.15  # Jump size volatility
        
        # Regime parameters
        self.n_regimes = 3
        self.regime_volatilities = [0.4, 0.8, 1.5]  # Low, Medium, High vol regimes
        self.regime_drifts = [0.2, 0.5, -0.1]  # Bull, Normal, Bear market drifts
        self.transition_matrix = np.array([
            [0.95, 0.04, 0.01],  # Low vol regime
            [0.02, 0.96, 0.02],  # Medium vol regime  
            [0.05, 0.20, 0.75]   # High vol regime
        ])
        
        # Current regime
        self.current_regime = 1  # Start in medium volatility regime
        
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate model parameters from Bitcoin historical data"""
        
        if 'close' not in historical_data.columns:
            raise ValueError("Historical data must contain 'close' column")
            
        prices = historical_data['close'].values
        returns = np.diff(np.log(prices))
        
        # Basic parameter estimation
        self.drift = np.mean(returns) * 252  # Annualized
        base_vol = np.std(returns) * np.sqrt(252)
        
        # Detect jumps (returns > 3 standard deviations)
        jump_threshold = 3 * np.std(returns)
        jumps = returns[np.abs(returns) > jump_threshold]
        
        if len(jumps) > 0:
            self.jump_intensity = len(jumps) / (len(returns) / 252)  # Jumps per year
            self.jump_size_mean = np.mean(jumps)
            self.jump_size_std = np.std(jumps)
        
        # Estimate regime parameters using volatility clustering
        self._estimate_regime_parameters(returns)
        
        logger.info(f"Calibrated {self.symbol} model: drift={self.drift:.3f}, base_vol={base_vol:.3f}")
        
    def _estimate_regime_parameters(self, returns: np.ndarray):
        """Estimate regime parameters from returns"""
        
        # Calculate rolling volatility
        window = 20
        rolling_vol = pd.Series(returns).rolling(window).std().dropna() * np.sqrt(252)
        
        # Use quantiles to define regime boundaries
        vol_33 = np.percentile(rolling_vol, 33)
        vol_66 = np.percentile(rolling_vol, 66)
        
        # Update regime volatilities based on data
        self.regime_volatilities = [
            min(0.6, vol_33),
            np.clip((vol_33 + vol_66) / 2, 0.6, 1.2),
            max(1.0, vol_66)
        ]
        
        # Estimate regime-dependent drifts
        low_vol_mask = rolling_vol < vol_33
        med_vol_mask = (rolling_vol >= vol_33) & (rolling_vol < vol_66)
        high_vol_mask = rolling_vol >= vol_66
        
        if np.sum(low_vol_mask) > 0:
            self.regime_drifts[0] = np.mean(returns[low_vol_mask.values]) * 252
        if np.sum(med_vol_mask) > 0:
            self.regime_drifts[1] = np.mean(returns[med_vol_mask.values]) * 252
        if np.sum(high_vol_mask) > 0:
            self.regime_drifts[2] = np.mean(returns[high_vol_mask.values]) * 252
            
    def simulate(self, n_steps: int, n_simulations: int = 10000) -> CryptoModelResult:
        """Simulate Bitcoin price paths with jumps and regime switching"""
        
        import time
        start_time = time.time()
        
        dt = 1.0 / 252  # Daily steps
        
        # Initialize arrays
        prices = np.zeros((n_simulations, n_steps + 1))
        returns = np.zeros((n_simulations, n_steps))
        volatilities = np.zeros((n_simulations, n_steps))
        regime_probs = np.zeros((n_simulations, n_steps, self.n_regimes))
        
        prices[:, 0] = self.initial_price
        
        for i in range(n_simulations):
            current_regime = self.current_regime
            
            for t in range(n_steps):
                # Regime switching
                transition_probs = self.transition_matrix[current_regime]
                current_regime = np.random.choice(self.n_regimes, p=transition_probs)
                
                # Current regime parameters
                current_drift = self.regime_drifts[current_regime]
                current_vol = self.regime_volatilities[current_regime]
                
                volatilities[i, t] = current_vol
                regime_probs[i, t, current_regime] = 1.0
                
                # Diffusion component
                drift_term = (current_drift - 0.5 * current_vol**2) * dt
                diffusion_term = current_vol * np.sqrt(dt) * np.random.normal()
                
                # Jump component
                jump_term = 0.0
                if np.random.poisson(self.jump_intensity * dt) > 0:
                    jump_size = np.random.normal(self.jump_size_mean, self.jump_size_std)
                    jump_term = jump_size
                    
                # Total return
                log_return = drift_term + diffusion_term + jump_term
                returns[i, t] = log_return
                
                # Update price
                prices[i, t + 1] = prices[i, t] * np.exp(log_return)
                
        execution_time = time.time() - start_time
        
        return CryptoModelResult(
            prices=prices,
            returns=returns,
            volatility=volatilities,
            regime_probabilities=regime_probs,
            correlation_matrix=None,
            on_chain_metrics=None,
            defi_metrics=None,
            execution_time=execution_time
        )

class AltcoinCorrelationModel(BaseCryptoModel):
    """
    Multi-asset cryptocurrency model with dynamic correlations
    Models the relationship between major altcoins and Bitcoin
    """
    
    def __init__(self, symbols: List[str], initial_prices: List[float]):
        self.symbols = symbols
        self.initial_prices = np.array(initial_prices)
        self.n_assets = len(symbols)
        
        # Model parameters
        self.drifts = np.array([0.3, 0.4, 0.2, 0.1])  # ETH, BNB, ADA, SOL
        self.volatilities = np.array([0.7, 0.9, 1.1, 1.3])
        
        # Dynamic correlation parameters
        self.base_correlation = np.array([
            [1.0, 0.7, 0.6, 0.5],  # BTC correlations
            [0.7, 1.0, 0.8, 0.7],  # ETH correlations
            [0.6, 0.8, 1.0, 0.6],  # BNB correlations
            [0.5, 0.7, 0.6, 1.0]   # Others
        ])
        
        # DCC-GARCH parameters for dynamic correlations
        self.alpha = 0.05  # Short-term correlation persistence
        self.beta = 0.90   # Long-term correlation persistence
        
    def calibrate(self, historical_data: Dict[str, pd.DataFrame]):
        """Calibrate model from multiple cryptocurrency time series"""
        
        returns_data = {}
        
        for symbol in self.symbols:
            if symbol not in historical_data:
                raise ValueError(f"Missing historical data for {symbol}")
                
            prices = historical_data[symbol]['close'].values
            returns = np.diff(np.log(prices))
            returns_data[symbol] = returns
            
        # Align all return series
        min_length = min(len(returns) for returns in returns_data.values())
        aligned_returns = np.array([returns_data[symbol][-min_length:] for symbol in self.symbols]).T
        
        # Estimate basic parameters
        self.drifts = np.mean(aligned_returns, axis=0) * 252
        self.volatilities = np.std(aligned_returns, axis=0) * np.sqrt(252)
        
        # Estimate base correlation matrix
        self.base_correlation = np.corrcoef(aligned_returns.T)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.base_correlation)
        eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
        self.base_correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        logger.info(f"Calibrated altcoin model for {len(self.symbols)} assets")
        
    def simulate(self, n_steps: int, n_simulations: int = 5000) -> CryptoModelResult:
        """Simulate correlated altcoin price paths with dynamic correlations"""
        
        import time
        start_time = time.time()
        
        dt = 1.0 / 252
        
        # Initialize arrays
        prices = np.zeros((n_simulations, n_steps + 1, self.n_assets))
        returns = np.zeros((n_simulations, n_steps, self.n_assets))
        correlations = np.zeros((n_simulations, n_steps, self.n_assets, self.n_assets))
        
        prices[:, 0, :] = self.initial_prices
        
        # Cholesky decomposition of base correlation
        L = np.linalg.cholesky(self.base_correlation)
        
        for i in range(n_simulations):
            current_correlation = self.base_correlation.copy()
            
            for t in range(n_steps):
                # Update dynamic correlation (simplified DCC-GARCH)
                if t > 0:
                    prev_returns = returns[i, t-1, :]
                    innovation = np.outer(prev_returns, prev_returns)
                    
                    # DCC update
                    current_correlation = (
                        (1 - self.alpha - self.beta) * self.base_correlation +
                        self.alpha * innovation +
                        self.beta * current_correlation
                    )
                    
                    # Ensure correlation matrix properties
                    current_correlation = self._ensure_correlation_matrix(current_correlation)
                    L = np.linalg.cholesky(current_correlation)
                    
                correlations[i, t, :, :] = current_correlation
                
                # Generate correlated random numbers
                independent_normals = np.random.normal(0, 1, self.n_assets)
                correlated_normals = L @ independent_normals
                
                # Calculate returns for each asset
                for j in range(self.n_assets):
                    drift_term = (self.drifts[j] - 0.5 * self.volatilities[j]**2) * dt
                    diffusion_term = self.volatilities[j] * np.sqrt(dt) * correlated_normals[j]
                    
                    log_return = drift_term + diffusion_term
                    returns[i, t, j] = log_return
                    
                    # Update price
                    prices[i, t + 1, j] = prices[i, t, j] * np.exp(log_return)
                    
        execution_time = time.time() - start_time
        
        # Average correlation matrix over time and simulations
        avg_correlation = np.mean(correlations, axis=(0, 1))
        
        return CryptoModelResult(
            prices=prices,
            returns=returns,
            volatility=None,
            regime_probabilities=None,
            correlation_matrix=avg_correlation,
            on_chain_metrics=None,
            defi_metrics=None,
            execution_time=execution_time
        )
        
    def _ensure_correlation_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure matrix is a valid correlation matrix"""
        
        # Set diagonal to 1
        np.fill_diagonal(matrix, 1.0)
        
        # Ensure symmetry
        matrix = (matrix + matrix.T) / 2
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Re-normalize diagonal
        np.fill_diagonal(matrix, 1.0)
        
        return matrix

class StablecoinModel(BaseCryptoModel):
    """
    Stablecoin model with peg deviations and liquidity dynamics
    Models USDT, USDC, DAI behavior during stress periods
    """
    
    def __init__(self, symbol: str = "USDT", peg_value: float = 1.0):
        super().__init__(symbol, peg_value)
        self.peg_value = peg_value
        
        # Model parameters
        self.mean_reversion_rate = 50.0  # Strong mean reversion to peg
        self.base_volatility = 0.01  # Very low base volatility
        self.stress_volatility = 0.05  # Higher volatility during stress
        self.stress_probability = 0.02  # Probability of stress event per day
        self.stress_duration = 5  # Average stress duration in days
        
        # Current state
        self.is_stressed = False
        self.stress_remaining = 0
        
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate stablecoin model from price data"""
        
        prices = historical_data['close'].values
        deviations = prices - self.peg_value
        
        # Estimate mean reversion rate
        returns = np.diff(np.log(prices))
        price_levels = prices[:-1]
        
        # Simple OLS for mean reversion: dr = -kappa * (log(P) - log(peg)) * dt + sigma * dW
        log_deviations = np.log(price_levels / self.peg_value)
        
        if len(log_deviations) > 10:
            slope = np.cov(returns, log_deviations[1:])[0, 1] / np.var(log_deviations[1:])
            self.mean_reversion_rate = -slope * 252  # Annualized
            
        # Estimate volatility regimes
        vol_rolling = pd.Series(returns).rolling(10).std() * np.sqrt(252)
        self.base_volatility = np.percentile(vol_rolling.dropna(), 25)
        self.stress_volatility = np.percentile(vol_rolling.dropna(), 95)
        
        logger.info(f"Calibrated {self.symbol} stablecoin model")
        
    def simulate(self, n_steps: int, n_simulations: int = 10000) -> CryptoModelResult:
        """Simulate stablecoin prices with mean reversion and stress events"""
        
        import time
        start_time = time.time()
        
        dt = 1.0 / 252
        
        # Initialize arrays
        prices = np.zeros((n_simulations, n_steps + 1))
        returns = np.zeros((n_simulations, n_steps))
        volatilities = np.zeros((n_simulations, n_steps))
        stress_indicators = np.zeros((n_simulations, n_steps))
        
        prices[:, 0] = self.initial_price
        
        for i in range(n_simulations):
            is_stressed = False
            stress_remaining = 0
            
            for t in range(n_steps):
                current_price = prices[i, t]
                
                # Update stress state
                if stress_remaining > 0:
                    stress_remaining -= 1
                    if stress_remaining == 0:
                        is_stressed = False
                elif np.random.random() < self.stress_probability:
                    is_stressed = True
                    stress_remaining = int(np.random.exponential(self.stress_duration))
                    
                stress_indicators[i, t] = float(is_stressed)
                
                # Choose volatility based on stress state
                current_vol = self.stress_volatility if is_stressed else self.base_volatility
                volatilities[i, t] = current_vol
                
                # Mean reversion component
                deviation = np.log(current_price / self.peg_value)
                mean_reversion = -self.mean_reversion_rate * deviation * dt
                
                # Random component
                random_component = current_vol * np.sqrt(dt) * np.random.normal()
                
                # Total log return
                log_return = mean_reversion + random_component
                returns[i, t] = log_return
                
                # Update price
                prices[i, t + 1] = current_price * np.exp(log_return)
                
        execution_time = time.time() - start_time
        
        return CryptoModelResult(
            prices=prices,
            returns=returns,
            volatility=volatilities,
            regime_probabilities=None,
            correlation_matrix=None,
            on_chain_metrics={'stress_periods': stress_indicators},
            defi_metrics=None,
            execution_time=execution_time
        )

class DeFiIntegrationModel:
    """
    DeFi (Decentralized Finance) integration model
    Incorporates yield farming, liquidity pool dynamics, and governance token behavior
    """
    
    def __init__(self):
        self.web3_client = None
        self.initialize_web3()
        
    def initialize_web3(self):
        """Initialize Web3 connection for on-chain data"""
        if WEB3_AVAILABLE:
            try:
                # Connect to Ethereum mainnet (replace with your provider)
                self.web3_client = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_KEY'))
                logger.info("Web3 client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Web3: {e}")
                
    async def fetch_defi_metrics(self, protocols: List[str]) -> Dict[str, Any]:
        """Fetch DeFi protocol metrics"""
        
        metrics = {}
        
        # Mock implementation - in practice would fetch from DeFiPulse, DeBank, etc.
        for protocol in protocols:
            metrics[protocol] = {
                'tvl': np.random.uniform(1e9, 10e9),  # Total Value Locked
                'apy': np.random.uniform(0.05, 0.25),  # Annual Percentage Yield
                'volume_24h': np.random.uniform(1e8, 1e9),
                'fees_24h': np.random.uniform(1e6, 10e6),
                'governance_token_price': np.random.uniform(10, 1000)
            }
            
        return metrics
        
    def model_yield_farming_returns(self, base_apy: float, volatility: float, 
                                  n_days: int) -> np.ndarray:
        """Model yield farming returns with impermanent loss"""
        
        dt = 1.0 / 365
        
        # Base yield component (relatively stable)
        base_yield = np.full(n_days, base_apy * dt)
        
        # Variable yield component (depends on trading volume and fees)
        variable_yield = np.random.normal(0, volatility * np.sqrt(dt), n_days)
        
        # Impermanent loss component (negative correlation with price stability)
        price_volatility = np.random.normal(0, 0.02, n_days)  # Daily price changes
        impermanent_loss = -0.5 * np.square(price_volatility)  # Quadratic IL
        
        total_yields = base_yield + variable_yield + impermanent_loss
        
        return total_yields
        
    def model_governance_token(self, base_price: float, protocol_metrics: Dict[str, float],
                             n_steps: int) -> np.ndarray:
        """Model governance token price based on protocol fundamentals"""
        
        dt = 1.0 / 252
        
        # Extract fundamental drivers
        tvl_growth = protocol_metrics.get('tvl_growth', 0.1)
        fee_yield = protocol_metrics.get('fee_yield', 0.05)
        
        # Model parameters
        base_drift = 0.2 + 0.5 * tvl_growth + 2.0 * fee_yield  # Fundamentals-driven drift
        volatility = 1.2  # High volatility for governance tokens
        
        # Generate price path
        returns = np.random.normal(
            (base_drift - 0.5 * volatility**2) * dt,
            volatility * np.sqrt(dt),
            n_steps
        )
        
        prices = np.zeros(n_steps + 1)
        prices[0] = base_price
        
        for i in range(n_steps):
            prices[i + 1] = prices[i] * np.exp(returns[i])
            
        return prices

class OnChainAnalyticsIntegrator:
    """
    On-chain analytics integration for cryptocurrency models
    Incorporates network metrics, transaction data, and social sentiment
    """
    
    def __init__(self):
        self.api_endpoints = {
            'glassnode': 'https://api.glassnode.com/v1/metrics/',
            'coinmetrics': 'https://api.coinmetrics.io/v4/',
            'santiment': 'https://api.santiment.net/graphql'
        }
        
    async def fetch_on_chain_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive on-chain metrics"""
        
        # Mock implementation - real version would fetch from APIs
        metrics = {
            'network_metrics': {
                'active_addresses': np.random.randint(500000, 1000000),
                'transaction_count': np.random.randint(200000, 400000),
                'hash_rate': np.random.uniform(100, 200),  # EH/s for Bitcoin
                'difficulty': np.random.uniform(20e12, 30e12),
                'mempool_size': np.random.randint(5000, 50000)
            },
            'holder_metrics': {
                'hodler_ratio': np.random.uniform(0.6, 0.8),
                'exchange_inflows': np.random.uniform(1000, 10000),
                'exchange_outflows': np.random.uniform(1000, 10000),
                'whale_activity': np.random.uniform(0, 1),
                'long_term_holder_supply': np.random.uniform(0.5, 0.7)
            },
            'market_metrics': {
                'realized_price': np.random.uniform(30000, 60000),
                'mvrv_ratio': np.random.uniform(0.8, 2.5),
                'nvt_ratio': np.random.uniform(50, 150),
                'sopr': np.random.uniform(0.95, 1.05),
                'fear_greed_index': np.random.randint(10, 90)
            }
        }
        
        return metrics
        
    def integrate_on_chain_signals(self, model_result: CryptoModelResult, 
                                 on_chain_data: Dict[str, Any]) -> CryptoModelResult:
        """Integrate on-chain signals into model results"""
        
        # Adjust model results based on on-chain metrics
        adjusted_result = model_result
        
        # Example: Adjust volatility based on network activity
        network_activity = on_chain_data['network_metrics']['active_addresses']
        activity_factor = network_activity / 750000  # Normalize around typical level
        
        if hasattr(adjusted_result, 'volatility') and adjusted_result.volatility is not None:
            adjusted_result.volatility *= activity_factor
            
        # Store on-chain metrics
        adjusted_result.on_chain_metrics = on_chain_data
        
        return adjusted_result

class CryptocurrencyPortfolioOptimizer:
    """
    Cryptocurrency portfolio optimization with unique crypto considerations
    """
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Lower than traditional markets
        
    def optimize_crypto_portfolio(self, expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                market_caps: np.ndarray,
                                liquidity_scores: np.ndarray,
                                regulatory_scores: np.ndarray) -> Dict[str, Any]:
        """
        Optimize cryptocurrency portfolio considering unique factors
        
        Args:
            expected_returns: Expected annual returns for each crypto
            covariance_matrix: Return covariance matrix
            market_caps: Market capitalizations (for liquidity weighting)
            liquidity_scores: Liquidity scores (0-1, higher is better)
            regulatory_scores: Regulatory clarity scores (0-1, higher is better)
        """
        
        n_assets = len(expected_returns)
        
        def objective(weights):
            # Portfolio return
            portfolio_return = np.dot(weights, expected_returns)
            
            # Portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            # Liquidity penalty (prefer more liquid assets)
            liquidity_penalty = -0.1 * np.dot(weights, liquidity_scores)
            
            # Regulatory penalty (prefer assets with better regulatory clarity)
            regulatory_penalty = -0.05 * np.dot(weights, regulatory_scores)
            
            # Concentration penalty (avoid over-concentration)
            concentration_penalty = 0.1 * np.sum(np.square(weights))
            
            # Total objective (minimize negative Sharpe ratio with penalties)
            risk_adjusted_return = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
            
            return -(risk_adjusted_return + liquidity_penalty + regulatory_penalty - concentration_penalty)
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w}  # Long-only
        ]
        
        # Bounds (max 30% in any single crypto for diversification)
        bounds = [(0.0, 0.3) for _ in range(n_assets)]
        
        # Initial guess (market cap weighted)
        initial_weights = market_caps / np.sum(market_caps)
        initial_weights = np.clip(initial_weights, 0.01, 0.3)
        initial_weights = initial_weights / np.sum(initial_weights)
        
        # Optimize
        from scipy.optimize import minimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            return {
                'optimal_weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_volatility,
                'optimization_success': True,
                'diversification_ratio': self._calculate_diversification_ratio(
                    optimal_weights, np.sqrt(np.diag(covariance_matrix)), portfolio_volatility
                )
            }
        else:
            return {'optimization_success': False, 'message': result.message}
            
    def _calculate_diversification_ratio(self, weights: np.ndarray, 
                                       individual_vols: np.ndarray,
                                       portfolio_vol: float) -> float:
        """Calculate diversification ratio"""
        weighted_vol = np.dot(weights, individual_vols)
        return weighted_vol / portfolio_vol

class CryptoMarketEngine:
    """
    Main engine combining all cryptocurrency models and analytics
    """
    
    def __init__(self):
        self.models = {}
        self.on_chain_integrator = OnChainAnalyticsIntegrator()
        self.defi_integrator = DeFiIntegrationModel()
        self.portfolio_optimizer = CryptocurrencyPortfolioOptimizer()
        
    def register_model(self, name: str, model: BaseCryptoModel):
        """Register a cryptocurrency model"""
        self.models[name] = model
        logger.info(f"Registered crypto model: {name}")
        
    async def simulate_crypto_market(self, config: Dict[str, Any]) -> Dict[str, CryptoModelResult]:
        """Simulate entire cryptocurrency market"""
        
        results = {}
        
        # Run individual asset simulations
        for name, model in self.models.items():
            logger.info(f"Simulating {name}")
            result = model.simulate(
                n_steps=config.get('n_steps', 252),
                n_simulations=config.get('n_simulations', 1000)
            )
            
            # Integrate on-chain data if available
            if config.get('include_on_chain', True):
                symbol = getattr(model, 'symbol', name.upper())
                on_chain_data = await self.on_chain_integrator.fetch_on_chain_metrics(symbol)
                result = self.on_chain_integrator.integrate_on_chain_signals(result, on_chain_data)
                
            results[name] = result
            
        # Integrate DeFi metrics if requested
        if config.get('include_defi', True):
            defi_protocols = config.get('defi_protocols', ['uniswap', 'compound', 'aave'])
            defi_metrics = await self.defi_integrator.fetch_defi_metrics(defi_protocols)
            
            # Store DeFi metrics in results
            for name in results:
                results[name].defi_metrics = defi_metrics
                
        return results
        
    def optimize_crypto_portfolio(self, simulation_results: Dict[str, CryptoModelResult],
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cryptocurrency portfolio based on simulation results"""
        
        # Extract expected returns and covariances from simulation results
        expected_returns = []
        return_series = []
        
        for name, result in simulation_results.items():
            # Calculate expected return from simulation
            final_prices = result.prices[:, -1]
            initial_prices = result.prices[:, 0]
            total_returns = (final_prices / initial_prices) - 1
            expected_returns.append(np.mean(total_returns))
            
            # Average returns across simulations for covariance calculation
            avg_returns = np.mean(result.returns, axis=0)  # Average across simulations
            return_series.append(avg_returns)
            
        expected_returns = np.array(expected_returns)
        covariance_matrix = np.cov(return_series)
        
        # Market data
        symbols = list(simulation_results.keys())
        market_caps = np.array([market_data.get(f'{s}_market_cap', 1e9) for s in symbols])
        liquidity_scores = np.array([market_data.get(f'{s}_liquidity', 0.5) for s in symbols])
        regulatory_scores = np.array([market_data.get(f'{s}_regulatory', 0.5) for s in symbols])
        
        # Optimize
        optimization_result = self.portfolio_optimizer.optimize_crypto_portfolio(
            expected_returns, covariance_matrix, market_caps, 
            liquidity_scores, regulatory_scores
        )
        
        # Add asset mapping
        if optimization_result['optimization_success']:
            optimization_result['asset_weights'] = dict(zip(symbols, optimization_result['optimal_weights']))
            
        return optimization_result

# Example usage and factory functions
def create_bitcoin_model(initial_price: float = 50000.0) -> BitcoinJumpDiffusionModel:
    """Create calibrated Bitcoin model"""
    return BitcoinJumpDiffusionModel("BTC", initial_price)

def create_altcoin_model(symbols: List[str], initial_prices: List[float]) -> AltcoinCorrelationModel:
    """Create multi-asset altcoin model"""
    return AltcoinCorrelationModel(symbols, initial_prices)

def create_stablecoin_model(symbol: str = "USDT") -> StablecoinModel:
    """Create stablecoin model"""
    return StablecoinModel(symbol, 1.0)
