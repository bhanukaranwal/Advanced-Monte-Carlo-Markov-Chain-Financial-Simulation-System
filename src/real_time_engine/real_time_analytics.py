"""
Real-time analytics for risk management and portfolio monitoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import asyncio
import logging

from .stream_processor import MarketTick, AggregatedData
from .kalman_filters import TrendFollowingKalman, VolatilityKalman

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    portfolio_value: float
    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    beta: float
    correlation_to_market: float
    concentration_risk: float
    
@dataclass
class PositionMetrics:
    """Metrics for individual positions"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    var_contribution: float
    volatility: float
    beta: float
    last_update: datetime

class RealTimeRiskAnalytics:
    """Real-time risk analytics engine"""
    
    def __init__(
        self,
        portfolio_data: Dict[str, float],  # symbol -> position size
        risk_free_rate: float = 0.02,
        confidence_levels: List[float] = [0.95, 0.99],
        lookback_window: int = 252,
        update_frequency: int = 60  # seconds
    ):
        self.portfolio = portfolio_data.copy()
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        
        # Data storage
        self.price_data = defaultdict(lambda: deque(maxlen=lookback_window))
        self.return_data = defaultdict(lambda: deque(maxlen=lookback_window))
        self.portfolio_values = deque(maxlen=lookback_window)
        self.portfolio_returns = deque(maxlen=lookback_window)
        
        # Risk metrics history
        self.risk_metrics_history = deque(maxlen=1000)
        self.position_metrics = {}
        
        # Market data (benchmark)
        self.market_data = deque(maxlen=lookback_window)
        self.market_returns = deque(maxlen=lookback_window)
        
        # Kalman filters for each asset
        self.trend_filters = {}
        self.volatility_filters = {}
        
        # Initialize filters
        for symbol in portfolio_data.keys():
            self.trend_filters[symbol] = TrendFollowingKalman()
            self.volatility_filters[symbol] = VolatilityKalman()
            
        # Covariance estimation
        self.return_covariance = None
        self.covariance_decay = 0.94  # RiskMetrics decay factor
        
        # Real-time state
        self.last_portfolio_value = 0.0
        self.peak_portfolio_value = 0.0
        self.running = False
        
    async def start_analytics(self):
        """Start real-time analytics"""
        self.running = True
        logger.info("Starting real-time risk analytics")
        
        # Start periodic risk calculation
        asyncio.create_task(self._periodic_risk_calculation())
        
    async def _periodic_risk_calculation(self):
        """Periodic risk metrics calculation"""
        while self.running:
            try:
                if len(self.portfolio_returns) > 10:  # Need minimum data
                    risk_metrics = self.calculate_risk_metrics()
                    if risk_metrics:
                        self.risk_metrics_history.append(risk_metrics)
                        
                        # Check for risk alerts
                        await self._check_risk_alerts(risk_metrics)
                        
            except Exception as e:
                logger.error(f"Error in periodic risk calculation: {e}")
                
            await asyncio.sleep(self.update_frequency)
            
    async def process_market_tick(self, tick: MarketTick):
        """Process incoming market tick"""
        symbol = tick.symbol
        price = tick.price
        timestamp = tick.timestamp
        
        # Update price data
        self.price_data[symbol].append((timestamp, price))
        
        # Calculate returns if we have previous price
        if len(self.price_data[symbol]) > 1:
            prev_price = self.price_data[symbol][-2][1]
            return_val = (price / prev_price) - 1
            self.return_data[symbol].append(return_val)
            
            # Update Kalman filters
            if symbol in self.trend_filters:
                self.trend_filters[symbol].predict_and_update(np.array([price]))
                
            if symbol in self.volatility_filters:
                self.volatility_filters[symbol].update_with_return(return_val)
                
        # Update portfolio value
        await self._update_portfolio_value()
        
        # Update position metrics
        self._update_position_metrics(symbol, price, timestamp)
        
    async def _update_portfolio_value(self):
        """Update total portfolio value"""
        total_value = 0.0
        
        for symbol, position_size in self.portfolio.items():
            if self.price_data[symbol]:
                latest_price = self.price_data[symbol][-1][1]
                total_value += position_size * latest_price
                
        if total_value > 0:
            # Calculate portfolio return
            if self.last_portfolio_value > 0:
                portfolio_return = (total_value / self.last_portfolio_value) - 1
                self.portfolio_returns.append(portfolio_return)
                
                # Update covariance matrix
                self._update_covariance_matrix()
                
            self.portfolio_values.append(total_value)
            self.last_portfolio_value = total_value
            
            # Update peak for drawdown calculation
            if total_value > self.peak_portfolio_value:
                self.peak_portfolio_value = total_value
                
    def _update_position_metrics(self, symbol: str, price: float, timestamp: datetime):
        """Update metrics for individual position"""
        if symbol not in self.portfolio:
            return
            
        position_size = self.portfolio[symbol]
        market_value = position_size * price
        
        # Calculate P&L (simplified)
        if symbol in self.position_metrics:
            prev_price = self.position_metrics[symbol].market_value / position_size
            unrealized_pnl = position_size * (price - prev_price)
        else:
            unrealized_pnl = 0.0
            
        # Get volatility from Kalman filter
        volatility = 0.2  # Default
        if symbol in self.volatility_filters:
            volatility = self.volatility_filters[symbol].get_volatility_estimate()
            
        # Calculate beta (simplified)
        beta = self._calculate_beta(symbol)
        
        # VaR contribution (simplified)
        var_contribution = abs(market_value) * volatility * 1.65  # 95% normal VaR
        
        self.position_metrics[symbol] = PositionMetrics(
            symbol=symbol,
            position_size=position_size,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=0.0,  # Would track actual trades
            daily_pnl=unrealized_pnl,  # Simplified
            var_contribution=var_contribution,
            volatility=volatility,
            beta=beta,
            last_update=timestamp
        )
        
    def _calculate_beta(self, symbol: str) -> float:
        """Calculate beta relative to market"""
        if len(self.return_data[symbol]) < 20 or len(self.market_returns) < 20:
            return 1.0  # Default beta
            
        # Use last 20 returns for beta calculation
        asset_returns = list(self.return_data[symbol])[-20:]
        market_returns = list(self.market_returns)[-20:]
        
        if len(asset_returns) != len(market_returns):
            return 1.0
            
        # Calculate beta using linear regression
        asset_returns = np.array(asset_returns)
        market_returns = np.array(market_returns)
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance > 0:
            return covariance / market_variance
        else:
            return 1.0
            
    def _update_covariance_matrix(self):
        """Update exponentially weighted covariance matrix"""
        symbols = list(self.portfolio.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return
            
        # Get latest returns
        latest_returns = []
        for symbol in symbols:
            if self.return_data[symbol]:
                latest_returns.append(self.return_data[symbol][-1])
            else:
                latest_returns.append(0.0)
                
        latest_returns = np.array(latest_returns)
        
        if self.return_covariance is None:
            # Initialize with sample covariance
            if all(len(self.return_data[symbol]) > 1 for symbol in symbols):
                return_matrix = np.array([
                    list(self.return_data[symbol])[-min(len(self.return_data[symbol]), 50):]
                    for symbol in symbols
                ])
                self.return_covariance = np.cov(return_matrix)
            else:
                self.return_covariance = np.eye(n_assets) * 0.0004  # Default 2% daily vol
        else:
            # Exponentially weighted update
            mean_return = np.mean([
                np.mean(list(self.return_data[symbol])[-10:]) if len(self.return_data[symbol]) >= 10 else 0
                for symbol in symbols
            ])
            
            deviation = latest_returns - mean_return
            outer_product = np.outer(deviation, deviation)
            
            self.return_covariance = (
                self.covariance_decay * self.return_covariance +
                (1 - self.covariance_decay) * outer_product
            )
            
    def calculate_risk_metrics(self) -> Optional[RiskMetrics]:
        """Calculate comprehensive risk metrics"""
        if len(self.portfolio_returns) < 10:
            return None
            
        current_time = datetime.now()
        
        # Portfolio value
        portfolio_value = self.portfolio_values[-1] if self.portfolio_values else 0.0
        
        # Convert returns to numpy array
        returns = np.array(list(self.portfolio_returns))
        
        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # VaR calculations
        var_95 = np.percentile(returns, 5) * portfolio_value
        var_99 = np.percentile(returns, 1) * portfolio_value
        
        # Expected Shortfall
        tail_returns = returns[returns <= np.percentile(returns, 5)]
        expected_shortfall = np.mean(tail_returns) * portfolio_value if len(tail_returns) > 0 else var_95
        
        # Maximum Drawdown
        if self.peak_portfolio_value > 0:
            max_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        else:
            max_drawdown = 0.0
            
        # Sharpe Ratio (annualized)
        mean_return = np.mean(returns)
        if volatility > 0:
            sharpe_ratio = (mean_return * 252 - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
            
        # Beta and correlation to market
        if len(self.market_returns) > 10:
            market_returns = np.array(list(self.market_returns)[-len(returns):])
            if len(market_returns) == len(returns):
                correlation = np.corrcoef(returns, market_returns)[0, 1]
                
                market_var = np.var(market_returns)
                if market_var > 0:
                    beta = np.cov(returns, market_returns)[0, 1] / market_var
                else:
                    beta = 1.0
            else:
                correlation = 0.0
                beta = 1.0
        else:
            correlation = 0.0
            beta = 1.0
            
        # Concentration risk (Herfindahl index)
        total_abs_value = sum(
            abs(pos_size * self.price_data[symbol][-1][1])
            for symbol, pos_size in self.portfolio.items()
            if self.price_data[symbol]
        )
        
        if total_abs_value > 0:
            weights_squared = [
                (abs(pos_size * self.price_data[symbol][-1][1]) / total_abs_value) ** 2
                for symbol, pos_size in self.portfolio.items()
                if self.price_data[symbol]
            ]
            concentration_risk = sum(weights_squared)
        else:
            concentration_risk = 1.0
            
        return RiskMetrics(
            timestamp=current_time,
            portfolio_value=portfolio_value,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            correlation_to_market=correlation,
            concentration_risk=concentration_risk
        )
        
    async def _check_risk_alerts(self, risk_metrics: RiskMetrics):
        """Check for risk limit breaches"""
        alerts = []
        
        # VaR limit check (example: 5% of portfolio value)
        var_limit = risk_metrics.portfolio_value * 0.05
        if abs(risk_metrics.var_95) > var_limit:
            alerts.append(f"VaR 95% breach: {risk_metrics.var_95:.2f} > {var_limit:.2f}")
            
        # Maximum drawdown check (example: 10%)
        if risk_metrics.max_drawdown > 0.10:
            alerts.append(f"Maximum drawdown breach: {risk_metrics.max_drawdown:.2%}")
            
        # Concentration risk check (example: Herfindahl > 0.4)
        if risk_metrics.concentration_risk > 0.4:
            alerts.append(f"High concentration risk: {risk_metrics.concentration_risk:.3f}")
            
        # Volatility check (example: > 50% annualized)
        if risk_metrics.volatility > 0.5:
            alerts.append(f"High volatility: {risk_metrics.volatility:.2%}")
            
        # Log alerts
        for alert in alerts:
            logger.warning(f"RISK ALERT: {alert}")
            
    def get_current_risk_metrics(self) -> Optional[RiskMetrics]:
        """Get most recent risk metrics"""
        return self.risk_metrics_history[-1] if self.risk_metrics_history else None
        
    def get_position_metrics(self) -> Dict[str, PositionMetrics]:
        """Get current position metrics"""
        return self.position_metrics.copy()
        
    def update_portfolio(self, new_portfolio: Dict[str, float]):
        """Update portfolio positions"""
        self.portfolio = new_portfolio.copy()
        
        # Initialize new filters if needed
        for symbol in new_portfolio.keys():
            if symbol not in self.trend_filters:
                self.trend_filters[symbol] = TrendFollowingKalman()
                self.volatility_filters[symbol] = VolatilityKalman()
                
    def get_risk_decomposition(self) -> Dict[str, float]:
        """Get risk contribution by position"""
        if self.return_covariance is None or len(self.portfolio) == 0:
            return {}
            
        symbols = list(self.portfolio.keys())
        positions = np.array([self.portfolio[symbol] for symbol in symbols])
        
        # Get latest prices
        prices = []
        for symbol in symbols:
            if self.price_data[symbol]:
                prices.append(self.price_data[symbol][-1][1])
            else:
                prices.append(100.0)  # Default price
                
        prices = np.array(prices)
        
        # Market values
        market_values = positions * prices
        total_value = np.sum(np.abs(market_values))
        
        if total_value == 0:
            return {}
            
        # Weights
        weights = market_values / total_value
        
        # Portfolio variance
        portfolio_var = weights.T @ self.return_covariance @ weights
        
        if portfolio_var <= 0:
            return {}
            
        # Marginal contributions
        marginal_contributions = self.return_covariance @ weights
        
        # Risk contributions
        risk_contributions = weights * marginal_contributions / portfolio_var
        
        return dict(zip(symbols, risk_contributions))
        
    async def stop_analytics(self):
        """Stop real-time analytics"""
        self.running = False
        logger.info("Stopped real-time risk analytics")

class RealTimePortfolioManager:
    """Real-time portfolio management system"""
    
    def __init__(
        self,
        initial_portfolio: Dict[str, float],
        initial_cash: float = 0.0,
        transaction_cost: float = 0.001
    ):
        self.portfolio = initial_portfolio.copy()
        self.cash = initial_cash
        self.transaction_cost = transaction_cost
        
        # Trading history
        self.trade_history = []
        self.pnl_history = deque(maxlen=1000)
        
        # Risk analytics
        self.risk_analytics = RealTimeRiskAnalytics(initial_portfolio)
        
        # Current market prices
        self.current_prices = {}
        
        # Portfolio optimization
        self.target_weights = None
        self.rebalance_threshold = 0.05  # 5% drift threshold
        
    async def start_management(self):
        """Start portfolio management"""
        await self.risk_analytics.start_analytics()
        logger.info("Started real-time portfolio management")
        
    async def process_market_update(self, tick: MarketTick):
        """Process market data update"""
        self.current_prices[tick.symbol] = tick.price
        
        # Update risk analytics
        await self.risk_analytics.process_market_tick(tick)
        
        # Check for rebalancing needs
        if self.target_weights:
            await self._check_rebalancing_needs()
            
    async def execute_trade(
        self,
        symbol: str,
        quantity: float,
        trade_type: str = 'market'
    ) -> bool:
        """
        Execute trade
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade (positive for buy, negative for sell)
            trade_type: Type of trade ('market', 'limit')
            
        Returns:
            Success status
        """
        if symbol not in self.current_prices:
            logger.error(f"No price available for {symbol}")
            return False
            
        price = self.current_prices[symbol]
        trade_value = abs(quantity * price)
        cost = trade_value * self.transaction_cost
        
        # Check if we have enough cash for buy orders
        if quantity > 0 and self.cash < trade_value + cost:
            logger.error(f"Insufficient cash for trade: need {trade_value + cost}, have {self.cash}")
            return False
            
        # Check if we have enough shares for sell orders
        current_position = self.portfolio.get(symbol, 0.0)
        if quantity < 0 and abs(quantity) > current_position:
            logger.error(f"Insufficient shares for sale: trying to sell {abs(quantity)}, have {current_position}")
            return False
            
        # Execute trade
        self.portfolio[symbol] = current_position + quantity
        self.cash -= quantity * price + cost
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'cost': cost,
            'cash_after': self.cash
        }
        
        self.trade_history.append(trade_record)
        
        # Update risk analytics portfolio
        self.risk_analytics.update_portfolio(self.portfolio)
        
        logger.info(f"Executed trade: {quantity} {symbol} @ {price}")
        return True
        
    async def _check_rebalancing_needs(self):
        """Check if portfolio needs rebalancing"""
        if not self.target_weights:
            return
            
        # Calculate current weights
        current_weights = self._calculate_current_weights()
        
        # Check drift from target
        max_drift = 0.0
        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)
            
        # Rebalance if drift exceeds threshold
        if max_drift > self.rebalance_threshold:
            await self._rebalance_portfolio()
            
    def _calculate_current_weights(self) -> Dict[str, float]:
        """Calculate current portfolio weights"""
        total_value = self._calculate_portfolio_value()
        
        if total_value <= 0:
            return {}
            
        weights = {}
        for symbol, quantity in self.portfolio.items():
            if symbol in self.current_prices and quantity != 0:
                market_value = quantity * self.current_prices[symbol]
                weights[symbol] = market_value / total_value
                
        return weights
        
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, quantity in self.portfolio.items():
            if symbol in self.current_prices:
                total_value += quantity * self.current_prices[symbol]
                
        return total_value
        
    async def _rebalance_portfolio(self):
        """Rebalance portfolio to target weights"""
        if not self.target_weights:
            return
            
        logger.info("Starting portfolio rebalancing")
        
        total_value = self._calculate_portfolio_value()
        current_weights = self._calculate_current_weights()
        
        # Calculate required trades
        for symbol, target_weight in self.target_weights.items():
            if symbol not in self.current_prices:
                continue
                
            target_value = total_value * target_weight
            current_value = current_weights.get(symbol, 0.0) * total_value
            
            trade_value = target_value - current_value
            
            if abs(trade_value) > total_value * 0.01:  # Only trade if significant
                price = self.current_prices[symbol]
                quantity = trade_value / price
                
                await self.execute_trade(symbol, quantity)
                
        logger.info("Portfolio rebalancing completed")
        
    def set_target_weights(self, target_weights: Dict[str, float]):
        """Set target portfolio weights"""
        # Normalize weights to sum to 1
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            self.target_weights = {
                symbol: weight / total_weight
                for symbol, weight in target_weights.items()
            }
        else:
            self.target_weights = target_weights
            
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        total_value = self._calculate_portfolio_value()
        current_weights = self._calculate_current_weights()
        
        # Calculate daily P&L
        daily_pnl = 0.0
        if len(self.pnl_history) > 0:
            daily_pnl = total_value - self.pnl_history[-1] if self.pnl_history[-1] != 0 else 0.0
            
        self.pnl_history.append(total_value)
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'positions': self.portfolio.copy(),
            'weights': current_weights,
            'daily_pnl': daily_pnl,
            'num_trades_today': len([
                trade for trade in self.trade_history
                if trade['timestamp'].date() == datetime.now().date()
            ]),
            'risk_metrics': self.risk_analytics.get_current_risk_metrics()
        }
        
    async def stop_management(self):
        """Stop portfolio management"""
        await self.risk_analytics.stop_analytics()
        logger.info("Stopped real-time portfolio management")

# Example usage and testing
if __name__ == "__main__":
    print("Testing Real-time Analytics...")
    
    async def test_real_time_analytics():
        # Test portfolio
        portfolio = {
            'AAPL': 100.0,
            'GOOGL': 50.0,
            'MSFT': 75.0
        }
        
        # Initialize risk analytics
        risk_analytics = RealTimeRiskAnalytics(
            portfolio_data=portfolio,
            risk_free_rate=0.02,
            lookback_window=100
        )
        
        await risk_analytics.start_analytics()
        
        # Simulate market ticks
        symbols = list(portfolio.keys())
        base_prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 330.0}
        
        print("Simulating market data...")
        
        for i in range(50):
            for symbol in symbols:
                # Generate random price movement
                price_change = np.random.normal(0, 0.02)  # 2% volatility
                new_price = base_prices[symbol] * (1 + price_change)
                base_prices[symbol] = new_price
                
                # Create market tick
                tick = MarketTick(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=new_price,
                    volume=1000 + np.random.randint(0, 500)
                )
                
                await risk_analytics.process_market_tick(tick)
                
            await asyncio.sleep(0.1)  # Small delay
            
        # Get risk metrics
        risk_metrics = risk_analytics.get_current_risk_metrics()
        if risk_metrics:
            print(f"\nRisk Metrics:")
            print(f"  Portfolio Value: ${risk_metrics.portfolio_value:,.2f}")
            print(f"  VaR 95%: ${risk_metrics.var_95:,.2f}")
            print(f"  Volatility: {risk_metrics.volatility:.2%}")
            print(f"  Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {risk_metrics.max_drawdown:.2%}")
            
        # Get position metrics
        position_metrics = risk_analytics.get_position_metrics()
        print(f"\nPosition Metrics:")
        for symbol, metrics in position_metrics.items():
            print(f"  {symbol}: Value=${metrics.market_value:,.2f}, "
                  f"Vol={metrics.volatility:.2%}, Beta={metrics.beta:.2f}")
                  
        await risk_analytics.stop_analytics()
        
        print("\nReal-time analytics test completed!")
        
    # Test Portfolio Manager
    async def test_portfolio_manager():
        portfolio = {'AAPL': 100.0, 'GOOGL': 50.0}
        initial_cash = 50000.0
        
        pm = RealTimePortfolioManager(
            initial_portfolio=portfolio,
            initial_cash=initial_cash
        )
        
        await pm.start_management()
        
        # Simulate market updates
        test_tick = MarketTick(
            symbol='AAPL',
            timestamp=datetime.now(),
            price=155.0,
            volume=1000
        )
        
        await pm.process_market_update(test_tick)
        
        # Execute a trade
        success = await pm.execute_trade('AAPL', 10.0)  # Buy 10 shares
        print(f"Trade execution success: {success}")
        
        # Get portfolio summary
        summary = pm.get_portfolio_summary()
        print(f"Portfolio Summary:")
        print(f"  Total Value: ${summary['total_value']:,.2f}")
        print(f"  Cash: ${summary['cash']:,.2f}")
        print(f"  Positions: {summary['positions']}")
        
        await pm.stop_management()
        
        print("Portfolio manager test completed!")
    
    # Run tests
    asyncio.run(test_real_time_analytics())
    asyncio.run(test_portfolio_manager())
