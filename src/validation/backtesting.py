"""
Comprehensive backtesting framework for trading strategies and risk models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    action: str  # 'buy', 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    @property
    def value(self) -> float:
        return self.quantity * self.price
        
    @property
    def total_cost(self) -> float:
        return abs(self.value) + self.commission + abs(self.value * self.slippage)

@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float
    num_trades: int
    win_rate: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    monthly_returns: pd.Series = field(default_factory=pd.Series)

class Portfolio:
    """Portfolio state management for backtesting"""
    
    def __init__(self, initial_capital: float, commission_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        slippage_rate: float = 0.0
    ) -> bool:
        """
        Execute a trade and update portfolio state
        
        Args:
            timestamp: Trade timestamp
            symbol: Asset symbol
            action: 'buy' or 'sell'
            quantity: Number of shares (positive)
            price: Execution price
            slippage_rate: Slippage as fraction of trade value
            
        Returns:
            Success status
        """
        # Calculate costs
        trade_value = quantity * price
        commission = trade_value * self.commission_rate
        slippage = trade_value * slippage_rate
        total_cost = commission + slippage
        
        # Check if we have enough cash for buy orders
        if action == 'buy':
            required_cash = trade_value + total_cost
            if self.cash < required_cash:
                logger.warning(f"Insufficient cash for trade: need {required_cash:.2f}, have {self.cash:.2f}")
                return False
                
        # Check if we have enough shares for sell orders
        elif action == 'sell':
            current_position = self.positions.get(symbol, Position(symbol, 0, 0))
            if current_position.quantity < quantity:
                logger.warning(f"Insufficient shares for sale: trying to sell {quantity}, have {current_position.quantity}")
                return False
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage
        )
        
        # Update portfolio
        self._update_position(symbol, action, quantity, price)
        
        # Update cash
        if action == 'buy':
            self.cash -= (trade_value + total_cost)
        else:  # sell
            self.cash += (trade_value - total_cost)
            
        self.trades.append(trade)
        return True
        
    def _update_position(self, symbol: str, action: str, quantity: float, price: float):
        """Update position after trade"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0)
            
        position = self.positions[symbol]
        
        if action == 'buy':
            # Update average price
            if position.quantity > 0:
                total_value = position.quantity * position.avg_price + quantity * price
                total_quantity = position.quantity + quantity
                position.avg_price = total_value / total_quantity
            else:
                position.avg_price = price
                
            position.quantity += quantity
            
        elif action == 'sell':
            # Calculate realized P&L
            realized_pnl = quantity * (price - position.avg_price)
            position.realized_pnl += realized_pnl
            position.quantity -= quantity
            
            # Remove position if fully closed
            if position.quantity <= 1e-8:  # Account for floating point precision
                self.positions.pop(symbol, None)
                
    def update_market_prices(self, timestamp: datetime, prices: Dict[str, float]):
        """Update portfolio with current market prices"""
        # Update unrealized P&L for all positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.unrealized_pnl = position.quantity * (current_price - position.avg_price)
                
        # Calculate total portfolio value
        total_value = self.cash
        for position in self.positions.values():
            if position.quantity > 0:
                total_value += position.market_value + position.unrealized_pnl
                
        self.equity_history.append((timestamp, total_value))
        
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Get current portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in prices and position.quantity > 0:
                market_value = position.quantity * prices[symbol]
                total_value += market_value
                
        return total_value
        
    def get_positions_summary(self) -> pd.DataFrame:
        """Get summary of current positions"""
        if not self.positions:
            return pd.DataFrame()
            
        data = []
        for position in self.positions.values():
            data.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'avg_price': position.avg_price,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            })
            
        return pd.DataFrame(data)

class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_free_rate: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        
        self.portfolio = None
        self.market_data: Optional[pd.DataFrame] = None
        self.strategy_func: Optional[Callable] = None
        
    def set_market_data(self, data: pd.DataFrame):
        """Set market data for backtesting"""
        required_columns = ['close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Market data must contain columns: {required_columns}")
            
        self.market_data = data.copy()
        logger.info(f"Market data set: {len(data)} rows, {len(data.columns)} columns")
        
    def set_strategy(self, strategy_func: Callable):
        """
        Set trading strategy function
        
        Args:
            strategy_func: Function that takes (data, portfolio, timestamp) and returns trade signals
        """
        self.strategy_func = strategy_func
        
    def run_backtest(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    ) -> BacktestResult:
        """
        Run backtest simulation
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: How often to rebalance
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        if self.market_data is None:
            raise ValueError("Market data not set")
        if self.strategy_func is None:
            raise ValueError("Strategy function not set")
            
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_capital, self.commission_rate)
        
        # Filter data by date range
        data = self.market_data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        if len(data) == 0:
            raise ValueError("No data available for specified date range")
            
        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        
        # Determine rebalancing dates
        rebalance_dates = self._get_rebalance_dates(data.index, rebalance_frequency)
        
        # Run simulation
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Get current prices
            if isinstance(row, pd.Series):
                prices = {'close': row['close']}
                # Add other assets if multi-asset
                for col in data.columns:
                    if 'close' in col.lower() and col != 'close':
                        symbol = col.replace('_close', '').replace('close_', '')
                        prices[symbol] = row[col]
            else:
                prices = {'close': float(row)}
                
            # Update portfolio with current prices
            self.portfolio.update_market_prices(timestamp, prices)
            
            # Execute strategy on rebalancing dates
            if timestamp in rebalance_dates or i == 0:  # Always execute on first day
                try:
                    # Get strategy signals
                    signals = self.strategy_func(
                        data.iloc[:i+1],  # Historical data up to current point
                        self.portfolio,
                        timestamp
                    )
                    
                    # Execute trades based on signals
                    self._execute_signals(timestamp, signals, prices)
                    
                except Exception as e:
                    logger.error(f"Strategy execution error at {timestamp}: {e}")
                    continue
                    
        # Calculate backtest results
        result = self._calculate_results(data.index[0], data.index[-1])
        
        logger.info(f"Backtest completed: {result.total_return:.2%} total return, "
                   f"{result.sharpe_ratio:.3f} Sharpe ratio")
        
        return result
        
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex, frequency: str) -> List[datetime]:
        """Get rebalancing dates based on frequency"""
        if frequency == 'daily':
            return date_index.tolist()
        elif frequency == 'weekly':
            return date_index[date_index.weekday == 0].tolist()  # Mondays
        elif frequency == 'monthly':
            return date_index[date_index.is_month_end].tolist()
        else:
            raise ValueError(f"Unknown rebalancing frequency: {frequency}")
            
    def _execute_signals(self, timestamp: datetime, signals: Dict, prices: Dict):
        """Execute trading signals"""
        for symbol, signal in signals.items():
            if isinstance(signal, dict):
                action = signal.get('action')
                quantity = signal.get('quantity', 0)
                price = signal.get('price', prices.get(symbol, prices.get('close', 0)))
            else:
                # Simple signal format: positive = buy, negative = sell
                if signal > 0:
                    action = 'buy'
                    quantity = abs(signal)
                elif signal < 0:
                    action = 'sell' 
                    quantity = abs(signal)
                else:
                    continue  # No action
                    
                price = prices.get(symbol, prices.get('close', 0))
                
            if quantity > 0 and price > 0:
                success = self.portfolio.execute_trade(
                    timestamp, symbol, action, quantity, price, self.slippage_rate
                )
                
                if success:
                    logger.debug(f"Executed: {action} {quantity} {symbol} @ {price}")
                    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        if not self.portfolio.equity_history:
            raise ValueError("No equity history available")
            
        # Create equity curve
        equity_df = pd.DataFrame(self.portfolio.equity_history, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        equity_curve = equity_df['equity']
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital / self.initial_capital) - 1
        
        # Annualized return
        trading_days = len(equity_curve)
        years = trading_days / 252
        annualized_return = (final_capital / self.initial_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown analysis
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # VaR 95%
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Trading statistics
        trades = self.portfolio.trades
        num_trades = len(trades)
        
        if num_trades > 0:
            # Calculate trade P&L
            trade_pnls = []
            for i, trade in enumerate(trades):
                if trade.action == 'sell' and i > 0:
                    # Find corresponding buy trade (simplified)
                    for j in range(i-1, -1, -1):
                        if (trades[j].symbol == trade.symbol and 
                            trades[j].action == 'buy'):
                            pnl = (trade.price - trades[j].price) * trade.quantity
                            trade_pnls.append(pnl)
                            break
                            
            if trade_pnls:
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
                
                win_rate = len(winning_trades) / len(trade_pnls)
                
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
            
        # Monthly returns
        equity_monthly = equity_curve.resample('M').last()
        monthly_returns = equity_monthly.pct_change().dropna()
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_series=drawdown,
            monthly_returns=monthly_returns
        )

class PerformanceAnalyzer:
    """Advanced performance analysis for backtest results"""
    
    def __init__(self):
        pass
        
    def analyze_performance(self, backtest_result: BacktestResult) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        analysis = {}
        
        # Return analysis
        analysis['returns'] = self._analyze_returns(backtest_result)
        
        # Risk analysis
        analysis['risk'] = self._analyze_risk(backtest_result)
        
        # Drawdown analysis
        analysis['drawdowns'] = self._analyze_drawdowns(backtest_result)
        
        # Trading analysis
        analysis['trading'] = self._analyze_trading(backtest_result)
        
        # Benchmark comparison (if applicable)
        analysis['benchmark'] = self._benchmark_analysis(backtest_result)
        
        return analysis
        
    def _analyze_returns(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze return characteristics"""
        if len(result.monthly_returns) == 0:
            return {}
            
        returns_analysis = {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'monthly_volatility': result.monthly_returns.std(),
            'best_month': result.monthly_returns.max(),
            'worst_month': result.monthly_returns.min(),
            'positive_months': (result.monthly_returns > 0).sum(),
            'negative_months': (result.monthly_returns < 0).sum(),
            'skewness': result.monthly_returns.skew(),
            'kurtosis': result.monthly_returns.kurtosis()
        }
        
        return returns_analysis
        
    def _analyze_risk(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze risk metrics"""
        risk_analysis = {
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'calmar_ratio': result.calmar_ratio,
            'var_95': result.var_95,
            'max_drawdown': result.max_drawdown,
            'volatility': result.volatility
        }
        
        # Additional risk metrics if we have daily returns
        if len(result.equity_curve) > 1:
            daily_returns = result.equity_curve.pct_change().dropna()
            
            # Value at Risk (multiple confidence levels)
            risk_analysis['var_99'] = np.percentile(daily_returns, 1)
            risk_analysis['var_90'] = np.percentile(daily_returns, 10)
            
            # Expected Shortfall
            var_95_threshold = np.percentile(daily_returns, 5)
            tail_returns = daily_returns[daily_returns <= var_95_threshold]
            risk_analysis['expected_shortfall_95'] = tail_returns.mean() if len(tail_returns) > 0 else 0
            
            # Maximum consecutive losing days
            losing_days = (daily_returns < 0).astype(int)
            risk_analysis['max_consecutive_losses'] = self._max_consecutive_ones(losing_days.values)
            
        return risk_analysis
        
    def _analyze_drawdowns(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze drawdown characteristics"""
        if len(result.drawdown_series) == 0:
            return {}
            
        drawdowns = result.drawdown_series
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_starts = drawdowns.index[in_drawdown & ~in_drawdown.shift(1, fill_value=False)]
        drawdown_ends = drawdowns.index[~in_drawdown & in_drawdown.shift(1, fill_value=False)]
        
        if len(drawdown_starts) == 0:
            return {'max_drawdown': 0, 'avg_drawdown_duration': 0, 'num_drawdowns': 0}
            
        # Analyze each drawdown period
        drawdown_durations = []
        drawdown_depths = []
        
        for i, start in enumerate(drawdown_starts):
            if i < len(drawdown_ends):
                end = drawdown_ends[i]
                duration = (end - start).days
                depth = drawdowns[start:end].min()
            else:
                # Ongoing drawdown
                duration = (drawdowns.index[-1] - start).days
                depth = drawdowns[start:].min()
                
            drawdown_durations.append(duration)
            drawdown_depths.append(depth)
            
        drawdown_analysis = {
            'max_drawdown': result.max_drawdown,
            'avg_drawdown': np.mean(drawdown_depths),
            'max_drawdown_duration': max(drawdown_durations) if drawdown_durations else 0,
            'avg_drawdown_duration': np.mean(drawdown_durations) if drawdown_durations else 0,
            'num_drawdowns': len(drawdown_starts),
            'recovery_factor': abs(result.total_return / result.max_drawdown) if result.max_drawdown < 0 else 0
        }
        
        return drawdown_analysis
        
    def _analyze_trading(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze trading behavior"""
        trading_analysis = {
            'num_trades': result.num_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor
        }
        
        if result.trades:
            # Trade frequency
            days = (result.end_date - result.start_date).days
            trading_analysis['trades_per_month'] = result.num_trades / (days / 30) if days > 0 else 0
            
            # Average holding periods (simplified)
            buy_trades = [t for t in result.trades if t.action == 'buy']
            sell_trades = [t for t in result.trades if t.action == 'sell']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Match buy and sell trades (simplified)
                holding_periods = []
                for sell_trade in sell_trades:
                    matching_buys = [t for t in buy_trades 
                                   if t.symbol == sell_trade.symbol and t.timestamp < sell_trade.timestamp]
                    if matching_buys:
                        buy_trade = matching_buys[-1]  # Last matching buy
                        holding_period = (sell_trade.timestamp - buy_trade.timestamp).days
                        holding_periods.append(holding_period)
                        
                if holding_periods:
                    trading_analysis['avg_holding_period'] = np.mean(holding_periods)
                    
        return trading_analysis
        
    def _benchmark_analysis(self, result: BacktestResult) -> Dict[str, Any]:
        """Compare against benchmark (placeholder)"""
        # This would compare against a benchmark like S&P 500
        # For now, return placeholder analysis
        return {
            'alpha': 0.0,  # Would calculate actual alpha
            'beta': 1.0,   # Would calculate actual beta
            'tracking_error': 0.0,
            'information_ratio': 0.0
        }
        
    def _max_consecutive_ones(self, binary_array: np.ndarray) -> int:
        """Find maximum consecutive 1s in binary array"""
        max_count = 0
        current_count = 0
        
        for value in binary_array:
            if value == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
                
        return max_count

# Example usage and testing
if __name__ == "__main__":
    print("Testing Backtesting Framework...")
    
    # Generate synthetic market data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Generate realistic price data with trend and volatility
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.015, n_days)  # Daily returns
    prices = [100.0]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'close': prices[1:],
        'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices[1:]],
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices[1:]],
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    # Ensure OHLC consistency
    market_data['high'] = np.maximum.reduce([
        market_data['open'], market_data['close'], market_data['high']
    ])
    market_data['low'] = np.minimum.reduce([
        market_data['open'], market_data['close'], market_data['low']
    ])
    
    print(f"Generated {len(market_data)} days of market data")
    
    # Define a simple momentum strategy
    def momentum_strategy(data, portfolio, timestamp):
        """Simple momentum strategy for testing"""
        if len(data) < 21:  # Need at least 20 days for calculation
            return {}
            
        # Calculate 20-day momentum
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-21]
        momentum = (current_price / past_price) - 1
        
        # Get current portfolio value
        current_prices = {'close': current_price}
        portfolio_value = portfolio.get_portfolio_value(current_prices)
        
        signals = {}
        
        # Simple momentum signal
        if momentum > 0.05:  # Strong positive momentum
            # Buy signal - invest 50% of portfolio value
            target_value = portfolio_value * 0.5
            quantity = target_value / current_price
            signals['close'] = {'action': 'buy', 'quantity': quantity}
            
        elif momentum < -0.05:  # Strong negative momentum
            # Sell signal - liquidate position
            current_position = portfolio.positions.get('close')
            if current_position and current_position.quantity > 0:
                signals['close'] = {'action': 'sell', 'quantity': current_position.quantity}
                
        return signals
    
    # Initialize and run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    engine.set_market_data(market_data)
    engine.set_strategy(momentum_strategy)
    
    # Run backtest
    start_date = datetime(2020, 6, 1)
    end_date = datetime(2023, 6, 1)
    
    result = engine.run_backtest(
        start_date=start_date,
        end_date=end_date,
        rebalance_frequency='weekly'
    )
    
    # Display results
    print(f"\nBacktest Results:")
    print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f"Initial Capital: ${result.initial_capital:,.2f}")
    print(f"Final Capital: ${result.final_capital:,.2f}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
    print(f"Calmar Ratio: {result.calmar_ratio:.3f}")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    
    # Test performance analyzer
    print("\nRunning Performance Analysis...")
    analyzer = PerformanceAnalyzer()
    performance_analysis = analyzer.analyze_performance(result)
    
    print(f"\nReturn Analysis:")
    returns_analysis = performance_analysis['returns']
    print(f"  Best Month: {returns_analysis.get('best_month', 0):.2%}")
    print(f"  Worst Month: {returns_analysis.get('worst_month', 0):.2%}")
    print(f"  Positive Months: {returns_analysis.get('positive_months', 0)}")
    print(f"  Negative Months: {returns_analysis.get('negative_months', 0)}")
    
    print(f"\nRisk Analysis:")
    risk_analysis = performance_analysis['risk']
    print(f"  VaR 95%: {risk_analysis.get('var_95', 0):.4f}")
    print(f"  VaR 99%: {risk_analysis.get('var_99', 0):.4f}")
    print(f"  Expected Shortfall 95%: {risk_analysis.get('expected_shortfall_95', 0):.4f}")
    
    print(f"\nDrawdown Analysis:")
    drawdown_analysis = performance_analysis['drawdowns']
    print(f"  Number of Drawdowns: {drawdown_analysis.get('num_drawdowns', 0)}")
    print(f"  Average Drawdown: {drawdown_analysis.get('avg_drawdown', 0):.2%}")
    print(f"  Max Drawdown Duration: {drawdown_analysis.get('max_drawdown_duration', 0)} days")
    
    print(f"\nTrading Analysis:")
    trading_analysis = performance_analysis['trading']
    print(f"  Trades per Month: {trading_analysis.get('trades_per_month', 0):.1f}")
    print(f"  Average Holding Period: {trading_analysis.get('avg_holding_period', 0):.1f} days")
    
    print("\nBacktesting framework test completed!")
