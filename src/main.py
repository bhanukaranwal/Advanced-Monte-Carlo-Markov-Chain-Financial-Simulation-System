"""
Main system orchestrator for the Advanced MC-Markov Finance system
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .data_engine import MarketDataIngester, FeatureEngineer
from .markov_engine import AdaptiveMarkovChain
from .monte_carlo_engine import MonteCarloEngine, VarianceReduction
from .analytics_engine import RiskAnalytics, RegimeDetection
from .real_time_engine import StreamProcessor
from .visualization import DashboardGenerator
from config import load_config, get_config

logger = logging.getLogger(__name__)

@dataclass
class SimulationResults:
    """Container for simulation results"""
    # Basic results
    paths: np.ndarray
    final_values: np.ndarray
    returns: np.ndarray
    
    # Risk metrics
    var_95: float
    var_99: float
    var_999: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    max_drawdown: float
    
    # Performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Distribution characteristics
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float
    
    # Simulation metadata
    n_paths: int
    n_steps: int
    time_horizon_days: int
    computation_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        result['paths'] = self.paths.tolist() if self.paths is not None else None
        result['final_values'] = self.final_values.tolist()
        result['returns'] = self.returns.tolist()
        return result
        
    def save(self, filepath: str):
        """Save results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class AdvancedMCMarkovSystem:
    """Main system orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = load_config('config')
        if config_path:
            with open(config_path, 'r') as f:
                import yaml
                custom_config = yaml.safe_load(f)
                self._merge_configs(self.config, custom_config)
        
        # Initialize components
        self.data_ingester = MarketDataIngester(self.config['data_sources'])
        self.feature_engineer = FeatureEngineer(self.config.get('feature_engineering', {}))
        self.monte_carlo_engine = MonteCarloEngine(self.config['monte_carlo'])
        self.risk_analytics = RiskAnalytics(self.config['risk_metrics'])
        self.regime_detector = RegimeDetection()
        
        # Optional components
        self.stream_processor = None
        self.dashboard = None
        
        # State
        self.markov_chains: Dict[str, AdaptiveMarkovChain] = {}
        self.trained_models: Dict[str, Any] = {}
        
        logger.info("Advanced MC-Markov system initialized")
        
    def _merge_configs(self, base_config: Dict, custom_config: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in custom_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
    
    async def run_simulation(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        simulation_paths: int = None,
        time_horizon: int = None,
        config: Optional[Dict] = None
    ) -> SimulationResults:
        """
        Run complete Monte Carlo-Markov simulation
        
        Args:
            symbols: List of financial instruments
            start_date: Historical data start date
            end_date: Historical data end date  
            simulation_paths: Number of Monte Carlo paths
            time_horizon: Simulation time horizon in days
            config: Custom configuration overrides
            
        Returns:
            SimulationResults object
        """
        start_time = datetime.now()
        
        # Override config if provided
        sim_config = self.config.copy()
        if config:
            self._merge_configs(sim_config, config)
            
        # Set defaults
        simulation_paths = simulation_paths or sim_config['simulation']['default_paths']
        time_horizon = time_horizon or sim_config['simulation']['default_steps']
        
        logger.info(f"Starting simulation for {symbols} with {simulation_paths} paths")
        
        try:
            # Step 1: Data ingestion and preprocessing
            logger.info("Step 1: Data ingestion...")
            market_data = await self._ingest_data(symbols, start_date, end_date)
            
            # Step 2: Feature engineering
            logger.info("Step 2: Feature engineering...")
            features = self._engineer_features(market_data)
            
            # Step 3: Regime detection and model training
            logger.info("Step 3: Training models...")
            await self._train_models(features, symbols)
            
            # Step 4: Monte Carlo simulation
            logger.info("Step 4: Running simulation...")
            simulation_results = await self._run_monte_carlo(
                symbols, simulation_paths, time_horizon, sim_config
            )
            
            # Step 5: Risk analytics
            logger.info("Step 5: Computing risk metrics...")
            risk_metrics = self._compute_risk_metrics(simulation_results)
            
            # Step 6: Compile results
            computation_time = (datetime.now() - start_time).total_seconds()
            
            results = SimulationResults(
                paths=simulation_results.get('paths'),
                final_values=simulation_results['final_values'],
                returns=simulation_results['returns'],
                var_95=risk_metrics['var_95'],
                var_99=risk_metrics['var_99'], 
                var_999=risk_metrics['var_999'],
                expected_shortfall_95=risk_metrics['es_95'],
                expected_shortfall_99=risk_metrics['es_99'],
                max_drawdown=risk_metrics['max_drawdown'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                sortino_ratio=risk_metrics['sortino_ratio'],
                calmar_ratio=risk_metrics['calmar_ratio'],
                mean_return=risk_metrics['mean_return'],
                volatility=risk_metrics['volatility'],
                skewness=risk_metrics['skewness'],
                kurtosis=risk_metrics['kurtosis'],
                n_paths=simulation_paths,
                n_steps=time_horizon,
                time_horizon_days=time_horizon,
                computation_time=computation_time
            )
            
            logger.info(f"Simulation completed in {computation_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    async def _ingest_data(
        self, 
        symbols: List[str], 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime]
    ) -> Dict[str, pd.DataFrame]:
        """Ingest market data for all symbols"""
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
            
        market_data = await self.data_ingester.ingest_historical_data(
            symbols, start_date, end_date
        )
        
        # Validate data quality
        for symbol, data in market_data.items():
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            logger.info(f"Loaded {len(data)} records for {symbol}")
            
        return market_data
    
    def _engineer_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features for all symbols"""
        features = {}
        
        for symbol, data in market_data.items():
            logger.info(f"Engineering features for {symbol}")
            
            # Technical indicators
            symbol_features = self.feature_engineer.extract_technical_indicators(data)
            
            # Returns and volatility
            symbol_features = self.feature_engineer.calculate_returns(symbol_features)
            symbol_features = self.feature_engineer.calculate_volatility_features(symbol_features)
            
            # Risk features
            symbol_features = self.feature_engineer.calculate_risk_features(symbol_features)
            
            # Regime features
            symbol_features = self.feature_engineer.extract_regime_features(symbol_features)
            
            # Interaction features
            symbol_features = self.feature_engineer.create_interaction_features(symbol_features)
            
            features[symbol] = symbol_features
            
        return features
    
    async def _train_models(self, features: Dict[str, pd.DataFrame], symbols: List[str]):
        """Train Markov chains and other models"""
        for symbol in symbols:
            logger.info(f"Training models for {symbol}")
            
            if symbol not in features:
                continue
                
            data = features[symbol]
            
            # Train Markov chain on returns
            if 'return_1d' in data.columns:
                returns = data['return_1d'].dropna()
                
                # Get symbol-specific config
                markov_config = get_config('model_params', f'markov_chain.{symbol.lower()}') or \
                               get_config('model_params', 'markov_chain.default')
                
                markov_chain = AdaptiveMarkovChain(
                    max_order=markov_config['max_order'],
                    min_observations=markov_config['min_observations'],
                    n_states=markov_config['n_states'],
                    adaptive_states=markov_config['adaptive_states']
                )
                
                markov_chain.fit(returns)
                self.markov_chains[symbol] = markov_chain
                
                logger.info(f"Markov chain fitted for {symbol}: "
                           f"{len(markov_chain.states)} states, "
                           f"order {markov_chain.optimal_order}")
    
    async def _run_monte_carlo(
        self, 
        symbols: List[str], 
        n_paths: int, 
        n_steps: int, 
        config: Dict
    ) -> Dict[str, np.ndarray]:
        """Run Monte Carlo simulation"""
        
        # Configure Monte Carlo engine
        mc_config = config['monte_carlo']
        
        # Initialize results storage
        all_paths = []
        all_returns = []
        final_values = []
        
        for symbol in symbols:
            if symbol not in self.markov_chains:
                logger.warning(f"No Markov chain available for {symbol}, skipping")
                continue
                
            markov_chain = self.markov_chains[symbol]
            
            # Generate Monte Carlo paths using Markov transitions
            symbol_paths = []
            symbol_returns = []
            
            for path_idx in range(n_paths):
                # Start from last observed state
                if markov_chain.state_sequences:
                    initial_state = markov_chain.state_sequences[-1]
                else:
                    initial_state = 0
                    
                # Simulate state path
                state_path = markov_chain.simulate_path(initial_state, n_steps)
                
                # Convert states to return values
                return_path = markov_chain.states_to_values(state_path)
                
                # Convert returns to price path
                price_path = self._returns_to_prices(return_path, initial_price=100.0)
                
                symbol_paths.append(price_path)
                symbol_returns.append(return_path)
                
            symbol_paths = np.array(symbol_paths)
            symbol_returns = np.array(symbol_returns)
            
            # Apply variance reduction if enabled
            if mc_config.get('variance_reduction', False):
                symbol_paths, symbol_returns = self._apply_variance_reduction(
                    symbol_paths, symbol_returns, mc_config
                )
            
            all_paths.append(symbol_paths)
            all_returns.extend(symbol_returns)
            final_values.extend(symbol_paths[:, -1])
            
        # Combine results
        if all_paths:
            # For portfolio, combine paths (simple equal weighting for now)
            combined_paths = np.mean(all_paths, axis=0) if len(all_paths) > 1 else all_paths[0]
        else:
            # Fallback: geometric Brownian motion
            combined_paths = self._fallback_simulation(n_paths, n_steps)
            all_returns = [(combined_paths[i, 1:] / combined_paths[i, :-1] - 1) 
                          for i in range(n_paths)]
            final_values = combined_paths[:, -1].tolist()
        
        return {
            'paths': combined_paths,
            'returns': np.array(all_returns),
            'final_values': np.array(final_values)
        }
    
    def _returns_to_prices(self, returns: List[float], initial_price: float = 100.0) -> np.ndarray:
        """Convert return series to price series"""
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices)
    
    def _apply_variance_reduction(
        self, 
        paths: np.ndarray, 
        returns: np.ndarray, 
        config: Dict
    ) -> tuple:
        """Apply variance reduction techniques"""
        
        # Antithetic variates
        if config.get('antithetic_variates', False):
            antithetic_returns = -returns
            antithetic_paths = np.array([
                self._returns_to_prices(ret_path) for ret_path in antithetic_returns
            ])
            
            # Combine original and antithetic
            paths = (paths + antithetic_paths) / 2
            returns = (returns + antithetic_returns) / 2
        
        return paths, returns
    
    def _fallback_simulation(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Fallback geometric Brownian motion simulation"""
        logger.warning("Using fallback GBM simulation")
        
        dt = 1/252  # Daily steps
        mu = 0.08   # Expected return
        sigma = 0.2 # Volatility
        
        # Generate random paths
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = 100.0  # Initial price
        
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW[:, t]
            )
            
        return paths
    
    def _compute_risk_metrics(self, simulation_results: Dict) -> Dict[str, float]:
        """Compute comprehensive risk metrics"""
        returns = simulation_results['returns']
        final_values = simulation_results['final_values']
        
        if len(returns) == 0:
            return self._default_risk_metrics()
        
        # Flatten returns if needed
        if returns.ndim > 1:
            flat_returns = returns.flatten()
        else:
            flat_returns = returns
            
        # Remove NaN and infinite values
        clean_returns = flat_returns[np.isfinite(flat_returns)]
        
        if len(clean_returns) == 0:
            return self._default_risk_metrics()
        
        # Calculate risk metrics
        metrics = {}
        
        # VaR calculations
        metrics['var_95'] = float(np.percentile(clean_returns, 5))
        metrics['var_99'] = float(np.percentile(clean_returns, 1))
        metrics['var_999'] = float(np.percentile(clean_returns, 0.1))
        
        # Expected Shortfall
        var_95_threshold = metrics['var_95']
        var_99_threshold = metrics['var_99']
        
        tail_returns_95 = clean_returns[clean_returns <= var_95_threshold]
        tail_returns_99 = clean_returns[clean_returns <= var_99_threshold]
        
        metrics['es_95'] = float(np.mean(tail_returns_95)) if len(tail_returns_95) > 0 else metrics['var_95']
        metrics['es_99'] = float(np.mean(tail_returns_99)) if len(tail_returns_99) > 0 else metrics['var_99']
        
        # Basic statistics
        metrics['mean_return'] = float(np.mean(clean_returns))
        metrics['volatility'] = float(np.std(clean_returns))
        metrics['skewness'] = float(self._safe_skewness(clean_returns))
        metrics['kurtosis'] = float(self._safe_kurtosis(clean_returns))
        
        # Performance ratios
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_return'] / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
            
        # Sortino ratio (downside deviation)
        downside_returns = clean_returns[clean_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else metrics['volatility']
        metrics['sortino_ratio'] = metrics['mean_return'] / downside_deviation if downside_deviation > 0 else 0.0
        
        # Maximum drawdown (simplified)
        if simulation_results.get('paths') is not None:
            paths = simulation_results['paths']
            if paths.ndim == 2:
                # Calculate drawdown for each path and take maximum
                max_dd = 0
                for path in paths:
                    running_max = np.maximum.accumulate(path)
                    drawdowns = (path - running_max) / running_max
                    max_dd = min(max_dd, np.min(drawdowns))
                metrics['max_drawdown'] = float(max_dd)
            else:
                metrics['max_drawdown'] = -0.1  # Default value
        else:
            metrics['max_drawdown'] = -0.1
            
        # Calmar ratio
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = metrics['mean_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0.0
            
        return metrics
    
    def _safe_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness safely"""
        try:
            from scipy import stats
            return stats.skew(data)
        except:
            return 0.0
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis safely"""
        try:
            from scipy import stats
            return stats.kurtosis(data)
        except:
            return 0.0
    
    def _default_risk_metrics(self) -> Dict[str, float]:
        """Return default risk metrics when calculation fails"""
        return {
            'var_95': 0.0,
            'var_99': 0.0,
            'var_999': 0.0,
            'es_95': 0.0,
            'es_99': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'mean_return': 0.0,
            'volatility': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    def enable_real_time_processing(self):
        """Enable real-time data processing"""
        if not self.stream_processor:
            self.stream_processor = StreamProcessor(self.config)
            logger.info("Real-time processing enabled")
    
    def enable_dashboard(self, port: int = 8050):
        """Enable web dashboard"""
        if not self.dashboard:
            self.dashboard = DashboardGenerator(self.config)
            logger.info(f"Dashboard enabled on port {port}")
    
    async def run_real_time_simulation(self, symbols: List[str], duration_hours: int = 24):
        """Run real-time simulation with live data"""
        if not self.stream_processor:
            self.enable_real_time_processing()
            
        logger.info(f"Starting real-time simulation for {symbols}")
        
        # This would integrate with the real-time engine
        # Implementation would depend on specific real-time requirements
        pass
    
    def save_models(self, directory: str):
        """Save trained models to directory"""
        save_dir = Path(directory)
        save_dir.mkdir(exist_ok=True)
        
        for symbol, markov_chain in self.markov_chains.items():
            filepath = save_dir / f"{symbol}_markov_chain.pkl"
            markov_chain.save_model(str(filepath))
            
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load trained models from directory"""
        load_dir = Path(directory)
        
        for model_file in load_dir.glob("*_markov_chain.pkl"):
            symbol = model_file.stem.replace("_markov_chain", "")
            self.markov_chains[symbol] = AdaptiveMarkovChain.load_model(str(model_file))
            
        logger.info(f"Models loaded from {directory}")

def main():
    """Main entry point for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced MC-Markov Finance Simulation")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "GOOGL", "MSFT"],
                       help="Financial symbols to simulate")
    parser.add_argument("--start-date", default="2020-01-01", 
                       help="Start date for historical data")
    parser.add_argument("--end-date", default="2023-12-31",
                       help="End date for historical data")
    parser.add_argument("--paths", type=int, default=10000,
                       help="Number of Monte Carlo paths")
    parser.add_argument("--steps", type=int, default=252,
                       help="Number of simulation steps")
    parser.add_argument("--config", help="Custom configuration file")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def run_simulation():
        # Initialize system
        system = AdvancedMCMarkovSystem(args.config)
        
        # Run simulation
        results = await system.run_simulation(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            simulation_paths=args.paths,
            time_horizon=args.steps
        )
        
        # Print results
        print("\n=== Simulation Results ===")
        print(f"VaR (95%): {results.var_95:.4f}")
        print(f"VaR (99%): {results.var_99:.4f}")
        print(f"Expected Shortfall (95%): {results.expected_shortfall_95:.4f}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.4f}")
        print(f"Maximum Drawdown: {results.max_drawdown:.4f}")
        print(f"Computation Time: {results.computation_time:.2f}s")
        
        # Save results if requested
        if args.output:
            results.save(args.output)
            print(f"Results saved to {args.output}")
    
    # Run the simulation
    asyncio.run(run_simulation())

if __name__ == "__main__":
    main()
