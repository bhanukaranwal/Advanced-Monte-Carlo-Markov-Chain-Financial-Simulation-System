"""
Distributed computing framework using Dask and Ray for large-scale simulations
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import dask
    import dask.array as da
    from dask.distributed import Client, LocalCluster, as_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DistributedConfig:
    """Configuration for distributed computing"""
    backend: str = 'dask'  # 'dask' or 'ray'
    n_workers: int = 4
    memory_per_worker: str = '4GB'
    scheduler_address: Optional[str] = None
    use_gpu: bool = False
    chunk_size: int = 10000

class BaseDistributedEngine(ABC):
    """Abstract base class for distributed engines"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.client = None
        
    @abstractmethod
    def initialize(self):
        """Initialize the distributed computing environment"""
        pass
        
    @abstractmethod
    def shutdown(self):
        """Shutdown the distributed computing environment"""
        pass
        
    @abstractmethod
    def submit_job(self, func: Callable, *args, **kwargs):
        """Submit a job for distributed execution"""
        pass

class DaskDistributedEngine(BaseDistributedEngine):
    """Dask-based distributed computing engine"""
    
    def initialize(self):
        """Initialize Dask cluster"""
        if not DASK_AVAILABLE:
            raise ImportError("Dask is required for distributed computing")
            
        logger.info("Initializing Dask cluster")
        
        if self.config.scheduler_address:
            # Connect to existing cluster
            self.client = Client(self.config.scheduler_address)
        else:
            # Create local cluster
            cluster = LocalCluster(
                n_workers=self.config.n_workers,
                memory_limit=self.config.memory_per_worker,
                processes=True
            )
            self.client = Client(cluster)
            
        logger.info(f"Dask cluster initialized: {self.client}")
        
    def shutdown(self):
        """Shutdown Dask cluster"""
        if self.client:
            self.client.close()
            logger.info("Dask cluster shutdown")
            
    def submit_job(self, func: Callable, *args, **kwargs):
        """Submit job to Dask cluster"""
        future = self.client.submit(func, *args, **kwargs)
        return future
        
    def scatter_data(self, data):
        """Scatter data across workers"""
        return self.client.scatter(data, broadcast=True)
        
    def gather_results(self, futures):
        """Gather results from futures"""
        return self.client.gather(futures)

@ray.remote
class RayMonteCarloWorker:
    """Ray worker for Monte Carlo simulations"""
    
    def __init__(self):
        self.initialized = True
        
    def simulate_chunk(self, n_simulations: int, n_steps: int, 
                      initial_price: float, drift: float, volatility: float,
                      random_seed: Optional[int] = None):
        """Simulate a chunk of Monte Carlo paths"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        dt = 1.0 / n_steps
        drift_term = (drift - 0.5 * volatility**2) * dt
        vol_term = volatility * np.sqrt(dt)
        
        # Generate random numbers
        randoms = np.random.normal(0, 1, (n_simulations, n_steps))
        
        # Calculate log returns
        log_returns = drift_term + vol_term * randoms
        
        # Calculate paths
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.full((n_simulations, 1), np.log(initial_price)),
            log_prices
        ])
        
        paths = np.exp(log_prices)
        
        return {
            'paths': paths,
            'final_prices': paths[:, -1],
            'n_simulations': n_simulations
        }

class RayDistributedEngine(BaseDistributedEngine):
    """Ray-based distributed computing engine"""
    
    def initialize(self):
        """Initialize Ray cluster"""
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed computing")
            
        logger.info("Initializing Ray cluster")
        
        if not ray.is_initialized():
            if self.config.scheduler_address:
                ray.init(address=self.config.scheduler_address)
            else:
                ray.init(
                    num_cpus=self.config.n_workers,
                    object_store_memory=int(self.config.memory_per_worker.replace('GB', '')) * 1024**3
                )
                
        logger.info(f"Ray cluster initialized: {ray.cluster_resources()}")
        
    def shutdown(self):
        """Shutdown Ray cluster"""
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shutdown")
            
    def submit_job(self, func: Callable, *args, **kwargs):
        """Submit job to Ray cluster"""
        remote_func = ray.remote(func)
        future = remote_func.remote(*args, **kwargs)
        return future

class DistributedMonteCarloEngine:
    """
    Distributed Monte Carlo engine supporting both Dask and Ray
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        
        if config.backend == 'dask':
            self.engine = DaskDistributedEngine(config)
        elif config.backend == 'ray':
            self.engine = RayDistributedEngine(config)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")
            
        self.engine.initialize()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        
    def shutdown(self):
        """Shutdown distributed engine"""
        self.engine.shutdown()
        
    def simulate_distributed_monte_carlo(
        self,
        total_simulations: int,
        n_steps: int,
        initial_price: float,
        drift: float,
        volatility: float,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run distributed Monte Carlo simulation
        """
        logger.info(f"Starting distributed Monte Carlo: {total_simulations} total simulations")
        start_time = time.time()
        
        # Calculate chunk distribution
        chunk_size = min(self.config.chunk_size, total_simulations // self.config.n_workers)
        n_chunks = (total_simulations + chunk_size - 1) // chunk_size
        
        if self.config.backend == 'dask':
            return self._simulate_with_dask(
                total_simulations, n_steps, initial_price, drift, volatility,
                chunk_size, n_chunks, random_seed, start_time
            )
        else:
            return self._simulate_with_ray(
                total_simulations, n_steps, initial_price, drift, volatility,
                chunk_size, n_chunks, random_seed, start_time
            )
            
    def _simulate_with_dask(self, total_simulations, n_steps, initial_price,
                           drift, volatility, chunk_size, n_chunks, random_seed, start_time):
        """Simulate using Dask"""
        
        @delayed
        def simulate_chunk(chunk_id, n_sims, seed_offset=0):
            seed = random_seed + seed_offset if random_seed else None
            return self._monte_carlo_chunk(
                n_sims, n_steps, initial_price, drift, volatility, seed
            )
            
        # Create delayed computations
        tasks = []
        remaining_sims = total_simulations
        
        for i in range(n_chunks):
            n_sims_chunk = min(chunk_size, remaining_sims)
            if n_sims_chunk > 0:
                task = simulate_chunk(i, n_sims_chunk, i * 1000)
                tasks.append(task)
                remaining_sims -= n_sims_chunk
                
        # Execute computations
        logger.info(f"Executing {len(tasks)} chunks with Dask")
        results = dask.compute(*tasks)
        
        # Combine results
        return self._combine_results(results, start_time)
        
    def _simulate_with_ray(self, total_simulations, n_steps, initial_price,
                          drift, volatility, chunk_size, n_chunks, random_seed, start_time):
        """Simulate using Ray"""
        
        # Create worker pool
        workers = [RayMonteCarloWorker.remote() for _ in range(self.config.n_workers)]
        
        # Submit tasks
        futures = []
        remaining_sims = total_simulations
        worker_idx = 0
        
        for i in range(n_chunks):
            n_sims_chunk = min(chunk_size, remaining_sims)
            if n_sims_chunk > 0:
                seed = random_seed + i * 1000 if random_seed else None
                worker = workers[worker_idx % len(workers)]
                
                future = worker.simulate_chunk.remote(
                    n_sims_chunk, n_steps, initial_price, drift, volatility, seed
                )
                futures.append(future)
                
                remaining_sims -= n_sims_chunk
                worker_idx += 1
                
        # Gather results
        logger.info(f"Executing {len(futures)} chunks with Ray")
        results = ray.get(futures)
        
        return self._combine_results(results, start_time)
        
    def _monte_carlo_chunk(self, n_simulations, n_steps, initial_price,
                          drift, volatility, random_seed=None):
        """Single chunk Monte Carlo simulation"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        dt = 1.0 / n_steps
        drift_term = (drift - 0.5 * volatility**2) * dt
        vol_term = volatility * np.sqrt(dt)
        
        randoms = np.random.normal(0, 1, (n_simulations, n_steps))
        log_returns = drift_term + vol_term * randoms
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.full((n_simulations, 1), np.log(initial_price)),
            log_prices
        ])
        
        paths = np.exp(log_prices)
        
        return {
            'paths': paths,
            'final_prices': paths[:, -1],
            'n_simulations': n_simulations
        }
        
    def _combine_results(self, results, start_time):
        """Combine results from all chunks"""
        all_paths = []
        all_final_prices = []
        total_simulations = 0
        
        for result in results:
            if result['paths'] is not None:
                all_paths.append(result['paths'])
                all_final_prices.extend(result['final_prices'])
                total_simulations += result['n_simulations']
                
        if all_paths:
            combined_paths = np.vstack(all_paths)
            final_prices = np.array(all_final_prices)
            
            execution_time = time.time() - start_time
            
            statistics = {
                'mean_final_price': float(np.mean(final_prices)),
                'std_final_price': float(np.std(final_prices)),
                'min_final_price': float(np.min(final_prices)),
                'max_final_price': float(np.max(final_prices)),
                'total_simulations': total_simulations,
                'execution_time': execution_time,
                'throughput_sims_per_sec': total_simulations / execution_time
            }
            
            logger.info(f"Distributed simulation completed: {total_simulations} sims in {execution_time:.2f}s")
            
            return {
                'paths': combined_paths,
                'final_prices': final_prices,
                'statistics': statistics
            }
        else:
            raise RuntimeError("No valid results from distributed simulation")

class DistributedRiskAnalytics:
    """Distributed risk analytics engine"""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.engine = DistributedMonteCarloEngine(config)
        
    def calculate_distributed_var(
        self,
        portfolio_data: Dict[str, np.ndarray],
        confidence_levels: List[float] = [0.95, 0.99],
        n_simulations: int = 100000
    ) -> Dict[str, Any]:
        """Calculate VaR using distributed computing"""
        
        logger.info("Starting distributed VaR calculation")
        
        if self.config.backend == 'dask':
            return self._calculate_var_dask(portfolio_data, confidence_levels, n_simulations)
        else:
            return self._calculate_var_ray(portfolio_data, confidence_levels, n_simulations)
            
    def _calculate_var_dask(self, portfolio_data, confidence_levels, n_simulations):
        """Calculate VaR using Dask"""
        
        @delayed
        def calculate_var_chunk(returns_chunk, confidence_level):
            return np.percentile(returns_chunk, (1 - confidence_level) * 100)
            
        results = {}
        
        for asset, returns in portfolio_data.items():
            # Convert to Dask array
            da_returns = da.from_array(returns, chunks=self.config.chunk_size)
            
            asset_vars = {}
            for cl in confidence_levels:
                var_task = calculate_var_chunk(da_returns, cl)
                asset_vars[f'var_{int(cl*100)}'] = var_task
                
            results[asset] = asset_vars
            
        # Compute all VaR values
        computed_results = dask.compute(results)[0]
        
        return computed_results
        
    def _calculate_var_ray(self, portfolio_data, confidence_levels, n_simulations):
        """Calculate VaR using Ray"""
        
        @ray.remote
        def calculate_var_chunk(returns, confidence_level):
            return np.percentile(returns, (1 - confidence_level) * 100)
            
        results = {}
        
        for asset, returns in portfolio_data.items():
            asset_vars = {}
            
            for cl in confidence_levels:
                future = calculate_var_chunk.remote(returns, cl)
                asset_vars[f'var_{int(cl*100)}'] = ray.get(future)
                
            results[asset] = asset_vars
            
        return results
