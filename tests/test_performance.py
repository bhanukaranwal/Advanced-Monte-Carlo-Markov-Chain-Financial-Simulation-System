"""
Performance benchmarks and tests
"""

import pytest
import numpy as np
import time
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from analytics_engine.risk_analytics import RiskAnalytics
from optimization.gpu_acceleration import GPUAccelerator

class TestMonteCarloPerformance:
    """Test Monte Carlo engine performance"""
    
    @pytest.mark.benchmark
    def test_gbm_performance_small(self, benchmark):
        """Benchmark small GBM simulation"""
        engine = GeometricBrownianMotionEngine(
            n_simulations=1000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        
        result = benchmark(engine.simulate_paths)
        assert result.shape == (1000, 253)
        
    @pytest.mark.benchmark
    def test_gbm_performance_large(self, benchmark):
        """Benchmark large GBM simulation"""
        engine = GeometricBrownianMotionEngine(
            n_simulations=10000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        
        result = benchmark(engine.simulate_paths)
        assert result.shape == (10000, 253)
        
    @pytest.mark.benchmark
    @pytest.mark.skipif(not GPUAccelerator().is_gpu_available(), reason="GPU not available")
    def test_gpu_acceleration_performance(self, benchmark):
        """Benchmark GPU accelerated simulation"""
        # This would test GPU performance if available
        pass

class TestRiskAnalyticsPerformance:
    """Test risk analytics performance"""
    
    @pytest.mark.benchmark
    def test_var_calculation_performance(self, benchmark):
        """Benchmark VaR calculation"""
        returns = np.random.normal(0, 0.02, 10000)
        risk_analytics = RiskAnalytics()
        
        result = benchmark(risk_analytics.calculate_var, returns, method='historical')
        assert 'var_95' in result
        
    @pytest.mark.benchmark
    def test_large_dataset_risk_analysis(self, benchmark):
        """Benchmark risk analysis on large dataset"""
        returns = np.random.normal(0, 0.02, 100000)
        risk_analytics = RiskAnalytics()
        
        result = benchmark(risk_analytics.calculate_comprehensive_risk_measures, returns)
        assert hasattr(result, 'var_95')

class TestMemoryUsage:
    """Test memory usage patterns"""
    
    def test_memory_efficiency_large_simulation(self):
        """Test memory efficiency for large simulations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run large simulation
        engine = GeometricBrownianMotionEngine(
            n_simulations=50000,
            n_steps=1000,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        
        paths = engine.simulate_paths()
        peak_memory = process.memory_info().rss
        
        # Clean up
        del paths, engine
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss
        
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        memory_cleanup = (peak_memory - final_memory) / 1024 / 1024    # MB
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Memory cleaned up: {memory_cleanup:.1f} MB")
        
        # Assert reasonable memory usage
        assert memory_increase < 2000  # Less than 2GB
        assert memory_cleanup > memory_increase * 0.8  # At least 80% cleanup

class TestConcurrencyPerformance:
    """Test concurrent execution performance"""
    
    @pytest.mark.benchmark
    def test_parallel_simulations(self, benchmark):
        """Benchmark parallel simulation execution"""
        def run_parallel_simulations():
            from concurrent.futures import ThreadPoolExecutor
            import numpy as np
            
            def run_simulation(params):
                engine = GeometricBrownianMotionEngine(**params)
                return engine.simulate_paths()
            
            # Define multiple simulation configurations
            configs = [
                {
                    'n_simulations': 2000,
                    'n_steps': 100,
                    'initial_price': 100.0,
                    'drift': 0.05,
                    'volatility': 0.2
                }
            ] * 5
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(run_simulation, configs))
                
            return results
        
        results = benchmark(run_parallel_simulations)
        assert len(results) == 5
        assert all(result.shape == (2000, 101) for result in results)
