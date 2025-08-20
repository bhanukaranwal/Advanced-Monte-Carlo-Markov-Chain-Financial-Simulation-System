"""
Performance profiling and optimization recommendations
"""
import time
import cProfile
import pstats
import io
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from functools import wraps
import psutil
import logging

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_percent: float
    function_calls: int
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class BenchmarkResult:
    """Benchmark test result"""
    test_name: str
    metrics: PerformanceMetrics
    parameters: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    category: str  # 'memory', 'cpu', 'algorithm', 'parallelization'
    priority: str  # 'high', 'medium', 'low'
    description: str
    potential_improvement: str
    implementation_effort: str  # 'low', 'medium', 'high'

class PerformanceProfiler:
    """Comprehensive performance profiling toolkit"""
    
    def __init__(self, enable_memory_tracking: bool = True):
        self.enable_memory_tracking = enable_memory_tracking
        self.profiling_results = {}
        self.benchmark_history = []
        
        if enable_memory_tracking:
            tracemalloc.start()
            
    def profile_function(
        self,
        func: Callable,
        *args,
        profile_memory: bool = True,
        profile_cpu: bool = True,
        **kwargs
    ) -> Tuple[Any, PerformanceMetrics]:
        """
        Profile a function execution
        
        Args:
            func: Function to profile
            *args: Function arguments
            profile_memory: Whether to profile memory usage
            profile_cpu: Whether to profile CPU usage
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (function_result, performance_metrics)
        """
        # Initialize tracking
        process = psutil.Process()
        
        if profile_memory and self.enable_memory_tracking:
            tracemalloc.clear_traces()
            memory_start = process.memory_info().rss / (1024 * 1024)  # MB
        else:
            memory_start = 0
            
        if profile_cpu:
            cpu_start = process.cpu_percent()
            
        # Start profiling
        profiler = cProfile.Profile()
        start_time = time.time()
        
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            execution_time = time.time() - start_time
            
            # Get memory usage
            if profile_memory and self.enable_memory_tracking:
                current, peak = tracemalloc.get_traced_memory()
                memory_end = process.memory_info().rss / (1024 * 1024)  # MB
                memory_usage = memory_end - memory_start
                peak_memory = peak / (1024 * 1024)  # MB
            else:
                memory_usage = 0
                peak_memory = 0
                
            # Get CPU usage
            if profile_cpu:
                cpu_usage = process.cpu_percent() - cpu_start
            else:
                cpu_usage = 0
                
            # Get function call statistics
            stats = pstats.Stats(profiler)
            function_calls = stats.total_calls
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                cpu_percent=cpu_usage,
                function_calls=function_calls
            )
            
            return result, metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_mb=0,
                peak_memory_mb=0,
                cpu_percent=0,
                function_calls=0
            )
            
            logger.error(f"Error during profiling: {e}")
            raise
            
    def profile_with_decorator(self, name: str = None):
        """Decorator for automatic function profiling"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = name or func.__name__
                result, metrics = self.profile_function(func, *args, **kwargs)
                
                self.profiling_results[func_name] = metrics
                logger.info(f"Profiled {func_name}: {metrics.execution_time:.4f}s, "
                           f"{metrics.memory_usage_mb:.2f}MB")
                
                return result
            return wrapper
        return decorator
        
    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        test_args: Tuple,
        test_kwargs: Dict = None,
        n_runs: int = 5
    ) -> pd.DataFrame:
        """
        Compare performance of different implementations
        
        Args:
            implementations: Dictionary of {name: function} implementations
            test_args: Arguments to pass to each function
            test_kwargs: Keyword arguments to pass to each function
            n_runs: Number of runs for averaging
            
        Returns:
            DataFrame with comparison results
        """
        if test_kwargs is None:
            test_kwargs = {}
            
        results = []
        
        for name, func in implementations.items():
            logger.info(f"Benchmarking {name}...")
            
            run_times = []
            memory_usages = []
            
            for run in range(n_runs):
                try:
                    _, metrics = self.profile_function(func, *test_args, **test_kwargs)
                    run_times.append(metrics.execution_time)
                    memory_usages.append(metrics.memory_usage_mb)
                    
                except Exception as e:
                    logger.error(f"Error in {name}, run {run}: {e}")
                    run_times.append(np.nan)
                    memory_usages.append(np.nan)
                    
            results.append({
                'implementation': name,
                'mean_time': np.nanmean(run_times),
                'std_time': np.nanstd(run_times),
                'min_time': np.nanmin(run_times),
                'max_time': np.nanmax(run_times),
                'mean_memory': np.nanmean(memory_usages),
                'std_memory': np.nanstd(memory_usages)
            })
            
        return pd.DataFrame(results)
        
    def memory_profile_line_by_line(self, func: Callable, *args, **kwargs):
        """Line-by-line memory profiling (requires memory_profiler)"""
        try:
            from memory_profiler import LineProfiler
            
            profiler = LineProfiler()
            profiler.add_function(func)
            
            profiler.enable_by_count()
            result = func(*args, **kwargs)
            profiler.disable_by_count()
            
            # Get profile results
            stream = io.StringIO()
            profiler.print_stats(stream=stream)
            profile_output = stream.getvalue()
            
            return result, profile_output
            
        except ImportError:
            logger.warning("memory_profiler not available for line-by-line profiling")
            return self.profile_function(func, *args, **kwargs)
            
    def analyze_bottlenecks(self, profile_stats: pstats.Stats) -> Dict[str, Any]:
        """Analyze profiling results to identify bottlenecks"""
        # Get top time-consuming functions
        stream = io.StringIO()
        profile_stats.print_stats(20, stream=stream)
        stats_output = stream.getvalue()
        
        # Parse statistics
        bottlenecks = {
            'top_functions_by_time': [],
            'top_functions_by_calls': [],
            'recommendations': []
        }
        
        # Sort by cumulative time
        stats_by_time = profile_stats.get_stats_profile()
        time_sorted = sorted(
            stats_by_time.func_profiles.items(),
            key=lambda x: x[1].cumtime,
            reverse=True
        )
        
        for i, (func, stats) in enumerate(time_sorted[:10]):
            bottlenecks['top_functions_by_time'].append({
                'function': f"{func[0]}:{func[1]}({func})",
                'cumulative_time': stats.cumtime,
                'self_time': stats.tottime,
                'calls': stats.ncalls
            })
            
        # Sort by call count
        call_sorted = sorted(
            stats_by_time.func_profiles.items(),
            key=lambda x: x[1].ncalls,
            reverse=True
        )
        
        for i, (func, stats) in enumerate(call_sorted[:10]):
            bottlenecks['top_functions_by_calls'].append({
                'function': f"{func[0]}:{func[1]}({func})",
                'calls': stats.ncalls,
                'time_per_call': stats.tottime / stats.ncalls if stats.ncalls > 0 else 0
            })
            
        return bottlenecks
        
    def generate_optimization_recommendations(
        self,
        metrics: PerformanceMetrics,
        bottlenecks: Dict[str, Any] = None
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on profiling results"""
        recommendations = []
        
        # Memory optimization recommendations
        if metrics.memory_usage_mb > 1000:  # > 1GB
            recommendations.append(OptimizationRecommendation(
                category='memory',
                priority='high',
                description='High memory usage detected',
                potential_improvement='Reduce memory consumption by 50-80%',
                implementation_effort='medium'
            ))
            
        if metrics.peak_memory_mb > metrics.memory_usage_mb * 2:
            recommendations.append(OptimizationRecommendation(
                category='memory',
                priority='medium',
                description='Memory usage spikes detected',
                potential_improvement='Implement streaming or chunked processing',
                implementation_effort='medium'
            ))
            
        # CPU optimization recommendations
        if metrics.execution_time > 60:  # > 1 minute
            recommendations.append(OptimizationRecommendation(
                category='cpu',
                priority='high',
                description='Long execution time detected',
                potential_improvement='Consider parallelization or algorithm optimization',
                implementation_effort='high'
            ))
            
        if metrics.function_calls > 1000000:  # > 1M calls
            recommendations.append(OptimizationRecommendation(
                category='algorithm',
                priority='medium',
                description='High number of function calls',
                potential_improvement='Vectorize operations or use compiled code',
                implementation_effort='medium'
            ))
            
        # Parallelization recommendations
        if metrics.execution_time > 10 and metrics.cpu_percent < 80:
            recommendations.append(OptimizationRecommendation(
                category='parallelization',
                priority='medium',
                description='Underutilized CPU resources',
                potential_improvement='Implement parallel processing',
                implementation_effort='medium'
            ))
            
        return recommendations

class BenchmarkSuite:
    """Comprehensive benchmark suite for financial computations"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.benchmark_results = []
        
    def benchmark_monte_carlo(
        self,
        n_paths_list: List[int] = [1000, 10000, 100000],
        n_steps_list: List[int] = [100, 252, 1000]
    ) -> List[BenchmarkResult]:
        """Benchmark Monte Carlo simulations"""
        results = []
        
        def gbm_simulation(n_paths: int, n_steps: int):
            """Simple GBM simulation for benchmarking"""
            dt = 1.0 / n_steps
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            
            S = np.zeros((n_paths, n_steps + 1))
            S[:, 0] = 100.0
            
            for t in range(n_steps):
                S[:, t + 1] = S[:, t] * np.exp(
                    (0.05 - 0.5 * 0.2**2) * dt + 0.2 * dW[:, t]
                )
                
            return S
            
        for n_paths in n_paths_list:
            for n_steps in n_steps_list:
                test_name = f"GBM_{n_paths}paths_{n_steps}steps"
                
                try:
                    _, metrics = self.profiler.profile_function(
                        gbm_simulation, n_paths, n_steps
                    )
                    
                    result = BenchmarkResult(
                        test_name=test_name,
                        metrics=metrics,
                        parameters={'n_paths': n_paths, 'n_steps': n_steps},
                        success=True
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=test_name,
                        metrics=PerformanceMetrics(0, 0, 0, 0, 0),
                        parameters={'n_paths': n_paths, 'n_steps': n_steps},
                        success=False,
                        error_message=str(e)
                    )
                    
                results.append(result)
                logger.info(f"Completed benchmark: {test_name}")
                
        self.benchmark_results.extend(results)
        return results
        
    def benchmark_matrix_operations(
        self,
        matrix_sizes: List[int] = [100, 500, 1000, 2000]
    ) -> List[BenchmarkResult]:
        """Benchmark matrix operations"""
        results = []
        
        operations = {
            'matrix_multiply': lambda A, B: np.dot(A, B),
            'matrix_inverse': lambda A, B: np.linalg.inv(A),
            'eigendecomposition': lambda A, B: np.linalg.eigh(A),
            'cholesky': lambda A, B: np.linalg.cholesky(A + np.eye(A.shape[0]) * 1e-6)
        }
        
        for size in matrix_sizes:
            # Generate test matrices
            np.random.seed(42)  # For reproducibility
            A = np.random.randn(size, size)
            A = A @ A.T  # Make positive definite
            B = np.random.randn(size, size)
            
            for op_name, op_func in operations.items():
                test_name = f"{op_name}_{size}x{size}"
                
                try:
                    _, metrics = self.profiler.profile_function(op_func, A, B)
                    
                    result = BenchmarkResult(
                        test_name=test_name,
                        metrics=metrics,
                        parameters={'matrix_size': size, 'operation': op_name},
                        success=True
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=test_name,
                        metrics=PerformanceMetrics(0, 0, 0, 0, 0),
                        parameters={'matrix_size': size, 'operation': op_name},
                        success=False,
                        error_message=str(e)
                    )
                    
                results.append(result)
                
        self.benchmark_results.extend(results)
        return results
        
    def benchmark_data_processing(
        self,
        data_sizes: List[int] = [10000, 100000, 1000000]
    ) -> List[BenchmarkResult]:
        """Benchmark data processing operations"""
        results = []
        
        def rolling_statistics(data, window=20):
            """Calculate rolling statistics"""
            df = pd.DataFrame({'values': data})
            return df.rolling(window=window).agg(['mean', 'std', 'min', 'max'])
            
        def correlation_matrix(data):
            """Calculate correlation matrix"""
            return np.corrcoef(data.T)
            
        operations = {
            'rolling_stats': rolling_statistics,
            'correlation_matrix': correlation_matrix
        }
        
        for size in data_sizes:
            # Generate test data
            np.random.seed(42)
            if size <= 100000:
                n_features = 10
            else:
                n_features = 5  # Reduce features for large datasets
                
            data = np.random.randn(size, n_features)
            
            for op_name, op_func in operations.items():
                test_name = f"{op_name}_{size}samples"
                
                try:
                    if op_name == 'rolling_stats':
                        _, metrics = self.profiler.profile_function(op_func, data[:, 0])
                    else:
                        _, metrics = self.profiler.profile_function(op_func, data)
                        
                    result = BenchmarkResult(
                        test_name=test_name,
                        metrics=metrics,
                        parameters={'data_size': size, 'operation': op_name},
                        success=True
                    )
                    
                except Exception as e:
                    result = BenchmarkResult(
                        test_name=test_name,
                        metrics=PerformanceMetrics(0, 0, 0, 0, 0),
                        parameters={'data_size': size, 'operation': op_name},
                        success=False,
                        error_message=str(e)
                    )
                    
                results.append(result)
                
        self.benchmark_results.extend(results)
        return results
        
    def generate_benchmark_report(self) -> pd.DataFrame:
        """Generate comprehensive benchmark report"""
        if not self.benchmark_results:
            return pd.DataFrame()
            
        report_data = []
        
        for result in self.benchmark_results:
            row = {
                'test_name': result.test_name,
                'success': result.success,
                'execution_time': result.metrics.execution_time,
                'memory_usage_mb': result.metrics.memory_usage_mb,
                'peak_memory_mb': result.metrics.peak_memory_mb,
                'cpu_percent': result.metrics.cpu_percent,
                'function_calls': result.metrics.function_calls
            }
            
            # Add parameter information
            for param, value in result.parameters.items():
                row[param] = value
                
            if not result.success:
                row['error'] = result.error_message
                
            report_data.append(row)
            
        return pd.DataFrame(report_data)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Performance Profiler...")
    
    # Initialize profiler
    profiler = PerformanceProfiler(enable_memory_tracking=True)
    
    # Test function profiling
    print("Testing function profiling:")
    
    def test_function(n: int):
        """Test function for profiling"""
        # Create large array
        data = np.random.randn(n, 100)
        
        # Perform some operations
        result = np.sum(data**2, axis=1)
        result = np.sort(result)
        
        return result
        
    # Profile the function
    result, metrics = profiler.profile_function(test_function, 10000)
    
    print(f"Execution time: {metrics.execution_time:.4f} seconds")
    print(f"Memory usage: {metrics.memory_usage_mb:.2f} MB")
    print(f"Peak memory: {metrics.peak_memory_mb:.2f} MB")
    print(f"Function calls: {metrics.function_calls}")
    
    # Test decorator profiling
    print("\nTesting decorator profiling:")
    
    @profiler.profile_with_decorator('matrix_operations')
    def matrix_operations(size: int):
        A = np.random.randn(size, size)
        B = np.random.randn(size, size)
        C = np.dot(A, B)
        eigenvals = np.linalg.eigvals(C)
        return eigenvals
        
    result = matrix_operations(500)
    print(f"Matrix operations completed, result shape: {result.shape}")
    
    # Test implementation comparison
    print("\nTesting implementation comparison:")
    
    def method_1(data):
        """Numpy-based method"""
        return np.mean(data, axis=0)
        
    def method_2(data):
        """Pure Python method"""
        return [sum(col) / len(col) for col in data.T]
        
    implementations = {
        'numpy_method': method_1,
        'python_method': method_2
    }
    
    test_data = np.random.randn(1000, 10)
    comparison = profiler.compare_implementations(
        implementations, (test_data,), n_runs=3
    )
    
    print("Implementation comparison:")
    print(comparison[['implementation', 'mean_time', 'mean_memory']].round(6))
    
    # Test benchmark suite
    print("\nTesting benchmark suite:")
    benchmark_suite = BenchmarkSuite(profiler)
    
    # Run smaller benchmarks for testing
    mc_results = benchmark_suite.benchmark_monte_carlo(
        n_paths_list=[1000, 5000],
        n_steps_list=[100, 252]
    )
    
    matrix_results = benchmark_suite.benchmark_matrix_operations(
        matrix_sizes=[100, 300]
    )
    
    # Generate report
    report = benchmark_suite.generate_benchmark_report()
    print(f"\nBenchmark report generated with {len(report)} tests")
    print("\nTop 5 slowest operations:")
    slowest = report.nlargest(5, 'execution_time')[['test_name', 'execution_time', 'memory_usage_mb']]
    print(slowest.round(4))
    
    # Generate optimization recommendations
    print("\nOptimization recommendations:")
    for result in mc_results[:2]:  # Show recommendations for first 2 tests
        if result.success:
            recommendations = profiler.generate_optimization_recommendations(result.metrics)
            if recommendations:
                print(f"\n{result.test_name}:")
                for rec in recommendations:
                    print(f"  {rec.category.upper()}: {rec.description}")
                    print(f"    Potential improvement: {rec.potential_improvement}")
            else:
                print(f"\n{result.test_name}: No specific recommendations")
    
    print("\nPerformance profiler test completed!")
