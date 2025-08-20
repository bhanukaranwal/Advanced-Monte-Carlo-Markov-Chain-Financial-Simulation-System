"""
Memory optimization for large-scale financial computations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Iterator
import gc
import psutil
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    memory_percent: float
    process_memory_gb: float

class MemoryManager:
    """Advanced memory management for financial computations"""
    
    def __init__(
        self,
        max_memory_usage: float = 0.8,  # Maximum memory usage as fraction
        chunk_size_mb: int = 100,       # Default chunk size in MB
        enable_monitoring: bool = True
    ):
        self.max_memory_usage = max_memory_usage
        self.chunk_size_mb = chunk_size_mb
        self.enable_monitoring = enable_monitoring
        
        # Memory tracking
        self.allocated_arrays = []
        self.memory_history = []
        
        # Get system memory info
        self.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        
        return MemoryStats(
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=memory.used / (1024**3),
            memory_percent=memory.percent,
            process_memory_gb=process.memory_info().rss / (1024**3)
        )
        
    def check_memory_availability(self, required_gb: float) -> bool:
        """Check if enough memory is available for allocation"""
        stats = self.get_memory_stats()
        return stats.available_memory_gb >= required_gb
        
    def estimate_array_memory(self, shape: Tuple[int, ...], dtype=np.float64) -> float:
        """Estimate memory requirements for array in GB"""
        dtype_size = np.dtype(dtype).itemsize
        n_elements = np.prod(shape)
        return (n_elements * dtype_size) / (1024**3)
        
    def allocate_optimal_array(
        self, 
        shape: Tuple[int, ...], 
        dtype=np.float64,
        fill_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Allocate array with memory optimization
        
        Args:
            shape: Array shape
            dtype: Data type
            fill_value: Optional fill value
            
        Returns:
            Allocated array
        """
        required_memory = self.estimate_array_memory(shape, dtype)
        
        if not self.check_memory_availability(required_memory * 1.2):  # 20% buffer
            logger.warning(f"Insufficient memory for array allocation: {required_memory:.2f} GB required")
            # Try to use smaller dtype if possible
            if dtype == np.float64:
                logger.info("Falling back to float32 to save memory")
                dtype = np.float32
                required_memory = self.estimate_array_memory(shape, dtype)
                
        try:
            if fill_value is None:
                array = np.empty(shape, dtype=dtype)
            else:
                array = np.full(shape, fill_value, dtype=dtype)
                
            self.allocated_arrays.append(array)
            
            if self.enable_monitoring:
                stats = self.get_memory_stats()
                self.memory_history.append(stats)
                
            return array
            
        except MemoryError as e:
            logger.error(f"Memory allocation failed: {e}")
            # Force garbage collection and try again
            self.cleanup_memory()
            
            try:
                if fill_value is None:
                    array = np.empty(shape, dtype=dtype)
                else:
                    array = np.full(shape, fill_value, dtype=dtype)
                return array
            except MemoryError:
                raise MemoryError(f"Cannot allocate {required_memory:.2f} GB of memory")
                
    def cleanup_memory(self):
        """Force memory cleanup"""
        # Clear tracked arrays
        self.allocated_arrays.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Memory cleanup completed")
        
    def get_optimal_chunk_size(self, total_size: int, dtype=np.float64) -> int:
        """Calculate optimal chunk size based on available memory"""
        stats = self.get_memory_stats()
        available_memory_bytes = stats.available_memory_gb * (1024**3) * self.max_memory_usage
        
        dtype_size = np.dtype(dtype).itemsize
        max_chunk_elements = int(available_memory_bytes / dtype_size)
        
        # Use smaller of calculated size or default chunk size
        default_chunk_elements = (self.chunk_size_mb * 1024 * 1024) // dtype_size
        
        return min(max_chunk_elements, default_chunk_elements, total_size)

class ChunkedProcessor:
    """Process large datasets in memory-efficient chunks"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
    def process_chunks(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        processing_func: Callable,
        chunk_size: Optional[int] = None,
        combine_func: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Process data in chunks to manage memory usage
        
        Args:
            data: Input data array or DataFrame
            processing_func: Function to apply to each chunk
            chunk_size: Size of each chunk (auto-determined if None)
            combine_func: Function to combine chunk results
            **kwargs: Additional arguments for processing_func
            
        Returns:
            Combined processing result
        """
        if isinstance(data, pd.DataFrame):
            return self._process_dataframe_chunks(
                data, processing_func, chunk_size, combine_func, **kwargs
            )
        else:
            return self._process_array_chunks(
                data, processing_func, chunk_size, combine_func, **kwargs
            )
            
    def _process_array_chunks(
        self,
        data: np.ndarray,
        processing_func: Callable,
        chunk_size: Optional[int],
        combine_func: Optional[Callable],
        **kwargs
    ) -> Any:
        """Process numpy array in chunks"""
        if chunk_size is None:
            chunk_size = self.memory_manager.get_optimal_chunk_size(
                len(data), data.dtype
            )
            
        logger.info(f"Processing {len(data)} samples in chunks of {chunk_size}")
        
        results = []
        
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk = data[i:end_idx]
            
            # Process chunk
            chunk_result = processing_func(chunk, **kwargs)
            results.append(chunk_result)
            
            # Clean up chunk from memory
            del chunk
            
            # Periodic memory cleanup
            if i % (chunk_size * 10) == 0:
                gc.collect()
                
        # Combine results
        if combine_func is not None:
            return combine_func(results)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results, (int, float)):
            return np.array(results)
        else:
            return results
            
    def _process_dataframe_chunks(
        self,
        data: pd.DataFrame,
        processing_func: Callable,
        chunk_size: Optional[int],
        combine_func: Optional[Callable],
        **kwargs
    ) -> Any:
        """Process DataFrame in chunks"""
        if chunk_size is None:
            # Estimate memory usage of DataFrame
            memory_usage = data.memory_usage(deep=True).sum() / (1024**3)  # GB
            total_rows = len(data)
            
            # Calculate chunk size to use ~100MB per chunk
            target_memory_gb = 0.1  # 100MB
            chunk_size = int(total_rows * target_memory_gb / memory_usage)
            chunk_size = max(chunk_size, 1000)  # Minimum chunk size
            
        logger.info(f"Processing {len(data)} rows in chunks of {chunk_size}")
        
        results = []
        
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk = data.iloc[i:end_idx].copy()
            
            # Process chunk
            chunk_result = processing_func(chunk, **kwargs)
            results.append(chunk_result)
            
            # Clean up
            del chunk
            
            if i % (chunk_size * 5) == 0:
                gc.collect()
                
        # Combine results
        if combine_func is not None:
            return combine_func(results)
        elif isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(results, np.ndarray):
            return np.concatenate(results)
        else:
            return results

class StreamingCalculator:
    """Streaming calculations for memory-efficient statistics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset calculator state"""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # For variance calculation
        self.m3 = 0.0  # For skewness calculation
        self.m4 = 0.0  # For kurtosis calculation
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
    def update(self, value: float):
        """Update statistics with new value"""
        self.count += 1
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Welford's online algorithm for variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
        # Higher moments for skewness and kurtosis
        if self.count > 1:
            n = self.count
            delta_n = delta / n
            delta_n2 = delta_n * delta_n
            term1 = delta * delta2 * (n - 1)
            
            self.m4 += (term1 * delta_n2 * (n**2 - 3*n + 3) + 
                       6 * delta_n2 * self.m2 - 4 * delta_n * self.m3)
            self.m3 += term1 * delta_n * (n - 2) - 3 * delta_n * self.m2
            
    def update_batch(self, values: Union[np.ndarray, List[float]]):
        """Update with batch of values"""
        if isinstance(values, list):
            values = np.array(values)
            
        for value in values:
            self.update(value)
            
    def get_statistics(self) -> Dict[str, float]:
        """Get calculated statistics"""
        if self.count == 0:
            return {}
            
        statistics = {
            'count': self.count,
            'mean': self.mean,
            'min': self.min_val,
            'max': self.max_val,
            'range': self.max_val - self.min_val
        }
        
        if self.count > 1:
            variance = self.m2 / (self.count - 1)
            statistics['variance'] = variance
            statistics['std'] = np.sqrt(variance)
            
            if self.count > 2 and variance > 0:
                # Skewness
                skewness = (self.m3 / self.count) / (variance ** 1.5)
                statistics['skewness'] = skewness
                
                if self.count > 3:
                    # Kurtosis (excess)
                    kurtosis = (self.m4 / self.count) / (variance ** 2) - 3
                    statistics['kurtosis'] = kurtosis
                    
        return statistics
        
    def percentile_estimate(self, percentile: float) -> float:
        """Estimate percentile (simplified approximation)"""
        # This is a very rough approximation using normal distribution
        # For accurate percentiles, would need more sophisticated streaming algorithm
        if self.count < 2:
            return self.mean
            
        variance = self.m2 / (self.count - 1)
        std = np.sqrt(variance)
        
        # Assume normal distribution for approximation
        from scipy import stats
        z_score = stats.norm.ppf(percentile / 100)
        return self.mean + z_score * std

class MemoryMappedArray:
    """Memory-mapped array for handling very large datasets"""
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype=np.float64,
        filename: Optional[str] = None,
        mode: str = 'w+'
    ):
        self.shape = shape
        self.dtype = dtype
        self.filename = filename
        self.mode = mode
        
        if filename is None:
            # Create temporary file
            import tempfile
            self.temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.filename = self.temp_file.name
        else:
            self.temp_file = None
            
        # Create memory-mapped array
        self.array = np.memmap(
            self.filename,
            dtype=dtype,
            mode=mode,
            shape=shape
        )
        
    def __getitem__(self, key):
        return self.array[key]
        
    def __setitem__(self, key, value):
        self.array[key] = value
        
    def __del__(self):
        """Clean up temporary file"""
        if hasattr(self, 'array'):
            del self.array
            
        if self.temp_file is not None:
            try:
                os.unlink(self.filename)
            except (OSError, FileNotFoundError):
                pass

class ParallelProcessor:
    """Parallel processing with memory optimization"""
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_processes: bool = False,
        memory_manager: Optional[MemoryManager] = None
    ):
        self.n_workers = n_workers or os.cpu_count()
        self.use_processes = use_processes
        self.memory_manager = memory_manager or MemoryManager()
        
    def parallel_apply(
        self,
        func: Callable,
        data_chunks: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Apply function to data chunks in parallel
        
        Args:
            func: Function to apply
            data_chunks: List of data chunks
            **kwargs: Additional arguments for func
            
        Returns:
            List of results from each chunk
        """
        if self.use_processes:
            # Use process pool for CPU-bound tasks
            executor_class = ProcessPoolExecutor
        else:
            # Use thread pool for I/O-bound tasks
            executor_class = ThreadPoolExecutor
            
        results = []
        
        with executor_class(max_workers=self.n_workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(func, chunk, **kwargs)
                for chunk in data_chunks
            ]
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
                    results.append(None)
                    
        return results
        
    def create_data_chunks(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        chunk_size: Optional[int] = None
    ) -> List[Union[np.ndarray, pd.DataFrame]]:
        """Create data chunks for parallel processing"""
        if chunk_size is None:
            # Divide data into chunks for each worker
            chunk_size = len(data) // self.n_workers
            chunk_size = max(chunk_size, 1)
            
        chunks = []
        
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            
            if isinstance(data, pd.DataFrame):
                chunk = data.iloc[i:end_idx].copy()
            else:
                chunk = data[i:end_idx].copy()
                
            chunks.append(chunk)
            
        return chunks

# Example usage and testing
if __name__ == "__main__":
    print("Testing Memory Optimization...")
    
    # Test Memory Manager
    print("Testing Memory Manager:")
    memory_manager = MemoryManager(max_memory_usage=0.5, chunk_size_mb=50)
    
    # Get initial memory stats
    initial_stats = memory_manager.get_memory_stats()
    print(f"Initial memory usage: {initial_stats.process_memory_gb:.2f} GB "
          f"({initial_stats.memory_percent:.1f}% system)")
    
    # Test array allocation
    print("\nTesting array allocation:")
    large_shape = (10000, 1000)
    required_memory = memory_manager.estimate_array_memory(large_shape, np.float64)
    print(f"Required memory for {large_shape} float64 array: {required_memory:.2f} GB")
    
    if memory_manager.check_memory_availability(required_memory):
        test_array = memory_manager.allocate_optimal_array(large_shape, np.float64)
        print(f"Successfully allocated array of shape {test_array.shape}")
        
        # Fill with test data
        test_array[:] = np.random.randn(*large_shape)
        
        final_stats = memory_manager.get_memory_stats()
        print(f"Memory usage after allocation: {final_stats.process_memory_gb:.2f} GB")
        
        # Clean up
        memory_manager.cleanup_memory()
        del test_array
        
        cleanup_stats = memory_manager.get_memory_stats()
        print(f"Memory usage after cleanup: {cleanup_stats.process_memory_gb:.2f} GB")
    else:
        print("Insufficient memory for large array allocation")
    
    # Test Chunked Processor
    print("\nTesting Chunked Processor:")
    chunked_processor = ChunkedProcessor(memory_manager)
    
    # Create test data
    test_data = np.random.randn(50000, 10)
    print(f"Test data shape: {test_data.shape}")
    
    # Define processing function
    def calculate_statistics(chunk):
        return {
            'mean': np.mean(chunk, axis=0),
            'std': np.std(chunk, axis=0),
            'count': len(chunk)
        }
    
    # Define combine function
    def combine_statistics(results):
        total_count = sum(r['count'] for r in results)
        
        # Weighted average for means
        combined_mean = np.sum([r['mean'] * r['count'] for r in results], axis=0) / total_count
        
        # Combined standard deviation (approximate)
        combined_var = np.sum([r['std']**2 * r['count'] for r in results], axis=0) / total_count
        combined_std = np.sqrt(combined_var)
        
        return {
            'mean': combined_mean,
            'std': combined_std,
            'total_count': total_count
        }
    
    # Process in chunks
    result = chunked_processor.process_chunks(
        test_data,
        calculate_statistics,
        chunk_size=10000,
        combine_func=combine_statistics
    )
    
    print(f"Chunked processing result:")
    print(f"  Total samples processed: {result['total_count']}")
    print(f"  Mean values: {result['mean'][:3]}...")  # Show first 3
    print(f"  Std values: {result['std'][:3]}...")    # Show first 3
    
    # Test Streaming Calculator
    print("\nTesting Streaming Calculator:")
    streaming_calc = StreamingCalculator()
    
    # Simulate streaming data
    np.random.seed(42)
    stream_data = np.random.normal(10, 2, 100000)
    
    # Process in batches
    batch_size = 1000
    for i in range(0, len(stream_data), batch_size):
        batch = stream_data[i:i+batch_size]
        streaming_calc.update_batch(batch)
        
        if i % 10000 == 0:
            stats = streaming_calc.get_statistics()
            print(f"  After {stats['count']} samples: "
                  f"mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    final_streaming_stats = streaming_calc.get_statistics()
    print(f"Final streaming statistics:")
    print(f"  Count: {final_streaming_stats['count']}")
    print(f"  Mean: {final_streaming_stats['mean']:.4f} (true: 10.0)")
    print(f"  Std: {final_streaming_stats['std']:.4f} (true: 2.0)")
    print(f"  Skewness: {final_streaming_stats.get('skewness', 'N/A')}")
    print(f"  Kurtosis: {final_streaming_stats.get('kurtosis', 'N/A')}")
    
    # Test Memory-Mapped Array
    print("\nTesting Memory-Mapped Array:")
    mmap_shape = (1000, 1000)
    mmap_array = MemoryMappedArray(mmap_shape, dtype=np.float32)
    
    # Fill with test data
    for i in range(0, mmap_shape[0], 100):
        end_i = min(i + 100, mmap_shape)
        mmap_array[i:end_i, :] = np.random.randn(end_i - i, mmap_shape[1])
    
    # Calculate statistics
    sample_mean = np.mean(mmap_array[:100, :100])  # Sample subset
    print(f"Memory-mapped array created and filled")
    print(f"Sample mean from subset: {sample_mean:.4f}")
    
    # Clean up
    del mmap_array
    
    print("\nMemory optimization test completed!")
