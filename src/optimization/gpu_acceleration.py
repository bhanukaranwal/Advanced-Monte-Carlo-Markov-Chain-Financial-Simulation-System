"""
GPU acceleration for Monte Carlo simulations and matrix operations
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import time
import logging
from dataclasses import dataclass

# Try to import GPU libraries
try:
    import cupy as cp
    import cupyx.scipy.stats as cp_stats
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_stats = None
    CUPY_AVAILABLE = False

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    cl = None
    cl_array = None
    OPENCL_AVAILABLE = False

try:
    import numba
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    numba = None
    cuda = None
    jit = None
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU device information"""
    device_name: str
    compute_capability: str
    memory_total: int
    memory_free: int
    multiprocessor_count: int
    max_threads_per_block: int

class GPUAccelerator:
    """Main GPU acceleration orchestrator"""
    
    def __init__(self, backend: str = 'auto', device_id: int = 0):
        """
        Initialize GPU accelerator
        
        Args:
            backend: GPU backend ('cupy', 'opencl', 'numba', 'auto')
            device_id: GPU device ID
        """
        self.backend = backend
        self.device_id = device_id
        self.device_info = None
        
        # Auto-detect best available backend
        if backend == 'auto':
            if CUPY_AVAILABLE:
                self.backend = 'cupy'
            elif NUMBA_AVAILABLE:
                self.backend = 'numba'
            elif OPENCL_AVAILABLE:
                self.backend = 'opencl'
            else:
                self.backend = 'cpu'
                logger.warning("No GPU backends available, falling back to CPU")
        
        # Initialize selected backend
        self._initialize_backend()
        
    def _initialize_backend(self):
        """Initialize the selected GPU backend"""
        if self.backend == 'cupy' and CUPY_AVAILABLE:
            try:
                cp.cuda.Device(self.device_id).use()
                self.device_info = self._get_cupy_device_info()
                logger.info(f"Initialized CuPy backend on device: {self.device_info.device_name}")
            except Exception as e:
                logger.error(f"Failed to initialize CuPy: {e}")
                self.backend = 'cpu'
                
        elif self.backend == 'numba' and NUMBA_AVAILABLE:
            try:
                # Select CUDA device
                if cuda.is_available():
                    cuda.select_device(self.device_id)
                    self.device_info = self._get_numba_device_info()
                    logger.info(f"Initialized Numba CUDA backend")
                else:
                    logger.warning("CUDA not available for Numba")
                    self.backend = 'cpu'
            except Exception as e:
                logger.error(f"Failed to initialize Numba CUDA: {e}")
                self.backend = 'cpu'
                
        elif self.backend == 'opencl' and OPENCL_AVAILABLE:
            try:
                self._initialize_opencl()
                logger.info("Initialized OpenCL backend")
            except Exception as e:
                logger.error(f"Failed to initialize OpenCL: {e}")
                self.backend = 'cpu'
                
    def _get_cupy_device_info(self) -> GPUInfo:
        """Get CuPy device information"""
        device = cp.cuda.Device()
        attrs = device.attributes
        
        return GPUInfo(
            device_name=device.name.decode(),
            compute_capability=f"{attrs['ComputeCapabilityMajor']}.{attrs['ComputeCapabilityMinor']}",
            memory_total=device.mem_info[1],
            memory_free=device.mem_info,
            multiprocessor_count=attrs['MultiProcessorCount'],
            max_threads_per_block=attrs['MaxThreadsPerBlock']
        )
        
    def _get_numba_device_info(self) -> GPUInfo:
        """Get Numba CUDA device information"""
        device = cuda.get_current_device()
        
        return GPUInfo(
            device_name=device.name.decode(),
            compute_capability=f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            memory_total=device.memory_info.total,
            memory_free=device.memory_info.free,
            multiprocessor_count=device.MULTIPROCESSOR_COUNT,
            max_threads_per_block=device.MAX_THREADS_PER_BLOCK
        )
        
    def _initialize_opencl(self):
        """Initialize OpenCL context"""
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
            
        # Use first GPU device
        devices = platforms[0].get_devices(cl.device_type.GPU)
        if not devices:
            raise RuntimeError("No GPU devices found")
            
        self.cl_context = cl.Context([devices[self.device_id]])
        self.cl_queue = cl.CommandQueue(self.cl_context)
        
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        return self.backend != 'cpu'
        
    def to_gpu(self, array: np.ndarray):
        """Transfer array to GPU memory"""
        if self.backend == 'cupy':
            return cp.asarray(array)
        elif self.backend == 'opencl':
            return cl_array.to_device(self.cl_queue, array)
        else:
            return array  # CPU fallback
            
    def to_cpu(self, gpu_array):
        """Transfer array from GPU to CPU memory"""
        if self.backend == 'cupy':
            return cp.asnumpy(gpu_array)
        elif self.backend == 'opencl':
            return gpu_array.get()
        else:
            return gpu_array  # Already on CPU
            
    def get_memory_info(self) -> Dict[str, int]:
        """Get GPU memory information"""
        if self.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'free_bytes': cp.cuda.runtime.memGetInfo()[0]
            }
        else:
            return {'used_bytes': 0, 'total_bytes': 0, 'free_bytes': 0}

class CUDAMonteCarloEngine:
    """CUDA-accelerated Monte Carlo simulation engine"""
    
    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        
        if not self.gpu.is_gpu_available():
            logger.warning("GPU not available, using CPU fallback")
            
    def generate_random_paths(
        self,
        n_paths: int,
        n_steps: int,
        n_assets: int = 1,
        random_type: str = 'normal'
    ) -> np.ndarray:
        """
        Generate random paths on GPU
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            n_assets: Number of assets
            random_type: Type of random numbers ('normal', 'uniform')
            
        Returns:
            Random paths array
        """
        if self.gpu.backend == 'cupy':
            return self._cupy_random_paths(n_paths, n_steps, n_assets, random_type)
        elif self.gpu.backend == 'numba':
            return self._numba_random_paths(n_paths, n_steps, n_assets, random_type)
        else:
            return self._cpu_random_paths(n_paths, n_steps, n_assets, random_type)
            
    def _cupy_random_paths(
        self, n_paths: int, n_steps: int, n_assets: int, random_type: str
    ) -> np.ndarray:
        """Generate random paths using CuPy"""
        shape = (n_paths, n_steps, n_assets) if n_assets > 1 else (n_paths, n_steps)
        
        if random_type == 'normal':
            gpu_paths = cp.random.normal(0, 1, shape, dtype=cp.float32)
        elif random_type == 'uniform':
            gpu_paths = cp.random.uniform(0, 1, shape, dtype=cp.float32)
        else:
            raise ValueError(f"Unknown random type: {random_type}")
            
        return self.gpu.to_cpu(gpu_paths)
        
    def _numba_random_paths(
        self, n_paths: int, n_steps: int, n_assets: int, random_type: str
    ) -> np.ndarray:
        """Generate random paths using Numba CUDA"""
        if not NUMBA_AVAILABLE:
            return self._cpu_random_paths(n_paths, n_steps, n_assets, random_type)
            
        @cuda.jit
        def generate_normal_kernel(rng_states, out):
            thread_id = cuda.grid(1)
            if thread_id < out.size:
                # Generate normal random number using Box-Muller
                # This is simplified - would need proper RNG state handling
                out.flat[thread_id] = 0.0  # Placeholder
                
        shape = (n_paths, n_steps, n_assets) if n_assets > 1 else (n_paths, n_steps)
        
        # Allocate GPU memory
        gpu_paths = cuda.device_array(shape, dtype=np.float32)
        
        # Launch kernel (simplified)
        threads_per_block = 256
        blocks_per_grid = (gpu_paths.size + threads_per_block - 1) // threads_per_block
        
        # Would need proper RNG initialization here
        rng_states = None  # Placeholder
        
        # For now, use numpy on CPU
        return self._cpu_random_paths(n_paths, n_steps, n_assets, random_type)
        
    def _cpu_random_paths(
        self, n_paths: int, n_steps: int, n_assets: int, random_type: str
    ) -> np.ndarray:
        """CPU fallback for random path generation"""
        shape = (n_paths, n_steps, n_assets) if n_assets > 1 else (n_paths, n_steps)
        
        if random_type == 'normal':
            return np.random.normal(0, 1, shape).astype(np.float32)
        elif random_type == 'uniform':
            return np.random.uniform(0, 1, shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown random type: {random_type}")
            
    def geometric_brownian_motion(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion on GPU
        
        Args:
            S0: Initial price
            mu: Drift parameter
            sigma: Volatility parameter
            T: Time horizon
            n_paths: Number of paths
            n_steps: Number of time steps
            
        Returns:
            Price paths
        """
        dt = T / n_steps
        
        if self.gpu.backend == 'cupy':
            return self._cupy_gbm(S0, mu, sigma, dt, n_paths, n_steps)
        else:
            return self._cpu_gbm(S0, mu, sigma, dt, n_paths, n_steps)
            
    def _cupy_gbm(
        self, S0: float, mu: float, sigma: float, dt: float, n_paths: int, n_steps: int
    ) -> np.ndarray:
        """CuPy implementation of GBM"""
        # Generate random increments
        dW = cp.random.normal(0, cp.sqrt(dt), (n_paths, n_steps), dtype=cp.float32)
        
        # Calculate log price increments
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * dW
        
        log_increments = drift_term + diffusion_term
        
        # Calculate price paths
        log_S = cp.zeros((n_paths, n_steps + 1), dtype=cp.float32)
        log_S[:, 0] = cp.log(S0)
        log_S[:, 1:] = cp.cumsum(log_increments, axis=1)
        
        S = cp.exp(log_S)
        
        return self.gpu.to_cpu(S)
        
    def _cpu_gbm(
        self, S0: float, mu: float, sigma: float, dt: float, n_paths: int, n_steps: int
    ) -> np.ndarray:
        """CPU implementation of GBM"""
        # Generate random increments
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        # Calculate log price increments
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * dW
        
        log_increments = drift_term + diffusion_term
        
        # Calculate price paths
        log_S = np.zeros((n_paths, n_steps + 1))
        log_S[:, 0] = np.log(S0)
        log_S[:, 1:] = np.cumsum(log_increments, axis=1)
        
        S = np.exp(log_S)
        
        return S
        
    def parallel_option_pricing(
        self,
        paths: np.ndarray,
        payoff_function: Callable,
        discount_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Parallel option pricing on GPU
        
        Args:
            paths: Price paths
            payoff_function: Option payoff function
            discount_factor: Discount factor
            
        Returns:
            Pricing results
        """
        if self.gpu.backend == 'cupy':
            return self._cupy_option_pricing(paths, payoff_function, discount_factor)
        else:
            return self._cpu_option_pricing(paths, payoff_function, discount_factor)
            
    def _cupy_option_pricing(
        self, paths: np.ndarray, payoff_function: Callable, discount_factor: float
    ) -> Dict[str, float]:
        """CuPy parallel option pricing"""
        # Transfer paths to GPU
        gpu_paths = self.gpu.to_gpu(paths)
        
        # Calculate payoffs (vectorized)
        if hasattr(payoff_function, '__name__') and 'call' in payoff_function.__name__:
            # European call option
            strike = getattr(payoff_function, 'strike', 100.0)
            payoffs = cp.maximum(gpu_paths[:, -1] - strike, 0)
        else:
            # General payoff function (apply to CPU data)
            cpu_paths = self.gpu.to_cpu(gpu_paths)
            payoffs = cp.array([payoff_function(path) for path in cpu_paths])
            
        # Calculate statistics on GPU
        discounted_payoffs = payoffs * discount_factor
        
        option_price = cp.mean(discounted_payoffs)
        std_error = cp.std(discounted_payoffs) / cp.sqrt(len(discounted_payoffs))
        
        # Confidence intervals
        confidence_95 = 1.96 * std_error
        
        return {
            'price': float(self.gpu.to_cpu(option_price)),
            'std_error': float(self.gpu.to_cpu(std_error)),
            'confidence_interval_95': (
                float(self.gpu.to_cpu(option_price - confidence_95)),
                float(self.gpu.to_cpu(option_price + confidence_95))
            )
        }
        
    def _cpu_option_pricing(
        self, paths: np.ndarray, payoff_function: Callable, discount_factor: float
    ) -> Dict[str, float]:
        """CPU fallback for option pricing"""
        # Calculate payoffs
        payoffs = np.array([payoff_function(path) for path in paths])
        
        # Calculate statistics
        discounted_payoffs = payoffs * discount_factor
        
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(len(discounted_payoffs))
        
        # Confidence intervals
        confidence_95 = 1.96 * std_error
        
        return {
            'price': float(option_price),
            'std_error': float(std_error),
            'confidence_interval_95': (
                float(option_price - confidence_95),
                float(option_price + confidence_95)
            )
        }

class OpenCLPathGenerator:
    """OpenCL-based path generation for cross-platform GPU acceleration"""
    
    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        self.kernels = {}
        
        if self.gpu.backend == 'opencl':
            self._compile_kernels()
            
    def _compile_kernels(self):
        """Compile OpenCL kernels"""
        # Random number generation kernel
        rng_kernel_source = """
        __kernel void generate_normal_random(
            __global float* output,
            __global uint* seeds,
            uint n_samples
        ) {
            int gid = get_global_id(0);
            if (gid >= n_samples) return;
            
            // Simple linear congruential generator
            uint seed = seeds[gid];
            seed = (1103515245 * seed + 12345) & 0x7fffffff;
            seeds[gid] = seed;
            
            // Box-Muller transformation (simplified)
            float u1 = (float)seed / 0x7fffffff;
            seed = (1103515245 * seed + 12345) & 0x7fffffff;
            float u2 = (float)seed / 0x7fffffff;
            
            float normal = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
            output[gid] = normal;
        }
        """
        
        # GBM simulation kernel
        gbm_kernel_source = """
        __kernel void simulate_gbm(
            __global float* paths,
            __global float* random_numbers,
            float S0,
            float mu,
            float sigma,
            float dt,
            uint n_paths,
            uint n_steps
        ) {
            int path_id = get_global_id(0);
            if (path_id >= n_paths) return;
            
            float S = S0;
            paths[path_id * (n_steps + 1)] = S;
            
            for (int t = 0; t < n_steps; t++) {
                float dW = random_numbers[path_id * n_steps + t];
                float drift = (mu - 0.5f * sigma * sigma) * dt;
                float diffusion = sigma * sqrt(dt) * dW;
                
                S = S * exp(drift + diffusion);
                paths[path_id * (n_steps + 1) + t + 1] = S;
            }
        }
        """
        
        try:
            program = cl.Program(self.gpu.cl_context, rng_kernel_source).build()
            self.kernels['rng'] = program.generate_normal_random
            
            program = cl.Program(self.gpu.cl_context, gbm_kernel_source).build()
            self.kernels['gbm'] = program.simulate_gbm
            
        except Exception as e:
            logger.error(f"Failed to compile OpenCL kernels: {e}")
            
    def generate_paths_opencl(
        self,
        S0: float,
        mu: float,
        sigma: float,
        T: float,
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        """Generate paths using OpenCL"""
        if 'gbm' not in self.kernels:
            # Fallback to CPU
            dt = T / n_steps
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            
            paths = np.zeros((n_paths, n_steps + 1))
            paths[:, 0] = S0
            
            for t in range(n_steps):
                drift = (mu - 0.5 * sigma**2) * dt
                diffusion = sigma * dW[:, t]
                paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion)
                
            return paths
            
        dt = T / n_steps
        
        # Allocate GPU memory
        paths_gpu = cl_array.zeros(self.gpu.cl_queue, (n_paths, n_steps + 1), np.float32)
        random_gpu = cl_array.zeros(self.gpu.cl_queue, (n_paths, n_steps), np.float32)
        
        # Generate random numbers
        seeds = np.random.randint(0, 2**31, n_paths, dtype=np.uint32)
        seeds_gpu = cl_array.to_device(self.gpu.cl_queue, seeds)
        
        # Launch RNG kernel
        self.kernels['rng'](
            self.gpu.cl_queue,
            (n_paths * n_steps,),
            None,
            random_gpu.data,
            seeds_gpu.data,
            np.uint32(n_paths * n_steps)
        )
        
        # Launch GBM kernel
        self.kernels['gbm'](
            self.gpu.cl_queue,
            (n_paths,),
            None,
            paths_gpu.data,
            random_gpu.data,
            np.float32(S0),
            np.float32(mu),
            np.float32(sigma),
            np.float32(dt),
            np.uint32(n_paths),
            np.uint32(n_steps)
        )
        
        # Transfer result back to CPU
        return paths_gpu.get()

class GPUMemoryManager:
    """GPU memory management utilities"""
    
    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        self.allocated_arrays = []
        
    def allocate_gpu_array(self, shape: Tuple[int, ...], dtype=np.float32):
        """Allocate array on GPU with tracking"""
        if self.gpu.backend == 'cupy':
            array = cp.zeros(shape, dtype=dtype)
        elif self.gpu.backend == 'opencl':
            array = cl_array.zeros(self.gpu.cl_queue, shape, dtype)
        else:
            array = np.zeros(shape, dtype=dtype)
            
        self.allocated_arrays.append(array)
        return array
        
    def free_gpu_memory(self):
        """Free all tracked GPU arrays"""
        if self.gpu.backend == 'cupy':
            # CuPy arrays are garbage collected automatically
            # But we can help by deleting references
            for array in self.allocated_arrays:
                del array
            # Force garbage collection
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            
        self.allocated_arrays.clear()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if self.gpu.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            total_bytes = mempool.total_bytes()
            
            return {
                'used_gb': used_bytes / (1024**3),
                'total_gb': total_bytes / (1024**3),
                'utilization': used_bytes / max(total_bytes, 1) * 100
            }
        else:
            return {'used_gb': 0, 'total_gb': 0, 'utilization': 0}

# Example usage and testing
if __name__ == "__main__":
    print("Testing GPU Acceleration...")
    
    # Initialize GPU accelerator
    gpu = GPUAccelerator(backend='auto')
    
    if gpu.is_gpu_available():
        print(f"GPU acceleration available using {gpu.backend}")
        if gpu.device_info:
            print(f"Device: {gpu.device_info.device_name}")
            print(f"Memory: {gpu.device_info.memory_total / (1024**3):.1f} GB")
    else:
        print("GPU acceleration not available, using CPU")
        
    # Test CUDA Monte Carlo Engine
    print("\nTesting CUDA Monte Carlo Engine:")
    mc_engine = CUDAMonteCarloEngine(gpu)
    
    # Parameters for GBM simulation
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    T = 1.0
    n_paths = 10000
    n_steps = 252
    
    # Time the simulation
    start_time = time.time()
    paths = mc_engine.geometric_brownian_motion(S0, mu, sigma, T, n_paths, n_steps)
    simulation_time = time.time() - start_time
    
    print(f"Generated {n_paths} paths with {n_steps} steps in {simulation_time:.3f} seconds")
    print(f"Final price statistics:")
    print(f"  Mean: {np.mean(paths[:, -1]):.2f}")
    print(f"  Std: {np.std(paths[:, -1]):.2f}")
    print(f"  Theoretical mean: {S0 * np.exp(mu * T):.2f}")
    
    # Test option pricing
    print("\nTesting GPU Option Pricing:")
    
    def european_call_payoff(path, strike=105.0):
        return max(path[-1] - strike, 0)
    
    # Set strike for the payoff function
    european_call_payoff.strike = 105.0
    
    start_time = time.time()
    pricing_result = mc_engine.parallel_option_pricing(
        paths, european_call_payoff, discount_factor=np.exp(-0.03 * T)
    )
    pricing_time = time.time() - start_time
    
    print(f"Option pricing completed in {pricing_time:.3f} seconds")
    print(f"Option price: {pricing_result['price']:.4f}")
    print(f"Standard error: {pricing_result['std_error']:.4f}")
    print(f"95% CI: [{pricing_result['confidence_interval_95'][0]:.4f}, "
          f"{pricing_result['confidence_interval_95'][1]:.4f}]")
    
    # Test memory management
    if gpu.is_gpu_available():
        print("\nTesting GPU Memory Management:")
        memory_manager = GPUMemoryManager(gpu)
        
        # Allocate some arrays
        for i in range(5):
            array = memory_manager.allocate_gpu_array((1000, 1000))
            
        memory_usage = memory_manager.get_memory_usage()
        print(f"Memory usage: {memory_usage['used_gb']:.2f} GB "
              f"({memory_usage['utilization']:.1f}%)")
        
        # Free memory
        memory_manager.free_gpu_memory()
        
        memory_usage_after = memory_manager.get_memory_usage()
        print(f"Memory after cleanup: {memory_usage_after['used_gb']:.2f} GB")
    
    print("\nGPU acceleration test completed!")
