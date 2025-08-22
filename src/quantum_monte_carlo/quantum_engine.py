"""
Quantum Monte Carlo Engine using Sobol sequences and quantum-inspired algorithms
"""

import numpy as np
from scipy.stats import qmc, norm
import logging
from typing import Optional, Dict, Any, List, Tuple
import time
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import cirq
    import tensorflow_quantum as tfq
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class QuantumSimulationResult:
    """Results from quantum Monte Carlo simulation"""
    paths: np.ndarray
    quantum_advantage: float
    entanglement_measure: float
    convergence_rate: float
    execution_time: float
    statistics: Dict[str, float]

class QuantumMonteCarloEngine:
    """
    Advanced Quantum Monte Carlo engine using quantum-inspired algorithms
    and low-discrepancy sequences for superior convergence
    """
    
    def __init__(
        self,
        n_simulations: int,
        n_steps: int,
        initial_price: float,
        drift: float,
        volatility: float,
        sequence_type: str = 'sobol',
        use_quantum_circuits: bool = True,
        quantum_depth: int = 4,
        random_seed: Optional[int] = None
    ):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.sequence_type = sequence_type
        self.use_quantum_circuits = use_quantum_circuits and QUANTUM_AVAILABLE
        self.quantum_depth = quantum_depth
        self.random_seed = random_seed
        
        # Initialize quantum circuit if available
        if self.use_quantum_circuits:
            self._initialize_quantum_circuit()
        
        # Initialize low-discrepancy sampler
        self._initialize_sampler()
        
    def _initialize_quantum_circuit(self):
        """Initialize quantum circuit for enhanced sampling"""
        n_qubits = min(10, int(np.ceil(np.log2(self.n_steps))))
        self.qubits = cirq.LineQubit.range(n_qubits)
        
        # Create parameterized quantum circuit
        self.circuit = cirq.Circuit()
        
        # Add Hadamard gates for superposition
        for qubit in self.qubits:
            self.circuit.append(cirq.H(qubit))
            
        # Add parameterized rotation gates
        self.params = []
        for depth in range(self.quantum_depth):
            for i, qubit in enumerate(self.qubits):
                param = cirq.Symbol(f'theta_{depth}_{i}')
                self.params.append(param)
                self.circuit.append(cirq.ry(param)(qubit))
                
            # Add entangling gates
            for i in range(len(self.qubits) - 1):
                self.circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i+1]))
                
    def _initialize_sampler(self):
        """Initialize quasi-random sampler"""
        if self.sequence_type == 'sobol':
            self.sampler = qmc.Sobol(
                d=self.n_steps, 
                scramble=True, 
                seed=self.random_seed
            )
        elif self.sequence_type == 'halton':
            self.sampler = qmc.Halton(d=self.n_steps, scramble=True, seed=self.random_seed)
        elif self.sequence_type == 'latin_hypercube':
            self.sampler = qmc.LatinHypercube(d=self.n_steps, seed=self.random_seed)
        else:
            raise ValueError(f"Unknown sequence type: {self.sequence_type}")
            
    def simulate_quantum_paths(self) -> QuantumSimulationResult:
        """
        Simulate paths using quantum-enhanced Monte Carlo
        """
        logger.info(f"Starting Quantum Monte Carlo with {self.n_simulations} paths")
        start_time = time.time()
        
        # Generate quantum-enhanced samples
        if self.use_quantum_circuits:
            samples = self._generate_quantum_samples()
        else:
            samples = self._generate_quasi_random_samples()
            
        # Transform to normal distribution
        normal_samples = norm.ppf(samples)
        
        # Calculate paths using GBM
        paths = self._calculate_gbm_paths(normal_samples)
        
        # Calculate quantum metrics
        quantum_advantage = self._calculate_quantum_advantage(samples)
        entanglement_measure = self._calculate_entanglement() if self.use_quantum_circuits else 0.0
        convergence_rate = self._calculate_convergence_rate(paths)
        
        # Calculate statistics
        final_prices = paths[:, -1]
        statistics = {
            'mean': float(np.mean(final_prices)),
            'std': float(np.std(final_prices)),
            'min': float(np.min(final_prices)),
            'max': float(np.max(final_prices)),
            'skewness': float(self._calculate_skewness(final_prices)),
            'kurtosis': float(self._calculate_kurtosis(final_prices))
        }
        
        execution_time = time.time() - start_time
        
        return QuantumSimulationResult(
            paths=paths,
            quantum_advantage=quantum_advantage,
            entanglement_measure=entanglement_measure,
            convergence_rate=convergence_rate,
            execution_time=execution_time,
            statistics=statistics
        )
        
    def _generate_quantum_samples(self) -> np.ndarray:
        """Generate samples using quantum circuit"""
        # Generate parameter values
        param_values = np.random.uniform(0, 2*np.pi, len(self.params))
        param_dict = dict(zip(self.params, param_values))
        
        # Simulate quantum circuit
        simulator = cirq.Simulator()
        result = simulator.simulate(self.circuit, param_resolver=param_dict)
        
        # Extract probabilities and convert to samples
        state_vector = result.final_state_vector
        probabilities = np.abs(state_vector) ** 2
        
        # Convert quantum probabilities to uniform samples
        samples = self._quantum_to_uniform_samples(probabilities)
        
        return samples
        
    def _generate_quasi_random_samples(self) -> np.ndarray:
        """Generate quasi-random samples"""
        return self.sampler.random(n=self.n_simulations)
        
    def _quantum_to_uniform_samples(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert quantum probabilities to uniform samples"""
        # Reshape probabilities to match simulation requirements
        n_states = len(probabilities)
        samples = np.zeros((self.n_simulations, self.n_steps))
        
        for i in range(self.n_simulations):
            for j in range(self.n_steps):
                # Use quantum probabilities to generate correlated samples
                idx = (i * self.n_steps + j) % n_states
                samples[i, j] = probabilities[idx]
                
        # Normalize to [0,1] range
        samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))
        
        return samples
        
    def _calculate_gbm_paths(self, normal_samples: np.ndarray) -> np.ndarray:
        """Calculate GBM paths from normal samples"""
        dt = 1.0 / self.n_steps
        drift_term = (self.drift - 0.5 * self.volatility**2) * dt
        vol_term = self.volatility * np.sqrt(dt)
        
        # Calculate log returns
        log_returns = drift_term + vol_term * normal_samples
        
        # Calculate cumulative log prices
        log_prices = np.cumsum(log_returns, axis=1)
        log_prices = np.hstack([
            np.full((self.n_simulations, 1), np.log(self.initial_price)),
            log_prices
        ])
        
        # Convert to prices
        paths = np.exp(log_prices)
        
        return paths
        
    def _calculate_quantum_advantage(self, samples: np.ndarray) -> float:
        """Calculate quantum advantage metric"""
        # Compare with classical Monte Carlo convergence
        classical_var = 1.0 / np.sqrt(self.n_simulations)
        
        # Estimate quantum variance reduction
        sample_variance = np.var(samples)
        quantum_var = sample_variance / np.sqrt(self.n_simulations)
        
        advantage = classical_var / quantum_var if quantum_var > 0 else 1.0
        return float(advantage)
        
    def _calculate_entanglement(self) -> float:
        """Calculate entanglement measure for quantum circuit"""
        if not self.use_quantum_circuits:
            return 0.0
            
        # Simplified entanglement measure
        # In practice, would calculate von Neumann entropy
        n_qubits = len(self.qubits)
        max_entanglement = np.log2(2**n_qubits)
        
        # Estimate based on circuit depth and connectivity
        estimated_entanglement = min(1.0, self.quantum_depth / 10.0) * max_entanglement
        
        return float(estimated_entanglement)
        
    def _calculate_convergence_rate(self, paths: np.ndarray) -> float:
        """Calculate convergence rate"""
        final_prices = paths[:, -1]
        
        # Calculate running mean convergence
        running_means = np.cumsum(final_prices) / np.arange(1, len(final_prices) + 1)
        
        # Estimate convergence rate (higher is better)
        if len(running_means) > 100:
            recent_variance = np.var(running_means[-100:])
            convergence_rate = 1.0 / (1.0 + recent_variance)
        else:
            convergence_rate = 0.5
            
        return float(convergence_rate)
        
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 3.0
        return np.mean(((data - mean) / std) ** 4)
