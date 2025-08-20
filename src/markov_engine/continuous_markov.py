"""
Continuous-time Markov Chain implementation for financial modeling
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.integrate import odeint
import logging

logger = logging.getLogger(__name__)

@dataclass 
class ContinuousState:
    """Continuous state representation"""
    id: int
    mean: float
    std: float
    intensity: float = 1.0

class ContinuousMarkovChain:
    """Continuous-time Markov chain with intensity-based transitions"""
    
    def __init__(
        self,
        n_states: int = 5,
        dt: float = 1/252,  # Daily timestep
        estimation_method: str = 'mle'  # 'mle' or 'method_of_moments'
    ):
        self.n_states = n_states
        self.dt = dt
        self.estimation_method = estimation_method
        
        # Core components
        self.states: Dict[int, ContinuousState] = {}
        self.intensity_matrix: Optional[np.ndarray] = None  # Q matrix
        self.transition_matrix: Optional[np.ndarray] = None  # P(t) = exp(Q*t)
        self.stationary_distribution: Optional[np.ndarray] = None
        
        # Data storage
        self.observation_times: Optional[np.ndarray] = None
        self.state_sequence: Optional[List[int]] = None
        self.fitted = False
        
    def fit(self, data: Union[pd.Series, np.ndarray], times: Optional[np.ndarray] = None) -> 'ContinuousMarkovChain':
        """
        Fit continuous-time Markov chain to data
        
        Args:
            data: Time series data
            times: Observation times (if irregular)
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.Series):
            if times is None and isinstance(data.index, pd.DatetimeIndex):
                # Convert datetime index to numeric times (days)
                times = (data.index - data.index[0]).total_seconds() / (24 * 3600)
                times = times.values
            data = data.values
            
        if times is None:
            times = np.arange(len(data)) * self.dt
            
        # Remove NaN values
        valid_mask = ~np.isnan(data)
        data = data[valid_mask]
        times = times[valid_mask]
        
        if len(data) < 10:
            raise ValueError("Insufficient data for continuous Markov chain fitting")
            
        self.observation_times = times
        
        # Create state space using quantiles
        self.states = self._create_continuous_states(data)
        
        # Discretize data into states
        self.state_sequence = self._discretize_data(data)
        
        # Estimate intensity matrix
        self.intensity_matrix = self._estimate_intensity_matrix()
        
        # Calculate transition matrix
        self.transition_matrix = self._calculate_transition_matrix(self.dt)
        
        # Calculate stationary distribution
        self.stationary_distribution = self._calculate_stationary_distribution()
        
        self.fitted = True
        logger.info(f"Continuous Markov chain fitted with {self.n_states} states")
        
        return self
        
    def _create_continuous_states(self, data: np.ndarray) -> Dict[int, ContinuousState]:
        """Create continuous state space based on data distribution"""
        states = {}
        
        # Use quantiles to define states
        quantiles = np.linspace(0, 1, self.n_states + 1)
        thresholds = np.quantile(data, quantiles)
        
        for i in range(self.n_states):
            # Get data in this state
            if i == self.n_states - 1:
                mask = (data >= thresholds[i]) & (data <= thresholds[i + 1])
            else:
                mask = (data >= thresholds[i]) & (data < thresholds[i + 1])
                
            state_data = data[mask]
            
            if len(state_data) > 0:
                mean_val = np.mean(state_data)
                std_val = np.std(state_data) if len(state_data) > 1 else 0.1
            else:
                mean_val = (thresholds[i] + thresholds[i + 1]) / 2
                std_val = (thresholds[i + 1] - thresholds[i]) / 4
                
            states[i] = ContinuousState(
                id=i,
                mean=mean_val,
                std=max(std_val, 1e-6)  # Avoid zero std
            )
            
        return states
        
    def _discretize_data(self, data: np.ndarray) -> List[int]:
        """Convert continuous data to discrete state sequence"""
        state_sequence = []
        
        # Create thresholds from states
        sorted_states = sorted(self.states.values(), key=lambda s: s.mean)
        thresholds = [s.mean for s in sorted_states]
        
        for value in data:
            # Find closest state
            distances = [abs(value - threshold) for threshold in thresholds]
            closest_state = distances.index(min(distances))
            state_sequence.append(closest_state)
            
        return state_sequence
        
    def _estimate_intensity_matrix(self) -> np.ndarray:
        """Estimate the intensity matrix Q"""
        if self.estimation_method == 'mle':
            return self._estimate_intensity_mle()
        else:
            return self._estimate_intensity_moments()
            
    def _estimate_intensity_mle(self) -> np.ndarray:
        """Maximum likelihood estimation of intensity matrix"""
        # Count transitions and holding times
        transition_counts = np.zeros((self.n_states, self.n_states))
        holding_times = np.zeros(self.n_states)
        
        for i in range(len(self.state_sequence) - 1):
            current_state = self.state_sequence[i]
            next_state = self.state_sequence[i + 1]
            time_diff = self.observation_times[i + 1] - self.observation_times[i]
            
            if current_state == next_state:
                holding_times[current_state] += time_diff
            else:
                transition_counts[current_state, next_state] += 1
                holding_times[current_state] += time_diff
                
        # Build intensity matrix
        Q = np.zeros((self.n_states, self.n_states))
        
        for i in range(self.n_states):
            if holding_times[i] > 0:
                # Off-diagonal elements (transition rates)
                for j in range(self.n_states):
                    if i != j:
                        Q[i, j] = transition_counts[i, j] / holding_times[i]
                        
                # Diagonal elements (negative sum of off-diagonal)
                Q[i, i] = -np.sum(Q[i, :])
                
        return Q
        
    def _estimate_intensity_moments(self) -> np.ndarray:
        """Method of moments estimation"""
        # Simplified method using empirical transition probabilities
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for i in range(len(self.state_sequence) - 1):
            current_state = self.state_sequence[i]
            next_state = self.state_sequence[i + 1]
            transition_counts[current_state, next_state] += 1
            
        # Normalize to get transition probabilities
        row_sums = transition_counts.sum(axis=1)
        P_empirical = np.divide(
            transition_counts, 
            row_sums[:, np.newaxis], 
            out=np.zeros_like(transition_counts), 
            where=row_sums[:, np.newaxis] != 0
        )
        
        # Convert to intensity matrix (approximation)
        # P(dt) ≈ I + Q*dt for small dt
        Q = (P_empirical - np.eye(self.n_states)) / self.dt
        
        return Q
        
    def _calculate_transition_matrix(self, t: float) -> np.ndarray:
        """Calculate transition matrix P(t) = exp(Q*t)"""
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not estimated")
            
        return expm(self.intensity_matrix * t)
        
    def _calculate_stationary_distribution(self) -> np.ndarray:
        """Calculate stationary distribution π such that πQ = 0"""
        if self.intensity_matrix is None:
            raise ValueError("Intensity matrix not estimated")
            
        # Solve πQ = 0 with constraint Σπᵢ = 1
        # This is equivalent to finding the left eigenvector for eigenvalue 0
        eigenvals, eigenvecs = np.linalg.eig(self.intensity_matrix.T)
        
        # Find eigenvalue closest to 0
        stationary_idx = np.argmin(np.abs(eigenvals))
        stationary_dist = np.real(eigenvecs[:, stationary_idx])
        
        # Normalize to probability distribution
        stationary_dist = np.abs(stationary_dist)
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        
        return stationary_dist
        
    def simulate_path(
        self, 
        initial_state: int, 
        total_time: float,
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate continuous-time Markov chain path
        
        Args:
            initial_state: Starting state
            total_time: Total simulation time
            dt: Time step (uses self.dt if None)
            
        Returns:
            Tuple of (times, states)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
            
        if dt is None:
            dt = self.dt
            
        times = np.arange(0, total_time + dt, dt)
        states = np.zeros(len(times), dtype=int)
        states[0] = initial_state
        
        current_state = initial_state
        
        for i in range(1, len(times)):
            # Get transition probabilities for time step dt
            P_dt = self._calculate_transition_matrix(dt)
            
            # Sample next state
            probabilities = P_dt[current_state, :]
            next_state = np.random.choice(self.n_states, p=probabilities)
            
            states[i] = next_state
            current_state = next_state
            
        return times, states
        
    def simulate_jump_times(
        self, 
        initial_state: int, 
        max_time: float,
        max_jumps: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate exact jump times using exponential holding times
        
        Args:
            initial_state: Starting state
            max_time: Maximum simulation time
            max_jumps: Maximum number of jumps
            
        Returns:
            Tuple of (jump_times, states)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
            
        jump_times = [0.0]
        states = [initial_state]
        
        current_state = initial_state
        current_time = 0.0
        
        for _ in range(max_jumps):
            # Get holding time rate (negative diagonal element)
            rate = -self.intensity_matrix[current_state, current_state]
            
            if rate <= 0:
                # Absorbing state
                break
                
            # Sample holding time from exponential distribution
            holding_time = np.random.exponential(1.0 / rate)
            next_time = current_time + holding_time
            
            if next_time > max_time:
                break
                
            # Sample next state based on transition probabilities
            off_diagonal = self.intensity_matrix[current_state, :].copy()
            off_diagonal[current_state] = 0  # Remove diagonal
            
            if np.sum(off_diagonal) <= 0:
                break
                
            probabilities = off_diagonal / np.sum(off_diagonal)
            next_state = np.random.choice(self.n_states, p=probabilities)
            
            jump_times.append(next_time)
            states.append(next_state)
            
            current_state = next_state
            current_time = next_time
            
        return np.array(jump_times), np.array(states)
        
    def get_expected_holding_time(self, state: int) -> float:
        """Get expected holding time for a state"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting holding times")
            
        rate = -self.intensity_matrix[state, state]
        return 1.0 / rate if rate > 0 else np.inf
        
    def get_transition_rate(self, from_state: int, to_state: int) -> float:
        """Get transition rate from one state to another"""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting transition rates")
            
        if from_state == to_state:
            return 0.0
            
        return self.intensity_matrix[from_state, to_state]
        
    def predict_distribution(self, initial_state: int, time: float) -> np.ndarray:
        """
        Predict state distribution at future time
        
        Args:
            initial_state: Starting state
            time: Prediction time
            
        Returns:
            Probability distribution over states
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        P_t = self._calculate_transition_matrix(time)
        initial_dist = np.zeros(self.n_states)
        initial_dist[initial_state] = 1.0
        
        return initial_dist @ P_t
        
    def compute_first_passage_time(
        self, 
        start_state: int, 
        target_state: int,
        method: str = 'analytic'
    ) -> float:
        """
        Compute expected first passage time between states
        
        Args:
            start_state: Starting state
            target_state: Target state
            method: 'analytic' or 'simulation'
            
        Returns:
            Expected first passage time
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before computing passage times")
            
        if method == 'analytic':
            return self._compute_passage_time_analytic(start_state, target_state)
        else:
            return self._compute_passage_time_simulation(start_state, target_state)
            
    def _compute_passage_time_analytic(self, start_state: int, target_state: int) -> float:
        """Analytic computation of first passage time"""
        if start_state == target_state:
            return 0.0
            
        # Solve system of linear equations for mean first passage times
        # m_i = 1/(-q_ii) + Σⱼ≠ₜ (q_ij/(-q_ii)) * m_j
        
        n = self.n_states
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        for i in range(n):
            if i == target_state:
                A[i, i] = 1.0
                b[i] = 0.0
            else:
                rate = -self.intensity_matrix[i, i]
                if rate > 0:
                    A[i, i] = 1.0
                    b[i] = 1.0 / rate
                    
                    for j in range(n):
                        if i != j and j != target_state:
                            A[i, j] = -self.intensity_matrix[i, j] / rate
                            
        try:
            passage_times = np.linalg.solve(A, b)
            return passage_times[start_state]
        except np.linalg.LinAlgError:
            logger.warning("Analytic solution failed, using simulation")
            return self._compute_passage_time_simulation(start_state, target_state)
            
    def _compute_passage_time_simulation(
        self, 
        start_state: int, 
        target_state: int, 
        n_simulations: int = 10000
    ) -> float:
        """Simulation-based computation of first passage time"""
        passage_times = []
        
        for _ in range(n_simulations):
            jump_times, states = self.simulate_jump_times(start_state, max_time=1000.0)
            
            # Find first time target state is reached
            target_indices = np.where(states == target_state)[0]
            
            if len(target_indices) > 0:
                passage_times.append(jump_times[target_indices])
            else:
                passage_times.append(1000.0)  # Max time if not reached
                
        return np.mean(passage_times)
        
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        if not self.fitted:
            return {"fitted": False}
            
        return {
            "fitted": True,
            "n_states": self.n_states,
            "dt": self.dt,
            "estimation_method": self.estimation_method,
            "intensity_matrix": self.intensity_matrix.tolist(),
            "stationary_distribution": self.stationary_distribution.tolist(),
            "states": {
                i: {
                    "mean": state.mean,
                    "std": state.std,
                    "expected_holding_time": self.get_expected_holding_time(i)
                } for i, state in self.states.items()
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Generate sample financial return data with regime switching
    np.random.seed(42)
    
    # Simulate regime-switching process
    n_obs = 1000
    regime_changes = [0, 300, 600, 1000]
    regimes = [
        {'mean': 0.001, 'std': 0.01},   # Low volatility
        {'mean': -0.002, 'std': 0.03},  # High volatility, negative drift
        {'mean': 0.0005, 'std': 0.015}  # Medium volatility
    ]
    
    # Generate data with irregular time intervals
    times = []
    returns = []
    current_time = 0
    
    for i in range(len(regime_changes) - 1):
        start_idx, end_idx = regime_changes[i], regime_changes[i + 1]
        regime = regimes[i]
        
        for j in range(start_idx, end_idx):
            # Irregular time intervals (business days with some gaps)
            if np.random.random() > 0.05:  # 95% chance of trading
                dt = np.random.uniform(0.8, 1.2)  # Slight variation in intervals
                current_time += dt
                times.append(current_time)
                
                ret = np.random.normal(regime['mean'], regime['std'])
                returns.append(ret)
                
    times = np.array(times)
    returns = np.array(returns)
    
    print("Testing Continuous-Time Markov Chain...")
    print(f"Generated {len(returns)} observations over {times[-1]:.1f} days")
    
    # Fit continuous Markov chain
    continuous_mc = ContinuousMarkovChain(n_states=4, dt=1.0)
    continuous_mc.fit(returns, times)
    
    # Display model info
    info = continuous_mc.get_model_info()
    print(f"\nModel fitted with {info['n_states']} states:")
    
    print("\nState Information:")
    for state_id, state_info in info['states'].items():
        print(f"  State {state_id}: mean={state_info['mean']:.4f}, "
              f"std={state_info['std']:.4f}, "
              f"E[holding_time]={state_info['expected_holding_time']:.2f}")
              
    print(f"\nStationary distribution: {info['stationary_distribution']}")
    
    # Test simulation
    print("\nTesting simulation...")
    sim_times, sim_states = continuous_mc.simulate_path(initial_state=0, total_time=100.0)
    print(f"Simulated {len(sim_states)} steps over {sim_times[-1]:.1f} time units")
    
    # Test jump-time simulation  
    jump_times, jump_states = continuous_mc.simulate_jump_times(initial_state=0, max_time=50.0)
    print(f"Jump simulation: {len(jump_times)} jumps over {jump_times[-1]:.1f} time units")
    
    # Test first passage time
    passage_time = continuous_mc.compute_first_passage_time(0, 2, method='simulation')
    print(f"Expected first passage time from state 0 to 2: {passage_time:.2f}")
    
    # Test prediction
    future_dist = continuous_mc.predict_distribution(initial_state=0, time=10.0)
    print(f"State distribution after 10 time units: {future_dist}")
    
    print("\nContinuous-time Markov chain test completed successfully!")
