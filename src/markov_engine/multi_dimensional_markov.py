"""
Multi-dimensional Markov Chain for modeling correlated financial time series
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
import logging

logger = logging.getLogger(__name__)

@dataclass
class MultiDimensionalState:
    """Multi-dimensional state representation"""
    id: int
    center: np.ndarray
    covariance: np.ndarray
    frequency: int = 0
    
class MultiDimensionalMarkovChain:
    """Multi-dimensional Markov chain for correlated time series"""
    
    def __init__(
        self,
        n_states: int = 8,
        n_dimensions: int = 2,
        max_order: int = 1,
        clustering_method: str = 'kmeans',  # 'kmeans', 'gmm', 'quantile'
        min_observations: int = 50
    ):
        self.n_states = n_states
        self.n_dimensions = n_dimensions
        self.max_order = max_order
        self.clustering_method = clustering_method
        self.min_observations = min_observations
        
        # Core components
        self.states: Dict[int, MultiDimensionalState] = {}
        self.transition_matrices: Dict[int, np.ndarray] = {}
        self.state_sequences: List[int] = []
        self.optimal_order: int = 1
        
        # Model selection
        self.aic_scores: Dict[int, float] = {}
        self.bic_scores: Dict[int, float] = {}
        
        # Data storage
        self.data_mean: Optional[np.ndarray] = None
        self.data_cov: Optional[np.ndarray] = None
        
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'MultiDimensionalMarkovChain':
        """
        Fit multi-dimensional Markov chain
        
        Args:
            data: Multi-dimensional time series data (T x D)
            
        Returns:
            Self for method chaining
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional (time x features)")
            
        if data.shape[1] != self.n_dimensions:
            logger.warning(f"Data dimensions ({data.shape[1]}) != n_dimensions ({self.n_dimensions})")
            self.n_dimensions = data.shape[1]
            
        # Remove rows with NaN
        valid_mask = ~np.isnan(data).any(axis=1)
        data = data[valid_mask]
        
        if len(data) < self.min_observations:
            raise ValueError(f"Insufficient data: {len(data)} < {self.min_observations}")
            
        # Store data statistics
        self.data_mean = np.mean(data, axis=0)
        self.data_cov = np.cov(data.T)
        
        # Create multi-dimensional state space
        self.states = self._create_state_space(data)
        
        # Convert data to state sequence
        self.state_sequences = self._data_to_states(data)
        
        # Determine optimal order
        self.optimal_order = self._select_optimal_order()
        
        # Estimate transition matrices
        for order in range(1, min(self.max_order + 1, len(self.state_sequences))):
            self.transition_matrices[order] = self._estimate_transition_matrix(order)
            
        logger.info(f"Multi-dimensional Markov chain fitted: "
                   f"{len(self.states)} states, {self.n_dimensions} dimensions, "
                   f"optimal order: {self.optimal_order}")
        
        return self
        
    def _create_state_space(self, data: np.ndarray) -> Dict[int, MultiDimensionalState]:
        """Create multi-dimensional state space"""
        states = {}
        
        if self.clustering_method == 'kmeans':
            states = self._create_states_kmeans(data)
        elif self.clustering_method == 'gmm':
            states = self._create_states_gmm(data)
        elif self.clustering_method == 'quantile':
            states = self._create_states_quantile(data)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
        return states
        
    def _create_states_kmeans(self, data: np.ndarray) -> Dict[int, MultiDimensionalState]:
        """Create states using K-means clustering"""
        states = {}
        
        try:
            kmeans = KMeans(
                n_clusters=self.n_states, 
                random_state=42,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(data)
            centers = kmeans.cluster_centers_
            
            for i in range(self.n_states):
                cluster_data = data[labels == i]
                
                if len(cluster_data) > 1:
                    covariance = np.cov(cluster_data.T)
                else:
                    # Use small diagonal covariance for single points
                    covariance = np.eye(self.n_dimensions) * 0.01
                    
                # Ensure covariance is positive definite
                covariance = self._ensure_positive_definite(covariance)
                
                states[i] = MultiDimensionalState(
                    id=i,
                    center=centers[i],
                    covariance=covariance,
                    frequency=int(np.sum(labels == i))
                )
                
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}, using quantile method")
            states = self._create_states_quantile(data)
            
        return states
        
    def _create_states_gmm(self, data: np.ndarray) -> Dict[int, MultiDimensionalState]:
        """Create states using Gaussian Mixture Model"""
        states = {}
        
        try:
            gmm = GaussianMixture(
                n_components=self.n_states,
                random_state=42,
                max_iter=200,
                tol=1e-4
            )
            labels = gmm.fit_predict(data)
            
            for i in range(self.n_states):
                states[i] = MultiDimensionalState(
                    id=i,
                    center=gmm.means_[i],
                    covariance=self._ensure_positive_definite(gmm.covariances_[i]),
                    frequency=int(np.sum(labels == i))
                )
                
        except Exception as e:
            logger.warning(f"GMM clustering failed: {e}, using K-means")
            states = self._create_states_kmeans(data)
            
        return states
        
    def _create_states_quantile(self, data: np.ndarray) -> Dict[int, MultiDimensionalState]:
        """Create states using quantile-based grid"""
        states = {}
        
        # Create grid based on marginal quantiles
        n_per_dim = int(np.ceil(self.n_states ** (1.0 / self.n_dimensions)))
        
        # Get quantiles for each dimension
        quantiles = np.linspace(0, 1, n_per_dim + 1)
        dim_thresholds = []
        
        for d in range(self.n_dimensions):
            thresholds = np.quantile(data[:, d], quantiles)
            dim_thresholds.append(thresholds)
            
        # Create states from grid
        state_id = 0
        
        for i in range(n_per_dim):
            if state_id >= self.n_states:
                break
                
            for j in range(n_per_dim):
                if state_id >= self.n_states:
                    break
                    
                # Define state region
                center = np.array([
                    (dim_thresholds[0][i] + dim_thresholds[i + 1]) / 2,
                    (dim_thresholds[1][j] + dim_thresholds[1][j + 1]) / 2
                ])
                
                # Find data points in this region
                mask = ((data[:, 0] >= dim_thresholds[i]) & 
                       (data[:, 0] < dim_thresholds[i + 1]) &
                       (data[:, 1] >= dim_thresholds[1][j]) & 
                       (data[:, 1] < dim_thresholds[1][j + 1]))
                       
                region_data = data[mask]
                
                if len(region_data) > 1:
                    covariance = np.cov(region_data.T)
                    actual_center = np.mean(region_data, axis=0)
                else:
                    # Default covariance and center
                    covariance = np.eye(self.n_dimensions) * 0.01
                    actual_center = center
                    
                covariance = self._ensure_positive_definite(covariance)
                
                states[state_id] = MultiDimensionalState(
                    id=state_id,
                    center=actual_center,
                    covariance=covariance,
                    frequency=len(region_data)
                )
                
                state_id += 1
                
        return states
        
    def _ensure_positive_definite(self, matrix: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
        """Ensure covariance matrix is positive definite"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, min_eigenval)
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
    def _data_to_states(self, data: np.ndarray) -> List[int]:
        """Convert data to state sequence using nearest state assignment"""
        state_sequence = []
        
        for point in data:
            # Find closest state by Mahalanobis distance
            min_distance = float('inf')
            closest_state = 0
            
            for state_id, state in self.states.items():
                try:
                    # Mahalanobis distance
                    diff = point - state.center
                    inv_cov = np.linalg.inv(state.covariance)
                    distance = np.sqrt(diff.T @ inv_cov @ diff)
                except np.linalg.LinAlgError:
                    # Fallback to Euclidean distance
                    distance = euclidean(point, state.center)
                    
                if distance < min_distance:
                    min_distance = distance
                    closest_state = state_id
                    
            state_sequence.append(closest_state)
            
        return state_sequence
        
    def _select_optimal_order(self) -> int:
        """Select optimal Markov chain order"""
        if len(self.state_sequences) < 10:
            return 1
            
        best_order = 1
        best_score = float('inf')
        
        for order in range(1, min(self.max_order + 1, len(self.state_sequences) // 5)):
            try:
                # Calculate log-likelihood
                ll = self._calculate_log_likelihood(order)
                
                # Calculate BIC
                n_params = len(self.states) ** (order + 1)
                n_obs = len(self.state_sequences) - order
                
                if n_obs > 0 and np.isfinite(ll):
                    bic = -2 * ll + np.log(n_obs) * n_params
                    self.bic_scores[order] = bic
                    
                    if bic < best_score:
                        best_score = bic
                        best_order = order
                        
            except Exception as e:
                logger.warning(f"Failed to evaluate order {order}: {e}")
                continue
                
        return best_order
        
    def _calculate_log_likelihood(self, order: int) -> float:
        """Calculate log-likelihood for given order"""
        if len(self.state_sequences) <= order:
            return -np.inf
            
        transition_matrix = self._estimate_transition_matrix(order)
        ll = 0.0
        
        for i in range(order, len(self.state_sequences)):
            if order == 1:
                prev_state = self.state_sequences[i - 1]
            else:
                # For higher orders, we need to handle state history
                history = tuple(self.state_sequences[i - order:i])
                prev_state = self._history_to_index(history[:-1])
                
            current_state = self.state_sequences[i]
            
            try:
                if prev_state < transition_matrix.shape[0]:
                    prob = transition_matrix[prev_state, current_state]
                    if prob > 0:
                        ll += np.log(prob)
                    else:
                        ll += np.log(1e-10)
            except IndexError:
                ll += np.log(1e-10)
                
        return ll
        
    def _history_to_index(self, history: Tuple[int, ...]) -> int:
        """Convert state history to linear index"""
        index = 0
        for i, state in enumerate(history):
            index += state * (len(self.states) ** i)
        return index
        
    def _estimate_transition_matrix(self, order: int) -> np.ndarray:
        """Estimate transition matrix for given order"""
        n_states = len(self.states)
        
        if order == 1:
            transition_counts = np.zeros((n_states, n_states))
            
            for i in range(len(self.state_sequences) - 1):
                current_state = self.state_sequences[i]
                next_state = self.state_sequences[i + 1]
                transition_counts[current_state, next_state] += 1
                
        else:
            n_history_states = n_states ** order
            transition_counts = np.zeros((n_history_states, n_states))
            
            for i in range(order, len(self.state_sequences)):
                history = tuple(self.state_sequences[i - order:i])
                next_state = self.state_sequences[i]
                history_idx = self._history_to_index(history)
                
                if history_idx < n_history_states:
                    transition_counts[history_idx, next_state] += 1
                    
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = np.divide(
            transition_counts,
            row_sums[:, np.newaxis],
            out=np.zeros_like(transition_counts, dtype=float),
            where=row_sums[:, np.newaxis] != 0
        )
        
        return transition_matrix
        
    def simulate_path(
        self,
        initial_state: Union[int, List[int]],
        length: int,
        order: Optional[int] = None,
        return_values: bool = True
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Simulate path from Markov chain
        
        Args:
            initial_state: Starting state or sequence
            length: Path length to simulate
            order: Order to use (default: optimal_order)
            return_values: Whether to return actual values
            
        Returns:
            State sequence or tuple of (states, values)
        """
        if order is None:
            order = self.optimal_order
            
        if order not in self.transition_matrices:
            raise ValueError(f"No transition matrix available for order {order}")
            
        transition_matrix = self.transition_matrices[order]
        
        if isinstance(initial_state, int):
            path = [initial_state]
        else:
            path = list(initial_state)
            
        for _ in range(length):
            if order == 1:
                current_state = path[-1]
                probabilities = transition_matrix[current_state]
            else:
                if len(path) >= order:
                    history = tuple(path[-order:])
                    history_idx = self._history_to_index(history)
                    if history_idx < transition_matrix.shape[0]:
                        probabilities = transition_matrix[history_idx]
                    else:
                        probabilities = np.ones(len(self.states)) / len(self.states)
                else:
                    probabilities = np.ones(len(self.states)) / len(self.states)
                    
            # Sample next state
            next_state = np.random.choice(len(self.states), p=probabilities)
            path.append(next_state)
            
        simulated_path = path[-length:] if len(path) > length else path
        
        if return_values:
            values = self.states_to_values(simulated_path)
            return simulated_path, values
        else:
            return simulated_path
            
    def states_to_values(self, state_sequence: List[int]) -> np.ndarray:
        """Convert state sequence to multi-dimensional values"""
        values = np.zeros((len(state_sequence), self.n_dimensions))
        
        for i, state_id in enumerate(state_sequence):
            if state_id in self.states:
                state = self.states[state_id]
                # Sample from multivariate normal distribution
                values[i] = np.random.multivariate_normal(state.center, state.covariance)
            else:
                # Fallback to mean values
                values[i] = self.data_mean if self.data_mean is not None else np.zeros(self.n_dimensions)
                
        return values
        
    def predict_distribution(
        self,
        current_history: Union[List[int], int],
        steps_ahead: int = 1,
        order: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict state distribution steps ahead
        
        Args:
            current_history: Current state or sequence
            steps_ahead: Number of steps to predict ahead
            order: Order to use
            
        Returns:
            Probability distribution over states
        """
        if order is None:
            order = self.optimal_order
            
        if order not in self.transition_matrices:
            raise ValueError(f"No transition matrix available for order {order}")
            
        transition_matrix = self.transition_matrices[order]
        
        # Initialize current distribution
        if isinstance(current_history, int):
            current_dist = np.zeros(len(self.states))
            current_dist[current_history] = 1.0
        else:
            if order == 1:
                current_dist = np.zeros(len(self.states))
                current_dist[current_history[-1]] = 1.0
            else:
                # For higher orders, need to handle state history
                current_dist = np.zeros(transition_matrix.shape[0])
                history_idx = self._history_to_index(tuple(current_history[-order:]))
                if history_idx < len(current_dist):
                    current_dist[history_idx] = 1.0
                    
        # Evolve distribution forward
        for _ in range(steps_ahead):
            if order == 1:
                current_dist = current_dist @ transition_matrix
            else:
                # For higher orders, need to aggregate back to state distribution
                next_dist = current_dist @ transition_matrix
                # This is simplified - full implementation would track state histories
                current_dist = next_dist
                
        return current_dist[:len(self.states)]
        
    def get_correlation_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get correlation structure between dimensions
        
        Returns:
            Tuple of (correlation_matrix, state_correlations)
        """
        if self.data_cov is None:
            raise ValueError("Model must be fitted first")
            
        # Overall correlation matrix
        std_devs = np.sqrt(np.diag(self.data_cov))
        correlation_matrix = self.data_cov / np.outer(std_devs, std_devs)
        
        # State-specific correlations
        state_correlations = np.zeros((len(self.states), self.n_dimensions, self.n_dimensions))
        
        for i, state in self.states.items():
            state_std = np.sqrt(np.diag(state.covariance))
            state_correlations[i] = state.covariance / np.outer(state_std, state_std)
            
        return correlation_matrix, state_correlations
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'n_states': len(self.states),
            'n_dimensions': self.n_dimensions,
            'optimal_order': self.optimal_order,
            'max_order': self.max_order,
            'clustering_method': self.clustering_method,
            'n_observations': len(self.state_sequences),
            'bic_scores': self.bic_scores,
            'data_mean': self.data_mean.tolist() if self.data_mean is not None else None,
            'data_covariance': self.data_cov.tolist() if self.data_cov is not None else None,
            'state_definitions': {
                state_id: {
                    'center': state.center.tolist(),
                    'covariance': state.covariance.tolist(),
                    'frequency': state.frequency
                } for state_id, state in self.states.items()
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Generate correlated multi-dimensional financial data
    np.random.seed(42)
    n_obs = 1000
    n_dims = 3
    
    # Create regime-switching correlation structure
    regimes = [
        {'mean': np.array([0.001, 0.0005, 0.002]), 'cov': np.array([[0.0004, 0.0001, 0.0002],
                                                                    [0.0001, 0.0009, -0.0001],
                                                                    [0.0002, -0.0001, 0.0016]])},
        {'mean': np.array([-0.002, -0.001, -0.003]), 'cov': np.array([[0.0025, 0.0015, 0.0018],
                                                                       [0.0015, 0.0036, 0.0012],
                                                                       [0.0018, 0.0012, 0.0049]])},
        {'mean': np.array([0.0005, 0.001, 0.0008]), 'cov': np.array([[0.0009, -0.0003, 0.0001],
                                                                     [-0.0003, 0.0016, -0.0002],
                                                                     [0.0001, -0.0002, 0.0025]])}
    ]
    
    # Generate regime-switching data
    regime_changes = [0, 300, 600, 1000]
    data_segments = []
    
    for i in range(len(regime_changes) - 1):
        start, end = regime_changes[i], regime_changes[i + 1]
        regime = regimes[i % len(regimes)]
        
        segment = np.random.multivariate_normal(
            regime['mean'], 
            regime['cov'], 
            end - start
        )
        data_segments.append(segment)
        
    multi_dim_data = np.vstack(data_segments)
    
    print("Testing Multi-Dimensional Markov Chain...")
    print(f"Generated {multi_dim_data.shape[0]} observations with {multi_dim_data.shape[1]} dimensions")
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(multi_dim_data, columns=['Asset1', 'Asset2', 'Asset3'])
    
    # Test different clustering methods
    for method in ['kmeans', 'gmm', 'quantile']:
        print(f"\nTesting with {method} clustering:")
        
        try:
            # Initialize and fit model
            md_markov = MultiDimensionalMarkovChain(
                n_states=6,
                n_dimensions=n_dims,
                max_order=2,
                clustering_method=method
            )
            
            md_markov.fit(df)
            
            # Get model info
            info = md_markov.get_model_info()
            print(f"  Fitted with {info['n_states']} states, optimal order: {info['optimal_order']}")
            
            # Test simulation
            initial_state = 0
            sim_states, sim_values = md_markov.simulate_path(initial_state, length=10, return_values=True)
            print(f"  Simulated path: {sim_states}")
            print(f"  Sample values shape: {sim_values.shape}")
            
            # Test prediction
            pred_dist = md_markov.predict_distribution(initial_state, steps_ahead=5)
            print(f"  5-step prediction distribution: {pred_dist}")
            
            # Get correlation structure
            overall_corr, state_corrs = md_markov.get_correlation_structure()
            print(f"  Overall correlation matrix:\n{overall_corr}")
            
        except Exception as e:
            print(f"  Failed with error: {e}")
    
    print("\nMulti-dimensional Markov chain test completed!")
