"""
Enhanced regime detection algorithms using advanced ML and statistical methods
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RegimeDetectionResult:
    """Results from regime detection analysis"""
    regime_probabilities: np.ndarray
    most_likely_regimes: np.ndarray
    regime_characteristics: Dict[str, Any]
    transition_matrix: np.ndarray
    log_likelihood: float
    n_regimes: int
    regime_names: List[str]
    confidence_scores: np.ndarray

class BaseRegimeDetector(ABC):
    """Abstract base class for regime detection algorithms"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BaseRegimeDetector':
        """Fit the regime detection model"""
        pass
        
    @abstractmethod
    def predict(self, data: np.ndarray) -> RegimeDetectionResult:
        """Predict regimes for new data"""
        pass
        
    @abstractmethod
    def get_regime_characteristics(self) -> Dict[str, Any]:
        """Get characteristics of detected regimes"""
        pass

class HiddenMarkovRegimeDetector(BaseRegimeDetector):
    """
    Enhanced Hidden Markov Model for regime detection with multiple features
    """
    
    def __init__(self, n_regimes: int = 3, n_iterations: int = 100, tolerance: float = 1e-6):
        super().__init__(n_regimes)
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.transition_matrix = None
        self.emission_params = None
        self.initial_probs = None
        
    def fit(self, data: np.ndarray) -> 'HiddenMarkovRegimeDetector':
        """
        Fit HMM using Baum-Welch algorithm
        
        Args:
            data: (T, n_features) array of observations
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        T, n_features = data.shape
        self.n_features = n_features
        
        # Initialize parameters
        self._initialize_parameters(data)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.n_iterations):
            # E-step: Forward-backward algorithm
            alpha, beta, log_likelihood = self._forward_backward(data)
            
            # M-step: Parameter updates
            self._update_parameters(data, alpha, beta)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tolerance:
                logger.info(f"HMM converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = log_likelihood
            
        self.log_likelihood = log_likelihood
        self.is_fitted = True
        return self
        
    def _initialize_parameters(self, data: np.ndarray):
        """Initialize HMM parameters"""
        T, n_features = data.shape
        
        # Initialize with K-means clustering
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        initial_labels = kmeans.fit_predict(data)
        
        # Initialize transition matrix (slightly favor staying in same state)
        self.transition_matrix = np.random.dirichlet(
            np.ones(self.n_regimes) * 0.1 + np.eye(self.n_regimes) * 2.0,
            size=self.n_regimes
        )
        
        # Initialize emission parameters (Gaussian)
        self.emission_params = {
            'means': np.zeros((self.n_regimes, n_features)),
            'covariances': np.zeros((self.n_regimes, n_features, n_features))
        }
        
        for k in range(self.n_regimes):
            mask = initial_labels == k
            if np.sum(mask) > 0:
                self.emission_params['means'][k] = np.mean(data[mask], axis=0)
                cov = np.cov(data[mask].T)
                if cov.ndim == 0:
                    cov = np.array([[cov]])
                elif cov.ndim == 1:
                    cov = np.diag(cov)
                self.emission_params['covariances'][k] = cov + np.eye(n_features) * 1e-6
            else:
                self.emission_params['means'][k] = np.mean(data, axis=0)
                self.emission_params['covariances'][k] = np.cov(data.T) + np.eye(n_features) * 1e-6
                
        # Initialize state probabilities
        self.initial_probs = np.ones(self.n_regimes) / self.n_regimes
        
    def _forward_backward(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm"""
        T = len(data)
        
        # Forward pass
        alpha = np.zeros((T, self.n_regimes))
        
        # Initial step
        for k in range(self.n_regimes):
            alpha[0, k] = self.initial_probs[k] * self._emission_probability(data[0], k)
            
        # Forward recursion
        for t in range(1, T):
            for j in range(self.n_regimes):
                alpha[t, j] = np.sum([
                    alpha[t-1, i] * self.transition_matrix[i, j] 
                    for i in range(self.n_regimes)
                ]) * self._emission_probability(data[t], j)
                
        # Backward pass
        beta = np.zeros((T, self.n_regimes))
        beta[T-1, :] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_regimes):
                beta[t, i] = np.sum([
                    self.transition_matrix[i, j] * 
                    self._emission_probability(data[t+1], j) * 
                    beta[t+1, j] 
                    for j in range(self.n_regimes)
                ])
                
        # Calculate log-likelihood
        log_likelihood = np.log(np.sum(alpha[T-1, :]))
        
        return alpha, beta, log_likelihood
        
    def _emission_probability(self, observation: np.ndarray, state: int) -> float:
        """Calculate emission probability for multivariate Gaussian"""
        mean = self.emission_params['means'][state]
        cov = self.emission_params['covariances'][state]
        
        try:
            return multivariate_normal.pdf(observation, mean, cov)
        except:
            # Fallback for numerical issues
            return 1e-10
            
    def _update_parameters(self, data: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        """Update HMM parameters (M-step)"""
        T = len(data)
        
        # Calculate gamma (state probabilities)
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)
        
        # Calculate xi (transition probabilities)
        xi = np.zeros((T-1, self.n_regimes, self.n_regimes))
        
        for t in range(T-1):
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    xi[t, i, j] = (
                        alpha[t, i] * 
                        self.transition_matrix[i, j] * 
                        self._emission_probability(data[t+1], j) * 
                        beta[t+1, j]
                    )
                    
            xi[t] = xi[t] / np.sum(xi[t])
            
        # Update initial probabilities
        self.initial_probs = gamma[0]
        
        # Update transition matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                self.transition_matrix[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
                
        # Update emission parameters
        for k in range(self.n_regimes):
            # Update means
            self.emission_params['means'][k] = (
                np.sum(gamma[:, k:k+1] * data, axis=0) / np.sum(gamma[:, k])
            )
            
            # Update covariances
            diff = data - self.emission_params['means'][k]
            weighted_diff = gamma[:, k:k+1] * diff
            self.emission_params['covariances'][k] = (
                np.dot(weighted_diff.T, diff) / np.sum(gamma[:, k])
            ) + np.eye(self.n_features) * 1e-6
            
    def predict(self, data: np.ndarray) -> RegimeDetectionResult:
        """Predict regimes using Viterbi algorithm"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        T = len(data)
        
        # Viterbi algorithm
        delta = np.zeros((T, self.n_regimes))
        psi = np.zeros((T, self.n_regimes), dtype=int)
        
        # Initialization
        for k in range(self.n_regimes):
            delta[0, k] = self.initial_probs[k] * self._emission_probability(data[0], k)
            
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_regimes):
                scores = delta[t-1] * self.transition_matrix[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = np.max(scores) * self._emission_probability(data[t], j)
                
        # Backward pass
        most_likely_regimes = np.zeros(T, dtype=int)
        most_likely_regimes[T-1] = np.argmax(delta[T-1])
        
        for t in range(T-2, -1, -1):
            most_likely_regimes[t] = psi[t+1, most_likely_regimes[t+1]]
            
        # Calculate state probabilities
        alpha, beta, log_likelihood = self._forward_backward(data)
        regime_probabilities = alpha * beta
        regime_probabilities = regime_probabilities / np.sum(regime_probabilities, axis=1, keepdims=True)
        
        # Calculate confidence scores
        confidence_scores = np.max(regime_probabilities, axis=1)
        
        return RegimeDetectionResult(
            regime_probabilities=regime_probabilities,
            most_likely_regimes=most_likely_regimes,
            regime_characteristics=self.get_regime_characteristics(),
            transition_matrix=self.transition_matrix,
            log_likelihood=log_likelihood,
            n_regimes=self.n_regimes,
            regime_names=self._get_regime_names(),
            confidence_scores=confidence_scores
        )
        
    def get_regime_characteristics(self) -> Dict[str, Any]:
        """Get regime characteristics"""
        if not self.is_fitted:
            return {}
            
        characteristics = {}
        regime_names = self._get_regime_names()
        
        for i, name in enumerate(regime_names):
            mean = self.emission_params['means'][i]
            cov = self.emission_params['covariances'][i]
            
            characteristics[name] = {
                'mean_return': float(mean[0]) if len(mean) > 0 else 0.0,
                'volatility': float(np.sqrt(cov[0, 0])) if cov.shape[0] > 0 else 0.0,
                'persistence': float(self.transition_matrix[i, i]),
                'mean_duration': 1.0 / (1.0 - self.transition_matrix[i, i]) if self.transition_matrix[i, i] < 1.0 else np.inf
            }
            
        return characteristics
        
    def _get_regime_names(self) -> List[str]:
        """Get regime names based on characteristics"""
        if not self.is_fitted:
            return [f"Regime_{i}" for i in range(self.n_regimes)]
            
        means = self.emission_params['means'][:, 0] if self.emission_params['means'].shape[1] > 0 else np.zeros(self.n_regimes)
        sorted_indices = np.argsort(means)
        
        names = []
        for i, idx in enumerate(sorted_indices):
            if i == 0:
                names.append("Bear Market")
            elif i == len(sorted_indices) - 1:
                names.append("Bull Market")
            else:
                names.append("Sideways Market")
                
        # Reorder to match original regime indices
        result = [""] * self.n_regimes
        for i, name in enumerate(names):
            result[sorted_indices[i]] = name
            
        return result

class NeuralRegimeDetector(BaseRegimeDetector):
    """
    Neural network-based regime detection using LSTM and attention mechanisms
    """
    
    def __init__(self, n_regimes: int = 3, sequence_length: int = 60, 
                 hidden_size: int = 128, n_layers: int = 2):
        super().__init__(n_regimes)
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, data: np.ndarray) -> 'NeuralRegimeDetector':
        """Fit neural regime detector"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.n_features = data.shape[1]
        
        # Create model
        self.model = RegimeDetectionLSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_regimes=self.n_regimes,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        # Prepare data
        X, y = self._prepare_training_data(data)
        
        # Train model
        self._train_model(X, y)
        
        self.is_fitted = True
        return self
        
    def _prepare_training_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for training"""
        # Use unsupervised pre-training with autoencoder-like objective
        X_sequences = []
        
        for i in range(len(data) - self.sequence_length):
            X_sequences.append(data[i:i + self.sequence_length])
            
        X = torch.FloatTensor(np.array(X_sequences)).to(self.device)
        
        # Create pseudo-labels using simple statistical regime detection
        returns = data[:, 0] if data.shape[1] > 0 else data.flatten()
        rolling_vol = pd.Series(returns).rolling(window=20).std().values
        rolling_mean = pd.Series(returns).rolling(window=20).mean().values
        
        # Simple regime labeling based on volatility and returns
        y_pseudo = np.zeros(len(data))
        
        for i in range(len(data)):
            if i < 20:  # Not enough history
                y_pseudo[i] = 1  # Default to middle regime
            else:
                vol = rolling_vol[i]
                ret = rolling_mean[i]
                
                vol_percentile = np.percentile(rolling_vol[20:i+1], 66)
                ret_percentile = np.percentile(rolling_mean[20:i+1], 33)
                
                if vol > vol_percentile:
                    y_pseudo[i] = 0  # High volatility (crisis/bear)
                elif ret > ret_percentile:
                    y_pseudo[i] = 2  # Bull market
                else:
                    y_pseudo[i] = 1  # Normal/sideways
                    
        # Extract labels for sequences
        y_sequences = y_pseudo[self.sequence_length:]
        y = torch.LongTensor(y_sequences).to(self.device)
        
        return X, y
        
    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100):
        """Train the neural model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Split data
        n_train = int(0.8 * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                
    def predict(self, data: np.ndarray) -> RegimeDetectionResult:
        """Predict regimes using neural model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        # Prepare sequences
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
            
        if len(sequences) == 0:
            # Handle case where data is too short
            sequences = [np.pad(data, ((self.sequence_length - len(data), 0), (0, 0)), mode='edge')]
            
        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            regime_probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            most_likely_regimes = np.argmax(regime_probabilities, axis=1)
            
        # Pad probabilities for sequences at the beginning
        if len(data) > self.sequence_length:
            padding = np.tile(regime_probabilities[0], (self.sequence_length - 1, 1))
            regime_probabilities = np.vstack([padding, regime_probabilities])
            most_likely_regimes = np.hstack([
                np.full(self.sequence_length - 1, most_likely_regimes[0]),
                most_likely_regimes
            ])
            
        confidence_scores = np.max(regime_probabilities, axis=1)
        
        return RegimeDetectionResult(
            regime_probabilities=regime_probabilities,
            most_likely_regimes=most_likely_regimes,
            regime_characteristics=self.get_regime_characteristics(),
            transition_matrix=self._estimate_transition_matrix(most_likely_regimes),
            log_likelihood=0.0,  # Not applicable for neural models
            n_regimes=self.n_regimes,
            regime_names=["Bear Market", "Sideways Market", "Bull Market"],
            confidence_scores=confidence_scores
        )
        
    def get_regime_characteristics(self) -> Dict[str, Any]:
        """Get regime characteristics from neural model"""
        # This is simplified - in practice would analyze learned representations
        return {
            "Bear Market": {"description": "High volatility, negative returns"},
            "Sideways Market": {"description": "Moderate volatility, neutral returns"},
            "Bull Market": {"description": "Lower volatility, positive returns"}
        }
        
    def _estimate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from regime sequence"""
        transition_counts = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regimes) - 1):
            transition_counts[regimes[i], regimes[i + 1]] += 1
            
        # Normalize
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_counts / row_sums
        
        return transition_matrix

class RegimeDetectionLSTM(nn.Module):
    """LSTM model for regime detection"""
    
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, 
                 n_regimes: int, sequence_length: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, n_regimes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep
        last_output = attended_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(last_output)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class EnsembleRegimeDetector(BaseRegimeDetector):
    """
    Ensemble regime detector combining multiple methods
    """
    
    def __init__(self, n_regimes: int = 3):
        super().__init__(n_regimes)
        
        # Initialize component detectors
        self.hmm_detector = HiddenMarkovRegimeDetector(n_regimes=n_regimes)
        self.neural_detector = NeuralRegimeDetector(n_regimes=n_regimes)
        
        # Weights for ensemble
        self.weights = np.array([0.6, 0.4])  # HMM, Neural
        
    def fit(self, data: np.ndarray) -> 'EnsembleRegimeDetector':
        """Fit all component detectors"""
        logger.info("Fitting HMM detector...")
        self.hmm_detector.fit(data)
        
        if len(data) >= 100:  # Neural detector needs more data
            logger.info("Fitting Neural detector...")
            self.neural_detector.fit(data)
        else:
            logger.warning("Insufficient data for Neural detector, using HMM only")
            self.weights = np.array([1.0, 0.0])
            
        self.is_fitted = True
        return self
        
    def predict(self, data: np.ndarray) -> RegimeDetectionResult:
        """Predict using ensemble of methods"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Get predictions from each detector
        hmm_result = self.hmm_detector.predict(data)
        
        if self.weights[1] > 0:
            neural_result = self.neural_detector.predict(data)
            
            # Combine probabilities
            ensemble_probs = (
                self.weights[0] * hmm_result.regime_probabilities +
                self.weights[1] * neural_result.regime_probabilities
            )
        else:
            ensemble_probs = hmm_result.regime_probabilities
            
        most_likely_regimes = np.argmax(ensemble_probs, axis=1)
        confidence_scores = np.max(ensemble_probs, axis=1)
        
        return RegimeDetectionResult(
            regime_probabilities=ensemble_probs,
            most_likely_regimes=most_likely_regimes,
            regime_characteristics=self.get_regime_characteristics(),
            transition_matrix=self._estimate_transition_matrix(most_likely_regimes),
            log_likelihood=hmm_result.log_likelihood,
            n_regimes=self.n_regimes,
            regime_names=hmm_result.regime_names,
            confidence_scores=confidence_scores
        )
        
    def get_regime_characteristics(self) -> Dict[str, Any]:
        """Get combined regime characteristics"""
        if not self.is_fitted:
            return {}
            
        hmm_chars = self.hmm_detector.get_regime_characteristics()
        
        if self.weights[1] > 0:
            neural_chars = self.neural_detector.get_regime_characteristics()
            # Combine characteristics (simplified)
            return hmm_chars
        else:
            return hmm_chars
            
    def _estimate_transition_matrix(self, regimes: np.ndarray) -> np.ndarray:
        """Estimate transition matrix from regime sequence"""
        return self.hmm_detector._estimate_transition_matrix(regimes)

class RegimeDetectionEngine:
    """
    Main engine for regime detection with multiple algorithms
    """
    
    def __init__(self):
        self.available_methods = {
            'hmm': HiddenMarkovRegimeDetector,
            'neural': NeuralRegimeDetector,
            'ensemble': EnsembleRegimeDetector
        }
        
    def detect_regimes(
        self, 
        data: np.ndarray, 
        method: str = 'ensemble',
        n_regimes: int = 3,
        **kwargs
    ) -> RegimeDetectionResult:
        """
        Detect market regimes using specified method
        
        Args:
            data: Time series data (returns, prices, etc.)
            method: Detection method ('hmm', 'neural', 'ensemble')
            n_regimes: Number of regimes to detect
            **kwargs: Additional parameters for specific methods
        """
        
        if method not in self.available_methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.available_methods.keys())}")
            
        logger.info(f"Starting regime detection with method: {method}")
        
        # Initialize detector
        detector_class = self.available_methods[method]
        detector = detector_class(n_regimes=n_regimes, **kwargs)
        
        # Fit and predict
        detector.fit(data)
        result = detector.predict(data)
        
        logger.info(f"Regime detection completed. Log-likelihood: {result.log_likelihood}")
        
        return result
        
    def compare_methods(
        self, 
        data: np.ndarray, 
        n_regimes: int = 3
    ) -> Dict[str, RegimeDetectionResult]:
        """Compare different regime detection methods"""
        
        results = {}
        
        for method_name in self.available_methods.keys():
            try:
                logger.info(f"Testing method: {method_name}")
                result = self.detect_regimes(data, method=method_name, n_regimes=n_regimes)
                results[method_name] = result
            except Exception as e:
                logger.error(f"Method {method_name} failed: {e}")
                
        return results
