"""
Hidden Markov Model implementation for regime detection
"""

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
import logging
from typing import Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class HiddenMarkovModel:
    """Hidden Markov Model for financial regime detection"""
    
    def __init__(self, n_states: int, n_observations: int):
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Model parameters
        self.initial_probabilities = None
        self.transition_matrix = None
        self.emission_matrix = None
        
    def fit(self, observations: np.ndarray, max_iterations: int = 100, 
            tolerance: float = 1e-6) -> float:
        """
        Fit HMM using Baum-Welch (EM) algorithm
        
        Returns final log-likelihood
        """
        logger.info(f"Fitting HMM with {self.n_states} states")
        
        # Initialize parameters
        self._initialize_parameters(observations)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Forward-backward algorithm
            log_likelihood, alpha, beta = self._forward_backward(observations)
            
            # M-step: Update parameters
            self._update_parameters(observations, alpha, beta)
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = log_likelihood
            
        self.log_likelihood = log_likelihood
        return log_likelihood
        
    def _initialize_parameters(self, observations: np.ndarray):
        """Initialize model parameters"""
        
        # Initialize transition matrix (slightly favor staying in same state)
        self.transition_matrix = np.random.dirichlet(
            np.ones(self.n_states) * 0.1 + np.eye(self.n_states) * 0.9,
            size=self.n_states
        )
        
        # Initialize emission matrix
        self.emission_matrix = np.random.dirichlet(
            np.ones(self.n_observations),
            size=self.n_states
        )
        
        # Initialize state probabilities uniformly
        self.initial_probabilities = np.ones(self.n_states) / self.n_states
        
    def _forward_backward(self, observations: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Forward-backward algorithm"""
        
        T = len(observations)
        
        # Forward pass
        log_alpha = np.zeros((T, self.n_states))
        log_alpha[0] = np.log(self.initial_probabilities) + \
                      np.log(self.emission_matrix[:, observations])
        
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.transition_matrix[:, j])
                ) + np.log(self.emission_matrix[j, observations[t]])
                
        # Backward pass
        log_beta = np.zeros((T, self.n_states))
        # log_beta[T-1] = 0 (already initialized)
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transition_matrix[i]) +
                    np.log(self.emission_matrix[:, observations[t+1]]) +
                    log_beta[t+1]
                )
                
        # Calculate log-likelihood
        log_likelihood = logsumexp(log_alpha[T-1])
        
        return log_likelihood, log_alpha, log_beta
        
    def _update_parameters(self, observations: np.ndarray, 
                          log_alpha: np.ndarray, log_beta: np.ndarray):
        """Update model parameters (M-step)"""
        
        T = len(observations)
        
        # Calculate gamma (state probabilities)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        
        # Calculate xi (transition probabilities)
        log_xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi[t, i, j] = (
                        log_alpha[t, i] +
                        np.log(self.transition_matrix[i, j]) +
                        np.log(self.emission_matrix[j, observations[t+1]]) +
                        log_beta[t+1, j]
                    )
                    
        # Normalize xi
        log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
        xi = np.exp(log_xi)
        
        # Update initial probabilities
        self.initial_probabilities = gamma[0]
        
        # Update transition matrix
        self.transition_matrix = np.sum(xi, axis=0)
        self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)
        
        # Update emission matrix
        for j in range(self.n_states):
            for k in range(self.n_observations):
                mask = (observations == k)
                self.emission_matrix[j, k] = np.sum(gamma[mask, j])
                
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1, keepdims=True)
        
    def viterbi_decode(self, observations: np.ndarray) -> np.ndarray:
        """Find most likely state sequence using Viterbi algorithm"""
        
        T = len(observations)
        
        # Initialize
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        log_delta[0] = np.log(self.initial_probabilities) + \
                      np.log(self.emission_matrix[:, observations])
        
        # Forward pass
        for t in range(1, T):
            for j in range(self.n_states):
                scores = log_delta[t-1] + np.log(self.transition_matrix[:, j])
                psi[t, j] = np.argmax(scores)
                log_delta[t, j] = np.max(scores) + \
                                np.log(self.emission_matrix[j, observations[t]])
                                
        # Backward pass
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(log_delta[T-1])
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
            
        return states
        
    def predict_states(self, observations: np.ndarray) -> np.ndarray:
        """Predict state probabilities for each time step"""
        log_likelihood, log_alpha, log_beta = self._forward_backward(observations)
        
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        
        return np.exp(log_gamma)
