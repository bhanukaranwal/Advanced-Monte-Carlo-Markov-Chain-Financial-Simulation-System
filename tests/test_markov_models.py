"""
Tests for Markov models functionality
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from markov_models.hidden_markov import HiddenMarkovModel
from markov_models.regime_switching import RegimeSwitchingModel
from markov_models.transition_matrices import TransitionMatrixEstimator

class TestHiddenMarkovModel:
    """Test Hidden Markov Model functionality"""
    
    def test_initialization(self):
        """Test HMM initialization"""
        hmm = HiddenMarkovModel(n_states=3, n_observations=2)
        
        assert hmm.n_states == 3
        assert hmm.n_observations == 2
        assert hmm.transition_matrix.shape == (3, 3)
        assert hmm.emission_matrix.shape == (3, 2)
        
    def test_forward_algorithm(self):
        """Test forward algorithm"""
        hmm = HiddenMarkovModel(n_states=2, n_observations=2)
        
        # Set known parameters
        hmm.transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        hmm.emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        hmm.initial_probabilities = np.array([0.6, 0.4])
        
        # Test sequence
        observations = np.array([0, 1, 0])
        
        log_prob, forward_probs = hmm.forward_algorithm(observations)
        
        assert forward_probs.shape == (3, 2)  # (T, N)
        assert not np.isnan(log_prob)
        assert np.all(forward_probs >= 0)
        
    def test_viterbi_algorithm(self):
        """Test Viterbi algorithm"""
        hmm = HiddenMarkovModel(n_states=2, n_observations=2)
        
        # Set parameters
        hmm.transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        hmm.emission_matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
        hmm.initial_probabilities = np.array([0.6, 0.4])
        
        observations = np.array([0, 1, 0, 1])
        
        most_likely_path = hmm.viterbi_decode(observations)
        
        assert len(most_likely_path) == len(observations)
        assert all(0 <= state < 2 for state in most_likely_path)
        
    def test_parameter_estimation(self, sample_returns_data):
        """Test parameter estimation with Baum-Welch"""
        # Convert returns to discrete observations
        returns = sample_returns_data.values
        
        # Simple discretization: 0 = negative, 1 = positive
        observations = (returns > 0).astype(int)
        
        hmm = HiddenMarkovModel(n_states=2, n_observations=2)
        hmm.fit(observations, max_iterations=10, tolerance=1e-4)
        
        # Check if parameters are valid probability matrices
        assert np.allclose(hmm.transition_matrix.sum(axis=1), 1.0)
        assert np.allclose(hmm.emission_matrix.sum(axis=1), 1.0)
        assert np.allclose(hmm.initial_probabilities.sum(), 1.0)
        
    def test_state_prediction(self):
        """Test state prediction"""
        hmm = HiddenMarkovModel(n_states=2, n_observations=2)
        
        # Set parameters
        hmm.transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
        hmm.emission_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
        hmm.initial_probabilities = np.array([0.5, 0.5])
        hmm.fitted = True
        
        observations = np.array([0, 1, 0])
        
        # Get state probabilities
        state_probs = hmm.predict_states(observations)
        
        assert state_probs.shape == (3, 2)
        assert np.allclose(state_probs.sum(axis=1), 1.0)

class TestRegimeSwitchingModel:
    """Test regime-switching model functionality"""
    
    def test_initialization(self, sample_returns_data):
        """Test regime-switching model initialization"""
        model = RegimeSwitchingModel(
            data=sample_returns_data,
            n_regimes=2,
            model_type='mean_reverting'
        )
        
        assert model.n_regimes == 2
        assert model.model_type == 'mean_reverting'
        assert len(model.data) == len(sample_returns_data)
        
    def test_regime_detection(self, sample_returns_data):
        """Test regime detection"""
        model = RegimeSwitchingModel(
            data=sample_returns_data,
            n_regimes=2,
            model_type='variance_switching'
        )
        
        # Fit model
        model.fit(max_iterations=5)  # Reduced for testing
        
        # Check if regimes are detected
        regimes = model.get_regime_sequence()
        
        assert len(regimes) == len(sample_returns_data)
        assert all(0 <= regime < 2 for regime in regimes)
        
        # Check regime probabilities
        regime_probs = model.get_regime_probabilities()
        
        assert regime_probs.shape == (len(sample_returns_data), 2)
        assert np.allclose(regime_probs.sum(axis=1), 1.0)
        
    def test_regime_characteristics(self, sample_returns_data):
        """Test regime characteristics extraction"""
        model = RegimeSwitchingModel(
            data=sample_returns_data,
            n_regimes=2,
            model_type='mean_reverting'
        )
        
        model.fit(max_iterations=5)
        
        characteristics = model.get_regime_characteristics()
        
        assert len(characteristics) == 2
        
        for regime in characteristics:
            assert 'mean' in regime
            assert 'volatility' in regime
            assert 'persistence' in regime
            
    def test_regime_forecasting(self, sample_returns_data):
        """Test regime forecasting"""
        model = RegimeSwitchingModel(
            data=sample_returns_data,
            n_regimes=2,
            model_type='variance_switching'
        )
        
        model.fit(max_iterations=5)
        
        # Forecast next regime probabilities
        forecast = model.forecast_regime_probabilities(steps=5)
        
        assert forecast.shape == (5, 2)
        assert np.allclose(forecast.sum(axis=1), 1.0)

class TestTransitionMatrixEstimator:
    """Test transition matrix estimation"""
    
    def test_mle_estimation(self):
        """Test MLE estimation of transition matrix"""
        # Create sample state sequence
        np.random.seed(42)
        states = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
        
        estimator = TransitionMatrixEstimator()
        transition_matrix = estimator.estimate_mle(states, n_states=2)
        
        assert transition_matrix.shape == (2, 2)
        assert np.allclose(transition_matrix.sum(axis=1), 1.0)
        assert np.all(transition_matrix >= 0)
        
    def test_bayesian_estimation(self):
        """Test Bayesian estimation with Dirichlet prior"""
        states = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1]
        
        estimator = TransitionMatrixEstimator()
        
        # Prior: slightly favor diagonal elements
        prior = np.array([[2, 1], [1, 2]])
        
        transition_matrix = estimator.estimate_bayesian(
            states, n_states=2, prior_alpha=prior
        )
        
        assert transition_matrix.shape == (2, 2)
        assert np.allclose(transition_matrix.sum(axis=1), 1.0)
        assert np.all(transition_matrix >= 0)
        
    def test_time_varying_estimation(self):
        """Test time-varying transition matrix estimation"""
        # Create longer sequence
        np.random.seed(42)
        states = np.random.choice([0, 1], size=100)
        
        estimator = TransitionMatrixEstimator()
        
        transition_matrices = estimator.estimate_time_varying(
            states, n_states=2, window_size=20
        )
        
        assert len(transition_matrices) == 100 - 20 + 1  # Number of windows
        
        for tm in transition_matrices:
            assert tm.shape == (2, 2)
            assert np.allclose(tm.sum(axis=1), 1.0)
            
    def test_confidence_intervals(self):
        """Test confidence intervals for transition probabilities"""
        states = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1] * 10  # Repeat for more data
        
        estimator = TransitionMatrixEstimator()
        
        transition_matrix, confidence_intervals = estimator.estimate_with_confidence(
            states, n_states=2, confidence_level=0.95
        )
        
        assert transition_matrix.shape == (2, 2)
        assert confidence_intervals.shape == (2, 2, 2)  # (from, to, [lower, upper])
        
        # Check that confidence intervals contain the point estimates
        for i in range(2):
            for j in range(2):
                lower, upper = confidence_intervals[i, j]
                assert lower <= transition_matrix[i, j] <= upper

if __name__ == "__main__":
    pytest.main([__file__])
