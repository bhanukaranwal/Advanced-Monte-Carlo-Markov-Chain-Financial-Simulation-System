"""
Regime detection algorithms for financial time series
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

@dataclass
class RegimeState:
    """Individual regime state characteristics"""
    regime_id: int
    mean_return: float
    volatility: float
    persistence: float
    frequency: float
    start_periods: List[int]
    end_periods: List[int]
    
@dataclass
class RegimeDetectionResult:
    """Results from regime detection"""
    n_regimes: int
    regime_sequence: List[int]
    regime_probabilities: np.ndarray
    regime_states: List[RegimeState]
    model_likelihood: float
    aic: float
    bic: float
    most_likely_regime: int
    regime_transition_matrix: np.ndarray

class BaseRegimeDetector:
    """Base class for regime detection algorithms"""
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.fitted = False
        
    def fit(self, data: Union[pd.Series, np.ndarray]) -> 'BaseRegimeDetector':
        """Fit regime detection model"""
        raise NotImplementedError
        
    def predict(self, data: Union[pd.Series, np.ndarray]) -> RegimeDetectionResult:
        """Predict regimes for new data"""
        raise NotImplementedError
        
    def get_regime_characteristics(self) -> List[RegimeState]:
        """Get characteristics of detected regimes"""
        raise NotImplementedError

class HiddenMarkovRegime(BaseRegimeDetector):
    """Hidden Markov Model for regime detection"""
    
    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = 'full',
        max_iter: int = 100,
        random_state: int = 42
    ):
        super().__init__(n_regimes)
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state
        
        # HMM components
        self.hmm_model = None
        self.transition_matrix = None
        self.emission_means = None
        self.emission_covars = None
        self.initial_probabilities = None
        
        # Results storage
        self.log_likelihood = None
        self.regime_sequence = None
        self.regime_probabilities = None
        
    def fit(self, data: Union[pd.Series, np.ndarray]) -> 'HiddenMarkovRegime':
        """
        Fit HMM regime detection model
        
        Args:
            data: Time series data (returns or prices)
            
        Returns:
            Fitted model
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        # Convert to returns if data looks like prices
        if np.all(data > 0) and np.mean(data) > 1:
            logger.info("Converting prices to returns")
            returns = np.diff(np.log(data))
        else:
            returns = data[~np.isnan(data)]
            
        # Reshape for HMM (needs 2D array)
        X = returns.reshape(-1, 1)
        
        # Initialize and fit HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.max_iter,
            random_state=self.random_state,
            verbose=False
        )
        
        try:
            self.hmm_model.fit(X)
            
            # Extract model parameters
            self.transition_matrix = self.hmm_model.transmat_
            self.emission_means = self.hmm_model.means_.flatten()
            self.emission_covars = self.hmm_model.covars_
            self.initial_probabilities = self.hmm_model.startprob_
            
            # Calculate log-likelihood
            self.log_likelihood = self.hmm_model.score(X)
            
            # Decode most likely state sequence
            self.regime_sequence = self.hmm_model.predict(X)
            
            # Get state probabilities
            self.regime_probabilities = self.hmm_model.predict_proba(X)
            
            self.fitted = True
            logger.info(f"HMM fitted successfully with {self.n_regimes} regimes")
            
        except Exception as e:
            logger.error(f"Failed to fit HMM: {e}")
            raise
            
        return self
        
    def predict(self, data: Union[pd.Series, np.ndarray]) -> RegimeDetectionResult:
        """Predict regimes for new data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(data, pd.Series):
            data = data.values
            
        # Convert to returns if needed
        if np.all(data > 0) and np.mean(data) > 1:
            returns = np.diff(np.log(data))
        else:
            returns = data[~np.isnan(data)]
            
        X = returns.reshape(-1, 1)
        
        # Predict regimes
        regime_sequence = self.hmm_model.predict(X)
        regime_probabilities = self.hmm_model.predict_proba(X)
        
        # Get regime characteristics
        regime_states = self.get_regime_characteristics()
        
        # Calculate information criteria
        n_params = (
            self.n_regimes * (self.n_regimes - 1) +  # Transition matrix
            self.n_regimes * 1 +  # Means
            self.n_regimes * 1 +  # Variances
            self.n_regimes - 1    # Initial probabilities
        )
        
        aic = -2 * self.log_likelihood + 2 * n_params
        bic = -2 * self.log_likelihood + np.log(len(returns)) * n_params
        
        # Most likely current regime
        most_likely_regime = regime_sequence[-1] if len(regime_sequence) > 0 else 0
        
        return RegimeDetectionResult(
            n_regimes=self.n_regimes,
            regime_sequence=regime_sequence.tolist(),
            regime_probabilities=regime_probabilities,
            regime_states=regime_states,
            model_likelihood=self.log_likelihood,
            aic=aic,
            bic=bic,
            most_likely_regime=most_likely_regime,
            regime_transition_matrix=self.transition_matrix
        )
        
    def get_regime_characteristics(self) -> List[RegimeState]:
        """Extract regime characteristics"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        regime_states = []
        
        for i in range(self.n_regimes):
            # Mean return and volatility
            mean_return = self.emission_means[i]
            
            if self.covariance_type == 'full':
                volatility = np.sqrt(self.emission_covars[i, 0, 0])
            else:
                volatility = np.sqrt(self.emission_covars[i, 0])
                
            # Persistence (probability of staying in same regime)
            persistence = self.transition_matrix[i, i]
            
            # Frequency (stationary probability)
            try:
                eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
                stationary_idx = np.argmin(np.abs(eigenvals - 1.0))
                stationary_dist = np.real(eigenvecs[:, stationary_idx])
                stationary_dist = np.abs(stationary_dist) / np.sum(np.abs(stationary_dist))
                frequency = stationary_dist[i]
            except:
                frequency = 1.0 / self.n_regimes
                
            # Find regime periods
            if self.regime_sequence is not None:
                regime_periods = np.where(np.array(self.regime_sequence) == i)[0]
                
                # Find contiguous periods
                start_periods = []
                end_periods = []
                
                if len(regime_periods) > 0:
                    current_start = regime_periods[0]
                    
                    for j in range(1, len(regime_periods)):
                        if regime_periods[j] - regime_periods[j-1] > 1:
                            # End of current period
                            end_periods.append(regime_periods[j-1])
                            start_periods.append(current_start)
                            current_start = regime_periods[j]
                            
                    # Don't forget last period
                    end_periods.append(regime_periods[-1])
                    start_periods.append(current_start)
            else:
                start_periods = []
                end_periods = []
                
            regime_states.append(RegimeState(
                regime_id=i,
                mean_return=mean_return,
                volatility=volatility,
                persistence=persistence,
                frequency=frequency,
                start_periods=start_periods,
                end_periods=end_periods
            ))
            
        return regime_states
        
    def forecast_regime_probabilities(self, steps_ahead: int = 1) -> np.ndarray:
        """Forecast regime probabilities"""
        if not self.fitted or self.regime_probabilities is None:
            raise ValueError("Model must be fitted first")
            
        # Current regime probabilities
        current_probs = self.regime_probabilities[-1]
        
        # Evolve probabilities forward
        forecast_probs = current_probs.copy()
        for _ in range(steps_ahead):
            forecast_probs = forecast_probs @ self.transition_matrix
            
        return forecast_probs

class ThresholdRegime(BaseRegimeDetector):
    """Threshold-based regime detection (Threshold Autoregressive model)"""
    
    def __init__(
        self,
        n_regimes: int = 2,
        threshold_variable: str = 'level',  # 'level', 'change', 'volatility'
        delay: int = 1,
        search_method: str = 'grid'  # 'grid', 'optimization'
    ):
        super().__init__(n_regimes)
        self.threshold_variable = threshold_variable
        self.delay = delay
        self.search_method = search_method
        
        # Model parameters
        self.thresholds = None
        self.regime_parameters = None
        self.threshold_series = None
        
    def fit(self, data: Union[pd.Series, np.ndarray]) -> 'ThresholdRegime':
        """
        Fit threshold regime model
        
        Args:
            data: Time series data
            
        Returns:
            Fitted model
        """
        if isinstance(data, pd.Series):
            data = data.values
            
        # Convert to returns if needed
        if np.all(data > 0) and np.mean(data) > 1:
            returns = np.diff(np.log(data))
            levels = data[1:]  # Corresponding price levels
        else:
            returns = data[~np.isnan(data)]
            levels = np.cumsum(returns)  # Cumulative returns as proxy for levels
            
        # Create threshold variable
        if self.threshold_variable == 'level':
            self.threshold_series = levels
        elif self.threshold_variable == 'change':
            self.threshold_series = returns
        elif self.threshold_variable == 'volatility':
            # Rolling volatility
            window = 20
            vol_series = []
            for i in range(window, len(returns)):
                vol = np.std(returns[i-window:i])
                vol_series.append(vol)
            self.threshold_series = np.array(vol_series)
            returns = returns[window:]  # Adjust returns to match
        else:
            raise ValueError(f"Unknown threshold variable: {self.threshold_variable}")
            
        # Apply delay
        if self.delay > 0:
            threshold_delayed = self.threshold_series[:-self.delay]
            returns_delayed = returns[self.delay:]
        else:
            threshold_delayed = self.threshold_series
            returns_delayed = returns
            
        # Find optimal thresholds
        if self.search_method == 'grid':
            self.thresholds = self._grid_search_thresholds(threshold_delayed, returns_delayed)
        else:
            self.thresholds = self._optimize_thresholds(threshold_delayed, returns_delayed)
            
        # Estimate regime parameters
        self.regime_parameters = self._estimate_regime_parameters(
            threshold_delayed, returns_delayed
        )
        
        self.fitted = True
        logger.info(f"Threshold model fitted with thresholds: {self.thresholds}")
        
        return self
        
    def _grid_search_thresholds(
        self, 
        threshold_series: np.ndarray, 
        returns: np.ndarray
    ) -> List[float]:
        """Find optimal thresholds using grid search"""
        if self.n_regimes == 2:
            # Single threshold for 2 regimes
            percentiles = np.linspace(10, 90, 81)
            threshold_candidates = np.percentile(threshold_series, percentiles)
            
            best_threshold = None
            best_likelihood = -np.inf
            
            for threshold in threshold_candidates:
                try:
                    # Split data into regimes
                    regime_0_mask = threshold_series <= threshold
                    regime_1_mask = threshold_series > threshold
                    
                    if np.sum(regime_0_mask) < 10 or np.sum(regime_1_mask) < 10:
                        continue  # Need minimum observations per regime
                        
                    # Calculate likelihood
                    ll = 0.0
                    
                    # Regime 0
                    returns_0 = returns[regime_0_mask]
                    if len(returns_0) > 0:
                        mu_0 = np.mean(returns_0)
                        sigma_0 = np.std(returns_0)
                        if sigma_0 > 0:
                            ll += np.sum(stats.norm.logpdf(returns_0, mu_0, sigma_0))
                            
                    # Regime 1
                    returns_1 = returns[regime_1_mask]
                    if len(returns_1) > 0:
                        mu_1 = np.mean(returns_1)
                        sigma_1 = np.std(returns_1)
                        if sigma_1 > 0:
                            ll += np.sum(stats.norm.logpdf(returns_1, mu_1, sigma_1))
                            
                    if ll > best_likelihood:
                        best_likelihood = ll
                        best_threshold = threshold
                        
                except:
                    continue
                    
            return [best_threshold] if best_threshold is not None else [np.median(threshold_series)]
            
        else:
            # Multiple thresholds for more regimes
            # Use quantiles as starting point
            quantiles = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
            return np.percentile(threshold_series, quantiles * 100).tolist()
            
    def _optimize_thresholds(
        self, 
        threshold_series: np.ndarray, 
        returns: np.ndarray
    ) -> List[float]:
        """Optimize thresholds using numerical optimization"""
        # Simplified optimization for 2 regimes
        if self.n_regimes != 2:
            return self._grid_search_thresholds(threshold_series, returns)
            
        def negative_log_likelihood(params):
            threshold = params[0]
            
            try:
                regime_0_mask = threshold_series <= threshold
                regime_1_mask = threshold_series > threshold
                
                if np.sum(regime_0_mask) < 5 or np.sum(regime_1_mask) < 5:
                    return 1e10
                    
                ll = 0.0
                
                # Regime 0
                returns_0 = returns[regime_0_mask]
                mu_0 = np.mean(returns_0)
                sigma_0 = np.std(returns_0)
                if sigma_0 > 0:
                    ll += np.sum(stats.norm.logpdf(returns_0, mu_0, sigma_0))
                    
                # Regime 1  
                returns_1 = returns[regime_1_mask]
                mu_1 = np.mean(returns_1)
                sigma_1 = np.std(returns_1)
                if sigma_1 > 0:
                    ll += np.sum(stats.norm.logpdf(returns_1, mu_1, sigma_1))
                    
                return -ll
                
            except:
                return 1e10
                
        # Optimize
        bounds = [(np.percentile(threshold_series, 5), np.percentile(threshold_series, 95))]
        initial_guess = [np.median(threshold_series)]
        
        result = minimize(
            negative_log_likelihood,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        if result.success:
            return [result.x[0]]
        else:
            return [np.median(threshold_series)]
            
    def _estimate_regime_parameters(
        self, 
        threshold_series: np.ndarray, 
        returns: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """Estimate parameters for each regime"""
        parameters = {}
        
        for i in range(self.n_regimes):
            if i == 0:
                # First regime: below first threshold
                mask = threshold_series <= self.thresholds[0]
            elif i == self.n_regimes - 1:
                # Last regime: above last threshold
                mask = threshold_series > self.thresholds[-1]
            else:
                # Middle regimes: between thresholds
                mask = ((threshold_series > self.thresholds[i-1]) & 
                       (threshold_series <= self.thresholds[i]))
                       
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                parameters[i] = {
                    'mean': np.mean(regime_returns),
                    'std': np.std(regime_returns),
                    'count': len(regime_returns)
                }
            else:
                parameters[i] = {
                    'mean': 0.0,
                    'std': 0.01,
                    'count': 0
                }
                
        return parameters
        
    def predict(self, data: Union[pd.Series, np.ndarray]) -> RegimeDetectionResult:
        """Predict regimes for new data"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if isinstance(data, pd.Series):
            data = data.values
            
        # Convert to returns and create threshold variable
        if np.all(data > 0) and np.mean(data) > 1:
            returns = np.diff(np.log(data))
            levels = data[1:]
        else:
            returns = data[~np.isnan(data)]
            levels = np.cumsum(returns)
            
        if self.threshold_variable == 'level':
            threshold_series = levels
        elif self.threshold_variable == 'change':
            threshold_series = returns
        elif self.threshold_variable == 'volatility':
            window = 20
            vol_series = []
            for i in range(window, len(returns)):
                vol = np.std(returns[i-window:i])
                vol_series.append(vol)
            threshold_series = np.array(vol_series)
            returns = returns[window:]
            
        # Apply delay
        if self.delay > 0 and len(threshold_series) > self.delay:
            threshold_delayed = threshold_series[:-self.delay]
            returns_delayed = returns[self.delay:]
        else:
            threshold_delayed = threshold_series
            returns_delayed = returns
            
        # Classify into regimes
        regime_sequence = []
        for threshold_val in threshold_delayed:
            regime = 0
            for i, threshold in enumerate(self.thresholds):
                if threshold_val > threshold:
                    regime = i + 1
            regime_sequence.append(regime)
            
        # Create regime probabilities (deterministic for threshold model)
        n_obs = len(regime_sequence)
        regime_probabilities = np.zeros((n_obs, self.n_regimes))
        for i, regime in enumerate(regime_sequence):
            regime_probabilities[i, regime] = 1.0
            
        # Get regime characteristics
        regime_states = self.get_regime_characteristics()
        
        # Calculate pseudo-likelihood
        log_likelihood = 0.0
        for i, regime in enumerate(regime_sequence):
            if i < len(returns_delayed):
                params = self.regime_parameters[regime]
                if params['std'] > 0:
                    log_likelihood += stats.norm.logpdf(
                        returns_delayed[i], params['mean'], params['std']
                    )
                    
        # Information criteria (simplified)
        n_params = self.n_regimes * 2 + len(self.thresholds)  # means, stds, thresholds
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(len(returns_delayed)) * n_params
        
        # Create transition matrix (empirical)
        transition_matrix = self._estimate_transition_matrix(regime_sequence)
        
        return RegimeDetectionResult(
            n_regimes=self.n_regimes,
            regime_sequence=regime_sequence,
            regime_probabilities=regime_probabilities,
            regime_states=regime_states,
            model_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            most_likely_regime=regime_sequence[-1] if regime_sequence else 0,
            regime_transition_matrix=transition_matrix
        )
        
    def _estimate_transition_matrix(self, regime_sequence: List[int]) -> np.ndarray:
        """Estimate empirical transition matrix"""
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regime_sequence) - 1):
            current_regime = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            transition_matrix[current_regime, next_regime] += 1
            
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1)
        for i in range(self.n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i] /= row_sums[i]
            else:
                transition_matrix[i, i] = 1.0  # Stay in same regime if no transitions observed
                
        return transition_matrix
        
    def get_regime_characteristics(self) -> List[RegimeState]:
        """Get regime characteristics for threshold model"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        regime_states = []
        
        for i in range(self.n_regimes):
            params = self.regime_parameters[i]
            
            # Persistence calculation (simplified)
            # For threshold models, persistence depends on threshold variable dynamics
            persistence = 0.9  # Default value
            
            # Frequency based on parameter estimation
            total_obs = sum(params['count'] for params in self.regime_parameters.values())
            frequency = params['count'] / total_obs if total_obs > 0 else 0.0
            
            regime_states.append(RegimeState(
                regime_id=i,
                mean_return=params['mean'],
                volatility=params['std'],
                persistence=persistence,
                frequency=frequency,
                start_periods=[],  # Would need to track these during fitting
                end_periods=[]
            ))
            
        return regime_states

class StructuralBreakRegime(BaseRegimeDetector):
    """Structural break detection using CUSUM and other tests"""
    
    def __init__(self, min_regime_length: int = 20, significance_level: float = 0.05):
        super().__init__(n_regimes=2)  # Will be determined endogenously
        self.min_regime_length = min_regime_length
        self.significance_level = significance_level
        self.break_points = None
        
    def fit(self, data: Union[pd.Series, np.ndarray]) -> 'StructuralBreakRegime':
        """Detect structural breaks using CUSUM test"""
        if isinstance(data, pd.Series):
            data = data.values
            
        # Convert to returns if needed
        if np.all(data > 0) and np.mean(data) > 1:
            returns = np.diff(np.log(data))
        else:
            returns = data[~np.isnan(data)]
            
        # CUSUM test for structural breaks
        self.break_points = self._cusum_test(returns)
        self.n_regimes = len(self.break_points) + 1
        
        self.fitted = True
        logger.info(f"Detected {len(self.break_points)} structural breaks")
        
        return self
        
    def _cusum_test(self, returns: np.ndarray) -> List[int]:
        """CUSUM test for structural breaks"""
        n = len(returns)
        break_points = []
        
        # Recursive implementation
        self._recursive_cusum(returns, 0, n-1, break_points)
        
        return sorted(break_points)
        
    def _recursive_cusum(
        self, 
        returns: np.ndarray, 
        start: int, 
        end: int, 
        break_points: List[int]
    ):
        """Recursive CUSUM test"""
        if end - start < 2 * self.min_regime_length:
            return
            
        segment = returns[start:end+1]
        n = len(segment)
        
        # Calculate CUSUM statistic
        mean_segment = np.mean(segment)
        cumsum = np.cumsum(segment - mean_segment)
        
        # Test statistic (simplified)
        test_stats = []
        for k in range(self.min_regime_length, n - self.min_regime_length):
            # Normalized CUSUM
            s_k = cumsum[k] / np.sqrt(n * np.var(segment))
            test_stats.append(abs(s_k))
            
        if not test_stats:
            return
            
        max_stat = max(test_stats)
        max_idx = test_stats.index(max_stat) + self.min_regime_length
        
        # Critical value (simplified)
        critical_value = 1.36  # Approximate 5% critical value
        
        if max_stat > critical_value:
            break_point = start + max_idx
            break_points.append(break_point)
            
            # Recursively test sub-segments
            self._recursive_cusum(returns, start, break_point, break_points)
            self._recursive_cusum(returns, break_point, end, break_points)

class RegimeDetection:
    """Main regime detection framework"""
    
    def __init__(self):
        self.fitted_models = {}
        self.model_comparison = {}
        
    def fit_model(
        self,
        data: Union[pd.Series, np.ndarray],
        model_type: str,
        model_name: str = None,
        **kwargs
    ) -> BaseRegimeDetector:
        """
        Fit regime detection model
        
        Args:
            data: Time series data
            model_type: Type of model ('hmm', 'threshold', 'structural_break')
            model_name: Optional name for the model
            **kwargs: Model-specific parameters
            
        Returns:
            Fitted model
        """
        if model_type == 'hmm':
            model = HiddenMarkovRegime(**kwargs)
        elif model_type == 'threshold':
            model = ThresholdRegime(**kwargs)
        elif model_type == 'structural_break':
            model = StructuralBreakRegime(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        fitted_model = model.fit(data)
        
        # Store fitted model
        name = model_name or f"{model_type}_{len(self.fitted_models)}"
        self.fitted_models[name] = fitted_model
        
        return fitted_model
        
    def compare_models(
        self,
        data: Union[pd.Series, np.ndarray],
        model_types: List[str] = ['hmm', 'threshold'],
        n_regimes_range: Tuple[int, int] = (2, 4)
    ) -> pd.DataFrame:
        """
        Compare multiple regime detection models
        
        Args:
            data: Time series data
            model_types: List of model types to compare
            n_regimes_range: Range of number of regimes to test
            
        Returns:
            Comparison results DataFrame
        """
        results = []
        
        for model_type in model_types:
            if model_type == 'structural_break':
                # Structural break determines regimes endogenously
                try:
                    model = self.fit_model(data, model_type, f"{model_type}_auto")
                    result = model.predict(data)
                    
                    results.append({
                        'model_type': model_type,
                        'n_regimes': result.n_regimes,
                        'log_likelihood': result.model_likelihood,
                        'aic': result.aic,
                        'bic': result.bic,
                        'model_name': f"{model_type}_auto"
                    })
                except Exception as e:
                    logger.warning(f"Failed to fit {model_type}: {e}")
            else:
                # Test different numbers of regimes
                for n_regimes in range(n_regimes_range[0], n_regimes_range[1] + 1):
                    try:
                        model = self.fit_model(
                            data, model_type, f"{model_type}_{n_regimes}",
                            n_regimes=n_regimes
                        )
                        result = model.predict(data)
                        
                        results.append({
                            'model_type': model_type,
                            'n_regimes': n_regimes,
                            'log_likelihood': result.model_likelihood,
                            'aic': result.aic,
                            'bic': result.bic,
                            'model_name': f"{model_type}_{n_regimes}"
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to fit {model_type} with {n_regimes} regimes: {e}")
                        
        comparison_df = pd.DataFrame(results)
        
        if not comparison_df.empty:
            # Rank models
            comparison_df['aic_rank'] = comparison_df['aic'].rank()
            comparison_df['bic_rank'] = comparison_df['bic'].rank()
            comparison_df = comparison_df.sort_values('bic')
            
        self.model_comparison = comparison_df
        return comparison_df
        
    def get_best_model(self, criterion: str = 'bic') -> Optional[BaseRegimeDetector]:
        """Get best model based on information criterion"""
        if self.model_comparison.empty:
            return None
            
        if criterion not in ['aic', 'bic']:
            raise ValueError("Criterion must be 'aic' or 'bic'")
            
        best_model_name = self.model_comparison.loc[
            self.model_comparison[criterion].idxmin(), 'model_name'
        ]
        
        return self.fitted_models.get(best_model_name)
        
    def detect_regime_changes(
        self,
        data: Union[pd.Series, np.ndarray],
        model_name: str
    ) -> List[Tuple[int, int, int]]:
        """
        Detect regime changes in time series
        
        Args:
            data: Time series data
            model_name: Name of fitted model to use
            
        Returns:
            List of (start_time, end_time, regime_id) tuples
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' not found")
            
        model = self.fitted_models[model_name]
        result = model.predict(data)
        
        # Find regime changes
        regime_changes = []
        current_regime = result.regime_sequence[0]
        start_time = 0
        
        for i, regime in enumerate(result.regime_sequence[1:], 1):
            if regime != current_regime:
                # Regime change detected
                regime_changes.append((start_time, i-1, current_regime))
                start_time = i
                current_regime = regime
                
        # Add final regime
        regime_changes.append((start_time, len(result.regime_sequence)-1, current_regime))
        
        return regime_changes

# Example usage and testing
if __name__ == "__main__":
    print("Testing Regime Detection...")
    
    # Generate synthetic data with regime changes
    np.random.seed(42)
    n_obs = 500
    
    # Regime 1: Low volatility, positive drift
    regime1_data = np.random.normal(0.001, 0.01, 150)
    
    # Regime 2: High volatility, negative drift  
    regime2_data = np.random.normal(-0.002, 0.03, 200)
    
    # Regime 3: Medium volatility, positive drift
    regime3_data = np.random.normal(0.0015, 0.015, 150)
    
    # Combine regimes
    synthetic_returns = np.concatenate([regime1_data, regime2_data, regime3_data])
    true_regimes = [0]*150 + [1]*200 + *150
    
    print(f"Generated {len(synthetic_returns)} observations with 3 true regimes")
    
    # Initialize regime detection
    regime_detector = RegimeDetection()
    
    # Test HMM regime detection
    print("\nTesting Hidden Markov Model:")
    hmm_model = regime_detector.fit_model(
        synthetic_returns, 'hmm', 'test_hmm', n_regimes=3
    )
    
    hmm_result = hmm_model.predict(synthetic_returns)
    print(f"HMM detected {hmm_result.n_regimes} regimes")
    print(f"Model likelihood: {hmm_result.model_likelihood:.2f}")
    print(f"AIC: {hmm_result.aic:.2f}, BIC: {hmm_result.bic:.2f}")
    
    # Show regime characteristics
    print("\nHMM Regime Characteristics:")
    for regime in hmm_result.regime_states:
        print(f"  Regime {regime.regime_id}: mean={regime.mean_return:.4f}, "
              f"vol={regime.volatility:.4f}, persistence={regime.persistence:.3f}")
              
    # Test Threshold model
    print("\nTesting Threshold Model:")
    threshold_model = regime_detector.fit_model(
        synthetic_returns, 'threshold', 'test_threshold', 
        n_regimes=3, threshold_variable='change'
    )
    
    threshold_result = threshold_model.predict(synthetic_returns)
    print(f"Threshold model AIC: {threshold_result.aic:.2f}, BIC: {threshold_result.bic:.2f}")
    
    # Compare models
    print("\nComparing Models:")
    comparison = regime_detector.compare_models(
        synthetic_returns, 
        model_types=['hmm', 'threshold'],
        n_regimes_range=(2, 4)
    )
    
    print(comparison[['model_type', 'n_regimes', 'aic', 'bic']].round(2))
    
    # Get best model
    best_model = regime_detector.get_best_model('bic')
    print(f"\nBest model by BIC: {type(best_model).__name__}")
    
    # Detect regime changes
    regime_changes = regime_detector.detect_regime_changes(synthetic_returns, 'test_hmm')
    print(f"\nDetected {len(regime_changes)} regime periods:")
    for start, end, regime in regime_changes[:5]:  # Show first 5
        print(f"  Period {start}-{end}: Regime {regime}")
        
    print("\nRegime detection test completed!")
