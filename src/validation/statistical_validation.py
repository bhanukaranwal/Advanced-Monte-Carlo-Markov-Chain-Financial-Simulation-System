"""
Statistical validation framework for model verification and cross-validation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Statistical validation result"""
    test_name: str
    test_statistic: float
    p_value: float
    critical_value: float
    rejection_threshold: float
    is_significant: bool
    interpretation: str
    additional_info: Dict[str, Any] = None

@dataclass
class CrossValidationResult:
    """Cross-validation result"""
    model_name: str
    cv_scores: np.ndarray
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    metrics: Dict[str, float]

class StatisticalValidator:
    """Comprehensive statistical validation framework"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.validation_results = []
        
    def normality_test(
        self,
        data: Union[np.ndarray, pd.Series],
        test_type: str = 'jarque_bera'
    ) -> ValidationResult:
        """
        Test for normality in data
        
        Args:
            data: Data to test
            test_type: Type of test ('jarque_bera', 'shapiro', 'anderson', 'ks')
            
        Returns:
            ValidationResult with test results
        """
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
            
        if len(data) == 0:
            return ValidationResult(
                test_name=f"normality_{test_type}",
                test_statistic=0,
                p_value=1,
                critical_value=0,
                rejection_threshold=self.significance_level,
                is_significant=False,
                interpretation="No data available for testing"
            )
            
        if test_type == 'jarque_bera':
            statistic, p_value = stats.jarque_bera(data)
            critical_value = stats.chi2.ppf(1 - self.significance_level, df=2)
            interpretation = "Data is normally distributed" if p_value > self.significance_level else "Data is not normally distributed"
            
        elif test_type == 'shapiro':
            statistic, p_value = stats.shapiro(data)
            critical_value = 0.95  # Approximate critical value for Shapiro-Wilk
            interpretation = "Data is normally distributed" if p_value > self.significance_level else "Data is not normally distributed"
            
        elif test_type == 'anderson':
            result = stats.anderson(data, dist='norm')
            statistic = result.statistic
            # Use 5% significance level critical value
            critical_value = result.critical_values[2]  # 5% level
            p_value = 0.05 if statistic > critical_value else 0.1  # Approximate
            interpretation = "Data is normally distributed" if statistic < critical_value else "Data is not normally distributed"
            
        elif test_type == 'ks':
            # Kolmogorov-Smirnov test against normal distribution
            normalized_data = (data - np.mean(data)) / np.std(data)
            statistic, p_value = stats.kstest(normalized_data, 'norm')
            critical_value = stats.ksone.ppf(1 - self.significance_level, len(data))
            interpretation = "Data is normally distributed" if p_value > self.significance_level else "Data is not normally distributed"
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        result = ValidationResult(
            test_name=f"normality_{test_type}",
            test_statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            rejection_threshold=self.significance_level,
            is_significant=p_value < self.significance_level,
            interpretation=interpretation,
            additional_info={
                'data_mean': np.mean(data),
                'data_std': np.std(data),
                'data_skewness': stats.skew(data),
                'data_kurtosis': stats.kurtosis(data),
                'sample_size': len(data)
            }
        )
        
        self.validation_results.append(result)
        return result
        
    def stationarity_test(
        self,
        data: Union[np.ndarray, pd.Series],
        test_type: str = 'adf'
    ) -> ValidationResult:
        """
        Test for stationarity in time series data
        
        Args:
            data: Time series data
            test_type: Type of test ('adf', 'kpss', 'pp')
            
        Returns:
            ValidationResult with test results
        """
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
        except ImportError:
            logger.warning("statsmodels not available for stationarity tests")
            return ValidationResult(
                test_name=f"stationarity_{test_type}",
                test_statistic=0,
                p_value=1,
                critical_value=0,
                rejection_threshold=self.significance_level,
                is_significant=False,
                interpretation="statsmodels not available"
            )
            
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
            
        if test_type == 'adf':
            # Augmented Dickey-Fuller test
            result = adfuller(data, autolag='AIC')
            statistic = result[0]
            p_value = result[1]
            critical_value = result['5%']  # 5% critical value
            interpretation = "Data is stationary" if p_value < self.significance_level else "Data is non-stationary"
            
        elif test_type == 'kpss':
            # KPSS test (null hypothesis: stationary)
            statistic, p_value, lags, critical_values = kpss(data, regression='c')
            critical_value = critical_values['5%']
            interpretation = "Data is stationary" if p_value > self.significance_level else "Data is non-stationary"
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        result = ValidationResult(
            test_name=f"stationarity_{test_type}",
            test_statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            rejection_threshold=self.significance_level,
            is_significant=p_value < self.significance_level if test_type == 'adf' else p_value > self.significance_level,
            interpretation=interpretation,
            additional_info={
                'sample_size': len(data),
                'test_type': test_type
            }
        )
        
        self.validation_results.append(result)
        return result
        
    def autocorrelation_test(
        self,
        data: Union[np.ndarray, pd.Series],
        max_lags: int = 20
    ) -> ValidationResult:
        """
        Test for autocorrelation in residuals
        
        Args:
            data: Data to test (typically residuals)
            max_lags: Maximum number of lags to test
            
        Returns:
            ValidationResult with Ljung-Box test results
        """
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
        except ImportError:
            logger.warning("statsmodels not available for autocorrelation tests")
            return ValidationResult(
                test_name="autocorrelation_ljungbox",
                test_statistic=0,
                p_value=1,
                critical_value=0,
                rejection_threshold=self.significance_level,
                is_significant=False,
                interpretation="statsmodels not available"
            )
            
        if isinstance(data, pd.Series):
            data = data.dropna().values
        else:
            data = data[~np.isnan(data)]
            
        # Ljung-Box test
        result = acorr_ljungbox(data, lags=max_lags, return_df=True)
        
        # Use the statistic and p-value for the specified lag
        statistic = result['lb_stat'].iloc[-1]
        p_value = result['lb_pvalue'].iloc[-1]
        critical_value = stats.chi2.ppf(1 - self.significance_level, df=max_lags)
        
        interpretation = "No significant autocorrelation" if p_value > self.significance_level else "Significant autocorrelation detected"
        
        validation_result = ValidationResult(
            test_name="autocorrelation_ljungbox",
            test_statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            rejection_threshold=self.significance_level,
            is_significant=p_value < self.significance_level,
            interpretation=interpretation,
            additional_info={
                'max_lags': max_lags,
                'all_statistics': result['lb_stat'].tolist(),
                'all_p_values': result['lb_pvalue'].tolist()
            }
        )
        
        self.validation_results.append(validation_result)
        return validation_result
        
    def heteroscedasticity_test(
        self,
        residuals: Union[np.ndarray, pd.Series],
        fitted_values: Union[np.ndarray, pd.Series] = None,
        test_type: str = 'breusch_pagan'
    ) -> ValidationResult:
        """
        Test for heteroscedasticity
        
        Args:
            residuals: Model residuals
            fitted_values: Fitted values from model
            test_type: Type of test ('breusch_pagan', 'white')
            
        Returns:
            ValidationResult with test results
        """
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan, het_white
        except ImportError:
            logger.warning("statsmodels not available for heteroscedasticity tests")
            return ValidationResult(
                test_name=f"heteroscedasticity_{test_type}",
                test_statistic=0,
                p_value=1,
                critical_value=0,
                rejection_threshold=self.significance_level,
                is_significant=False,
                interpretation="statsmodels not available"
            )
            
        if isinstance(residuals, pd.Series):
            residuals = residuals.dropna().values
        else:
            residuals = residuals[~np.isnan(residuals)]
            
        if fitted_values is None:
            # Use squared residuals as proxy for fitted values
            fitted_values = residuals ** 2
        elif isinstance(fitted_values, pd.Series):
            fitted_values = fitted_values.dropna().values
            
        # Ensure same length
        min_length = min(len(residuals), len(fitted_values))
        residuals = residuals[:min_length]
        fitted_values = fitted_values[:min_length]
        
        if test_type == 'breusch_pagan':
            # Breusch-Pagan test
            lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, fitted_values.reshape(-1, 1))
            statistic = lm
            p_value = lm_pvalue
            critical_value = stats.chi2.ppf(1 - self.significance_level, df=1)
            
        elif test_type == 'white':
            # White test
            lm, lm_pvalue, fvalue, f_pvalue = het_white(residuals, fitted_values.reshape(-1, 1))
            statistic = lm
            p_value = lm_pvalue
            critical_value = stats.chi2.ppf(1 - self.significance_level, df=2)  # Approximate
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        interpretation = "Homoscedastic residuals" if p_value > self.significance_level else "Heteroscedastic residuals"
        
        result = ValidationResult(
            test_name=f"heteroscedasticity_{test_type}",
            test_statistic=statistic,
            p_value=p_value,
            critical_value=critical_value,
            rejection_threshold=self.significance_level,
            is_significant=p_value < self.significance_level,
            interpretation=interpretation,
            additional_info={
                'test_type': test_type,
                'sample_size': len(residuals)
            }
        )
        
        self.validation_results.append(result)
        return result
        
    def independence_test(
        self,
        data1: Union[np.ndarray, pd.Series],
        data2: Union[np.ndarray, pd.Series],
        test_type: str = 'chi2'
    ) -> ValidationResult:
        """
        Test for independence between two variables
        
        Args:
            data1: First variable
            data2: Second variable
            test_type: Type of test ('chi2', 'mutual_info')
            
        Returns:
            ValidationResult with test results
        """
        if isinstance(data1, pd.Series):
            data1 = data1.dropna().values
        if isinstance(data2, pd.Series):
            data2 = data2.dropna().values
            
        # Ensure same length
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]
        
        if test_type == 'chi2':
            # Chi-square test of independence
            # First, create contingency table
            # Discretize continuous data into bins
            bins1 = np.histogram_bin_edges(data1, bins='auto')
            bins2 = np.histogram_bin_edges(data2, bins='auto')
            
            # Limit number of bins to avoid sparse contingency table
            if len(bins1) > 11:
                bins1 = np.linspace(data1.min(), data1.max(), 11)
            if len(bins2) > 11:
                bins2 = np.linspace(data2.min(), data2.max(), 11)
                
            digitized1 = np.digitize(data1, bins1[1:-1])
            digitized2 = np.digitize(data2, bins2[1:-1])
            
            # Create contingency table
            contingency_table = pd.crosstab(digitized1, digitized2)
            
            # Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            critical_value = stats.chi2.ppf(1 - self.significance_level, df=dof)
            
            interpretation = "Variables are independent" if p_value > self.significance_level else "Variables are not independent"
            
            result = ValidationResult(
                test_name="independence_chi2",
                test_statistic=chi2,
                p_value=p_value,
                critical_value=critical_value,
                rejection_threshold=self.significance_level,
                is_significant=p_value < self.significance_level,
                interpretation=interpretation,
                additional_info={
                    'degrees_of_freedom': dof,
                    'contingency_table_shape': contingency_table.shape
                }
            )
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        self.validation_results.append(result)
        return result
        
    def model_specification_test(
        self,
        actual: Union[np.ndarray, pd.Series],
        predicted: Union[np.ndarray, pd.Series],
        test_type: str = 'ramsey_reset'
    ) -> ValidationResult:
        """
        Test for model specification
        
        Args:
            actual: Actual values
            predicted: Predicted values
            test_type: Type of test ('ramsey_reset', 'harvey_collier')
            
        Returns:
            ValidationResult with test results
        """
        if isinstance(actual, pd.Series):
            actual = actual.dropna().values
        if isinstance(predicted, pd.Series):
            predicted = predicted.dropna().values
            
        # Ensure same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]
        
        # Calculate residuals
        residuals = actual - predicted
        
        if test_type == 'ramsey_reset':
            # Simplified RESET test
            # Test if powers of fitted values help explain residuals
            fitted_squared = predicted ** 2
            fitted_cubed = predicted ** 3
            
            # Regression of residuals on fitted values and their powers
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Model 1: residuals ~ constant
            baseline_r2 = 0
            
            # Model 2: residuals ~ fitted + fitted^2 + fitted^3
            X = np.column_stack([predicted, fitted_squared, fitted_cubed])
            reg = LinearRegression().fit(X, residuals)
            full_r2 = r2_score(residuals, reg.predict(X))
            
            # F-test for improvement
            n = len(residuals)
            k1 = 0  # baseline model parameters
            k2 = 3  # full model parameters
            
            f_stat = ((full_r2 - baseline_r2) / (k2 - k1)) / ((1 - full_r2) / (n - k2 - 1))
            p_value = 1 - stats.f.cdf(f_stat, k2 - k1, n - k2 - 1)
            critical_value = stats.f.ppf(1 - self.significance_level, k2 - k1, n - k2 - 1)
            
            interpretation = "Model specification is correct" if p_value > self.significance_level else "Model specification may be incorrect"
            
            result = ValidationResult(
                test_name="specification_ramsey_reset",
                test_statistic=f_stat,
                p_value=p_value,
                critical_value=critical_value,
                rejection_threshold=self.significance_level,
                is_significant=p_value < self.significance_level,
                interpretation=interpretation,
                additional_info={
                    'r_squared_improvement': full_r2 - baseline_r2,
                    'full_r_squared': full_r2
                }
            )
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        self.validation_results.append(result)
        return result
        
    def generate_validation_report(self) -> pd.DataFrame:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return pd.DataFrame()
            
        report_data = []
        
        for result in self.validation_results:
            report_data.append({
                'test_name': result.test_name,
                'test_statistic': result.test_statistic,
                'p_value': result.p_value,
                'critical_value': result.critical_value,
                'is_significant': result.is_significant,
                'interpretation': result.interpretation
            })
            
        return pd.DataFrame(report_data)

class ModelValidator:
    """Model-specific validation framework"""
    
    def __init__(self, statistical_validator: StatisticalValidator = None):
        self.statistical_validator = statistical_validator or StatisticalValidator()
        
    def validate_monte_carlo_model(
        self,
        theoretical_moments: Dict[str, float],
        simulated_data: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, ValidationResult]:
        """
        Validate Monte Carlo simulation against theoretical moments
        
        Args:
            theoretical_moments: Dictionary of theoretical moments
            simulated_data: Simulated data
            confidence_level: Confidence level for tests
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Sample moments
        sample_mean = np.mean(simulated_data)
        sample_var = np.var(simulated_data)
        sample_std = np.std(simulated_data)
        sample_skew = stats.skew(simulated_data)
        sample_kurt = stats.kurtosis(simulated_data)
        
        n = len(simulated_data)
        
        # Test mean
        if 'mean' in theoretical_moments:
            theoretical_mean = theoretical_moments['mean']
            theoretical_std = theoretical_moments.get('std', sample_std)
            
            # One-sample t-test
            t_stat = (sample_mean - theoretical_mean) / (sample_std / np.sqrt(n))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
            critical_value = stats.t.ppf(1 - (1-confidence_level)/2, df=n-1)
            
            results['mean_test'] = ValidationResult(
                test_name='monte_carlo_mean',
                test_statistic=t_stat,
                p_value=p_value,
                critical_value=critical_value,
                rejection_threshold=1-confidence_level,
                is_significant=abs(t_stat) > critical_value,
                interpretation=f"Sample mean {'differs significantly' if abs(t_stat) > critical_value else 'matches'} theoretical mean",
                additional_info={
                    'theoretical_mean': theoretical_mean,
                    'sample_mean': sample_mean,
                    'difference': sample_mean - theoretical_mean
                }
            )
            
        # Test variance
        if 'variance' in theoretical_moments:
            theoretical_var = theoretical_moments['variance']
            
            # Chi-square test for variance
            chi2_stat = (n - 1) * sample_var / theoretical_var
            p_value = 2 * min(stats.chi2.cdf(chi2_stat, df=n-1), 
                             1 - stats.chi2.cdf(chi2_stat, df=n-1))
            critical_value_lower = stats.chi2.ppf((1-confidence_level)/2, df=n-1)
            critical_value_upper = stats.chi2.ppf(1-(1-confidence_level)/2, df=n-1)
            
            results['variance_test'] = ValidationResult(
                test_name='monte_carlo_variance',
                test_statistic=chi2_stat,
                p_value=p_value,
                critical_value=(critical_value_lower + critical_value_upper) / 2,  # Average for display
                rejection_threshold=1-confidence_level,
                is_significant=chi2_stat < critical_value_lower or chi2_stat > critical_value_upper,
                interpretation=f"Sample variance {'differs significantly' if (chi2_stat < critical_value_lower or chi2_stat > critical_value_upper) else 'matches'} theoretical variance",
                additional_info={
                    'theoretical_variance': theoretical_var,
                    'sample_variance': sample_var,
                    'ratio': sample_var / theoretical_var
                }
            )
            
        return results
        
    def validate_markov_chain(
        self,
        transition_matrix: np.ndarray,
        observed_transitions: np.ndarray,
        state_sequence: List[int]
    ) -> Dict[str, ValidationResult]:
        """
        Validate Markov chain model
        
        Args:
            transition_matrix: Theoretical transition matrix
            observed_transitions: Observed transition counts
            state_sequence: Observed state sequence
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        n_states = transition_matrix.shape[0]
        
        # Chi-square test for transition probabilities
        expected_transitions = np.zeros_like(observed_transitions)
        total_transitions = np.sum(observed_transitions, axis=1)
        
        for i in range(n_states):
            if total_transitions[i] > 0:
                expected_transitions[i] = total_transitions[i] * transition_matrix[i]
                
        # Avoid division by zero
        expected_transitions = np.maximum(expected_transitions, 1e-8)
        
        chi2_stat = np.sum((observed_transitions - expected_transitions)**2 / expected_transitions)
        df = n_states * (n_states - 1)  # Degrees of freedom
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=df)
        critical_value = stats.chi2.ppf(0.95, df=df)
        
        results['transition_test'] = ValidationResult(
            test_name='markov_transition_matrix',
            test_statistic=chi2_stat,
            p_value=p_value,
            critical_value=critical_value,
            rejection_threshold=0.05,
            is_significant=chi2_stat > critical_value,
            interpretation=f"Transition matrix {'differs significantly' if chi2_stat > critical_value else 'matches'} observed transitions",
            additional_info={
                'degrees_of_freedom': df,
                'observed_transitions': observed_transitions.tolist(),
                'expected_transitions': expected_transitions.tolist()
            }
        )
        
        # Test for Markov property (independence of non-adjacent states)
        if len(state_sequence) > 2:
            # Simplified test: check if P(X_t+1 | X_t, X_t-1) = P(X_t+1 | X_t)
            # This is a complex test, so we'll do a simplified version
            
            # Count three-step transitions
            three_step_counts = np.zeros((n_states, n_states, n_states))
            for i in range(len(state_sequence) - 2):
                s1, s2, s3 = state_sequence[i], state_sequence[i+1], state_sequence[i+2]
                three_step_counts[s1, s2, s3] += 1
                
            # Test if P(X_t+1 | X_t, X_t-1) = P(X_t+1 | X_t)
            # This is a simplified approximation
            markov_property_violations = 0
            total_tests = 0
            
            for i in range(n_states):
                for j in range(n_states):
                    for k in range(n_states):
                        if three_step_counts[i, j, k] > 5:  # Minimum count for meaningful test
                            total_tests += 1
                            # Compare observed vs expected under Markov assumption
                            observed = three_step_counts[i, j, k]
                            expected = observed_transitions[j, k] / max(total_transitions[j], 1)
                            if abs(observed - expected) > 2 * np.sqrt(expected):  # Rough test
                                markov_property_violations += 1
                                
            violation_rate = markov_property_violations / max(total_tests, 1)
            
            results['markov_property'] = ValidationResult(
                test_name='markov_property',
                test_statistic=violation_rate,
                p_value=violation_rate,  # Simplified
                critical_value=0.1,  # 10% violation threshold
                rejection_threshold=0.1,
                is_significant=violation_rate > 0.1,
                interpretation=f"Markov property {'violated' if violation_rate > 0.1 else 'satisfied'} (violation rate: {violation_rate:.2%})",
                additional_info={
                    'total_tests': total_tests,
                    'violations': markov_property_violations
                }
            )
            
        return results

class CrossValidator:
    """Time series cross-validation framework"""
    
    def __init__(self, n_splits: int = 5, test_size: int = None):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def time_series_cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Union[str, Callable] = 'neg_mean_squared_error',
        fit_params: Dict = None
    ) -> CrossValidationResult:
        """
        Perform time series cross-validation
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target values
            scoring: Scoring metric
            fit_params: Parameters to pass to fit method
            
        Returns:
            CrossValidationResult with validation metrics
        """
        if fit_params is None:
            fit_params = {}
            
        # Use TimeSeriesSplit for proper time series validation
        if self.test_size is not None:
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        else:
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, fit_params=fit_params)
        
        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        # Confidence interval
        confidence_interval = (
            mean_score - 1.96 * std_score / np.sqrt(len(cv_scores)),
            mean_score + 1.96 * std_score / np.sqrt(len(cv_scores))
        )
        
        # Additional metrics
        metrics = {
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'median_score': np.median(cv_scores)
        }
        
        return CrossValidationResult(
            model_name=type(model).__name__,
            cv_scores=cv_scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=confidence_interval,
            metrics=metrics
        )
        
    def rolling_window_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        window_size: int,
        step_size: int = 1,
        metric_func: Callable = mean_squared_error
    ) -> Dict[str, Any]:
        """
        Rolling window validation
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target values
            window_size: Size of training window
            step_size: Step size for rolling window
            metric_func: Metric function to use
            
        Returns:
            Dictionary with validation results
        """
        scores = []
        predictions = []
        actual_values = []
        
        for i in range(window_size, len(X), step_size):
            # Training data
            X_train = X[i-window_size:i]
            y_train = y[i-window_size:i]
            
            # Test data (next point)
            if i < len(X):
                X_test = X[i:i+1]
                y_test = y[i:i+1]
                
                # Fit model and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate score
                score = metric_func(y_test, y_pred)
                scores.append(score)
                predictions.extend(y_pred)
                actual_values.extend(y_test)
                
        return {
            'scores': np.array(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'predictions': np.array(predictions),
            'actual_values': np.array(actual_values),
            'total_predictions': len(predictions)
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Statistical Validation Framework...")
    
    # Generate test data
    np.random.seed(42)
    
    # Test normality validation
    print("Testing normality tests:")
    validator = StatisticalValidator()
    
    # Normal data
    normal_data = np.random.normal(0, 1, 1000)
    normal_result = validator.normality_test(normal_data, 'jarque_bera')
    print(f"Normal data - Jarque-Bera test: {normal_result.interpretation} (p={normal_result.p_value:.4f})")
    
    # Non-normal data (exponential)
    exp_data = np.random.exponential(1, 1000)
    exp_result = validator.normality_test(exp_data, 'jarque_bera')
    print(f"Exponential data - Jarque-Bera test: {exp_result.interpretation} (p={exp_result.p_value:.4f})")
    
    # Test stationarity
    print("\nTesting stationarity tests:")
    # Stationary data (white noise)
    stationary_data = np.random.normal(0, 1, 500)
    stat_result = validator.stationarity_test(stationary_data, 'adf')
    print(f"White noise - ADF test: {stat_result.interpretation} (p={stat_result.p_value:.4f})")
    
    # Non-stationary data (random walk)
    non_stationary_data = np.cumsum(np.random.normal(0, 1, 500))
    non_stat_result = validator.stationarity_test(non_stationary_data, 'adf')
    print(f"Random walk - ADF test: {non_stat_result.interpretation} (p={non_stat_result.p_value:.4f})")
    
    # Test Model Validator
    print("\nTesting Monte Carlo validation:")
    model_validator = ModelValidator(validator)
    
    # Generate Monte Carlo data with known parameters
    theoretical_mean = 0.05
    theoretical_var = 0.04
    mc_data = np.random.normal(theoretical_mean, np.sqrt(theoretical_var), 10000)
    
    theoretical_moments = {
        'mean': theoretical_mean,
        'variance': theoretical_var,
        'std': np.sqrt(theoretical_var)
    }
    
    mc_results = model_validator.validate_monte_carlo_model(theoretical_moments, mc_data)
    
    for test_name, result in mc_results.items():
        print(f"{test_name}: {result.interpretation} (p={result.p_value:.4f})")
        
    # Test Cross Validator
    print("\nTesting Cross Validation:")
    from sklearn.linear_model import LinearRegression
    
    # Generate time series data
    n_samples = 200
    X = np.random.randn(n_samples, 3)  # 3 features
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    # Add time dependency
    for i in range(1, n_samples):
        y[i] += 0.3 * y[i-1]  # AR(1) component
        
    cross_validator = CrossValidator(n_splits=5)
    model = LinearRegression()
    
    cv_result = cross_validator.time_series_cross_validate(model, X, y, scoring='neg_mean_squared_error')
    
    print(f"Cross-validation results for {cv_result.model_name}:")
    print(f"  Mean score: {cv_result.mean_score:.4f}")
    print(f"  Std score: {cv_result.std_score:.4f}")
    print(f"  95% CI: [{cv_result.confidence_interval[0]:.4f}, {cv_result.confidence_interval[1]:.4f}]")
    
    # Test rolling window validation
    print("\nTesting Rolling Window Validation:")
    rolling_results = cross_validator.rolling_window_validate(
        model, X, y, window_size=50, step_size=5
    )
    
    print(f"Rolling window validation:")
    print(f"  Mean score: {rolling_results['mean_score']:.4f}")
    print(f"  Std score: {rolling_results['std_score']:.4f}")
    print(f"  Total predictions: {rolling_results['total_predictions']}")
    
    # Generate validation report
    print("\nValidation Report:")
    report = validator.generate_validation_report()
    print(report[['test_name', 'p_value', 'is_significant', 'interpretation']])
    
    print("\nStatistical validation framework test completed!")
