"""
Copula models for multivariate dependence modeling in finance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import gamma, digamma
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

class BaseCopula:
    """Base class for copula models"""
    
    def __init__(self):
        self.fitted = False
        self.parameters = {}
        self.log_likelihood = None
        
    def fit(self, data: np.ndarray) -> 'BaseCopula':
        """Fit copula to data"""
        raise NotImplementedError
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from copula"""
        raise NotImplementedError
        
    def pdf(self, u: np.ndarray) -> np.ndarray:
        """Probability density function"""
        raise NotImplementedError
        
    def cdf(self, u: np.ndarray) -> np.ndarray:
        """Cumulative distribution function"""
        raise NotImplementedError
        
    def aic(self) -> float:
        """Akaike Information Criterion"""
        if self.log_likelihood is None:
            return np.inf
        k = len(self.parameters)
        return 2 * k - 2 * self.log_likelihood
        
    def bic(self, n_obs: int) -> float:
        """Bayesian Information Criterion"""
        if self.log_likelihood is None:
            return np.inf
        k = len(self.parameters)
        return k * np.log(n_obs) - 2 * self.log_likelihood

class GaussianCopula(BaseCopula):
    """Gaussian (Normal) Copula"""
    
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.correlation_matrix = None
        
    def fit(self, data: np.ndarray) -> 'GaussianCopula':
        """
        Fit Gaussian copula to data
        
        Args:
            data: Uniform data in [0,1]^d
            
        Returns:
            Fitted copula
        """
        if data.shape[1] != self.dimension:
            raise ValueError(f"Data dimension {data.shape[1]} != copula dimension {self.dimension}")
            
        # Convert uniform data to standard normal
        normal_data = stats.norm.ppf(np.clip(data, 1e-6, 1-1e-6))
        
        # Estimate correlation matrix
        self.correlation_matrix = np.corrcoef(normal_data.T)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        self.correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Calculate log-likelihood
        self.log_likelihood = self._calculate_log_likelihood(data)
        
        # Store parameters
        self.parameters = {'correlation_matrix': self.correlation_matrix}
        self.fitted = True
        
        logger.info(f"Fitted Gaussian copula with {self.dimension} dimensions")
        return self
        
    def _calculate_log_likelihood(self, data: np.ndarray) -> float:
        """Calculate log-likelihood"""
        normal_data = stats.norm.ppf(np.clip(data, 1e-6, 1-1e-6))
        
        # Determinant and inverse of correlation matrix
        det_R = np.linalg.det(self.correlation_matrix)
        inv_R = np.linalg.inv(self.correlation_matrix)
        
        # Log-likelihood calculation
        ll = 0.0
        for i in range(len(data)):
            x = normal_data[i]
            quadratic_form = x.T @ (inv_R - np.eye(self.dimension)) @ x
            ll += -0.5 * (np.log(det_R) + quadratic_form)
            
        return ll
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from Gaussian copula"""
        if not self.fitted:
            raise ValueError("Copula must be fitted before sampling")
            
        # Generate multivariate normal samples
        normal_samples = np.random.multivariate_normal(
            np.zeros(self.dimension),
            self.correlation_matrix,
            n_samples
        )
        
        # Transform to uniform using normal CDF
        uniform_samples = stats.norm.cdf(normal_samples)
        
        return uniform_samples
        
    def pdf(self, u: np.ndarray) -> np.ndarray:
        """Copula density function"""
        if not self.fitted:
            raise ValueError("Copula must be fitted before computing PDF")
            
        # Transform to normal
        normal_data = stats.norm.ppf(np.clip(u, 1e-6, 1-1e-6))
        
        # Gaussian copula density
        det_R = np.linalg.det(self.correlation_matrix)
        inv_R = np.linalg.inv(self.correlation_matrix)
        
        densities = []
        for i in range(len(normal_data)):
            x = normal_data[i]
            quadratic_form = x.T @ (inv_R - np.eye(self.dimension)) @ x
            density = (1 / np.sqrt(det_R)) * np.exp(-0.5 * quadratic_form)
            densities.append(density)
            
        return np.array(densities)
        
    def conditional_distribution(self, u: np.ndarray, given_indices: List[int]) -> Callable:
        """Compute conditional distribution"""
        if not self.fitted:
            raise ValueError("Copula must be fitted")
            
        # This is a simplified implementation
        # Full implementation would use conditional multivariate normal formulas
        def conditional_cdf(v):
            # Placeholder implementation
            return stats.norm.cdf(v)
            
        return conditional_cdf

class TCopula(BaseCopula):
    """Student t-Copula"""
    
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.correlation_matrix = None
        self.degrees_of_freedom = None
        
    def fit(self, data: np.ndarray) -> 'TCopula':
        """Fit t-copula to data"""
        if data.shape[1] != self.dimension:
            raise ValueError(f"Data dimension {data.shape[1]} != copula dimension {self.dimension}")
            
        # Initial parameter estimates
        # Use method of moments for correlation matrix
        normal_data = stats.norm.ppf(np.clip(data, 1e-6, 1-1e-6))
        initial_corr = np.corrcoef(normal_data.T)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(initial_corr)
        eigenvals = np.maximum(eigenvals, 1e-8)
        initial_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Estimate degrees of freedom using MLE
        def neg_log_likelihood(params):
            nu = params[0]
            if nu <= 2:
                return 1e10  # Invalid degrees of freedom
                
            # Transform data using t-distribution
            t_data = stats.t.ppf(np.clip(data, 1e-6, 1-1e-6), df=nu)
            
            # Check for invalid values
            if np.any(~np.isfinite(t_data)):
                return 1e10
                
            # Calculate log-likelihood (simplified)
            ll = 0.0
            try:
                det_R = np.linalg.det(initial_corr)
                inv_R = np.linalg.inv(initial_corr)
                
                for i in range(len(data)):
                    x = t_data[i]
                    quadratic_form = x.T @ inv_R @ x
                    
                    # t-copula density components
                    gamma_term = (
                        gamma((nu + self.dimension) / 2) * gamma(nu / 2)**(self.dimension - 1) /
                        (gamma(nu / 2) * gamma((nu + 1) / 2)**(self.dimension))
                    )
                    
                    density_term = (
                        gamma_term / np.sqrt(det_R) *
                        (1 + quadratic_form / nu)**(-(nu + self.dimension) / 2) *
                        np.prod((1 + x**2 / nu)**(-(nu + 1) / 2))
                    )
                    
                    ll += np.log(max(density_term, 1e-10))
                    
            except (np.linalg.LinAlgError, ValueError):
                return 1e10
                
            return -ll
            
        # Optimize degrees of freedom
        result = minimize(
            neg_log_likelihood,
            x0=[5.0],  # Initial guess
            bounds=[(2.1, 50)],  # Reasonable bounds for nu
            method='L-BFGS-B'
        )
        
        if result.success:
            self.degrees_of_freedom = result.x[0]
        else:
            logger.warning("Failed to estimate degrees of freedom, using default value of 5")
            self.degrees_of_freedom = 5.0
            
        # Re-estimate correlation matrix with estimated nu
        t_data = stats.t.ppf(np.clip(data, 1e-6, 1-1e-6), df=self.degrees_of_freedom)
        self.correlation_matrix = np.corrcoef(t_data.T)
        
        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        self.correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Calculate final log-likelihood
        self.log_likelihood = -neg_log_likelihood([self.degrees_of_freedom])
        
        self.parameters = {
            'correlation_matrix': self.correlation_matrix,
            'degrees_of_freedom': self.degrees_of_freedom
        }
        self.fitted = True
        
        logger.info(f"Fitted t-copula: nu={self.degrees_of_freedom:.2f}")
        return self
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from t-copula"""
        if not self.fitted:
            raise ValueError("Copula must be fitted before sampling")
            
        # Generate multivariate t samples
        # First generate chi-squared random variable
        chi2_samples = np.random.chisquare(self.degrees_of_freedom, n_samples)
        
        # Generate multivariate normal samples
        normal_samples = np.random.multivariate_normal(
            np.zeros(self.dimension),
            self.correlation_matrix,
            n_samples
        )
        
        # Transform to multivariate t
        t_samples = normal_samples * np.sqrt(self.degrees_of_freedom / chi2_samples[:, np.newaxis])
        
        # Transform to uniform using t CDF
        uniform_samples = stats.t.cdf(t_samples, df=self.degrees_of_freedom)
        
        return uniform_samples

class ClaytonCopula(BaseCopula):
    """Clayton Copula (Archimedean family)"""
    
    def __init__(self, dimension: int = 2):
        super().__init__()
        self.dimension = dimension
        self.theta = None
        
    def fit(self, data: np.ndarray) -> 'ClaytonCopula':
        """Fit Clayton copula using method of moments or MLE"""
        if data.shape[1] != self.dimension:
            raise ValueError(f"Data dimension {data.shape[1]} != copula dimension {self.dimension}")
            
        if self.dimension != 2:
            raise NotImplementedError("Clayton copula only implemented for bivariate case")
            
        # Method of moments estimation using Kendall's tau
        tau = stats.kendalltau(data[:, 0], data[:, 1])[0]
        
        # Relationship: tau = theta / (theta + 2)
        if tau > 0:
            self.theta = 2 * tau / (1 - tau)
        else:
            self.theta = 0.1  # Small positive value
            
        # Ensure theta > 0
        self.theta = max(self.theta, 0.01)
        
        # Calculate log-likelihood
        self.log_likelihood = self._calculate_log_likelihood(data)
        
        self.parameters = {'theta': self.theta}
        self.fitted = True
        
        logger.info(f"Fitted Clayton copula: theta={self.theta:.4f}")
        return self
        
    def _calculate_log_likelihood(self, data: np.ndarray) -> float:
        """Calculate log-likelihood for Clayton copula"""
        u, v = data[:, 0], data[:, 1]
        
        # Clip to avoid numerical issues
        u = np.clip(u, 1e-6, 1-1e-6)
        v = np.clip(v, 1e-6, 1-1e-6)
        
        # Clayton copula log-likelihood
        ll = 0.0
        for i in range(len(u)):
            ui, vi = u[i], v[i]
            
            # Density
            term1 = np.log(1 + self.theta)
            term2 = -(1 + self.theta) * (np.log(ui) + np.log(vi))
            term3 = -(2 + 1/self.theta) * np.log(ui**(-self.theta) + vi**(-self.theta) - 1)
            
            ll += term1 + term2 + term3
            
        return ll
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from Clayton copula"""
        if not self.fitted:
            raise ValueError("Copula must be fitted before sampling")
            
        # Generate samples using conditional method
        u1 = np.random.uniform(0, 1, n_samples)
        w = np.random.uniform(0, 1, n_samples)
        
        # Conditional distribution: C_{2|1}^{-1}(w|u1)
        u2 = u1 * (w**(-self.theta / (1 + self.theta)) - 1 + u1**self.theta)**(-1/self.theta)
        
        return np.column_stack([u1, u2])

class GumbelCopula(BaseCopula):
    """Gumbel Copula (Archimedean family)"""
    
    def __init__(self, dimension: int = 2):
        super().__init__()
        self.dimension = dimension
        self.theta = None
        
    def fit(self, data: np.ndarray) -> 'GumbelCopula':
        """Fit Gumbel copula"""
        if self.dimension != 2:
            raise NotImplementedError("Gumbel copula only implemented for bivariate case")
            
        # Method of moments using Kendall's tau
        tau = stats.kendalltau(data[:, 0], data[:, 1])[0]
        
        # Relationship: tau = 1 - 1/theta
        if tau > 0:
            self.theta = 1 / (1 - tau)
        else:
            self.theta = 1.1  # Minimum value for Gumbel
            
        # Ensure theta >= 1
        self.theta = max(self.theta, 1.001)
        
        # Calculate log-likelihood
        self.log_likelihood = self._calculate_log_likelihood(data)
        
        self.parameters = {'theta': self.theta}
        self.fitted = True
        
        logger.info(f"Fitted Gumbel copula: theta={self.theta:.4f}")
        return self
        
    def _calculate_log_likelihood(self, data: np.ndarray) -> float:
        """Calculate log-likelihood for Gumbel copula"""
        u, v = data[:, 0], data[:, 1]
        
        # Clip to avoid numerical issues
        u = np.clip(u, 1e-6, 1-1e-6)
        v = np.clip(v, 1e-6, 1-1e-6)
        
        ll = 0.0
        for i in range(len(u)):
            ui, vi = u[i], v[i]
            
            # Gumbel copula components
            A = (-np.log(ui))**self.theta + (-np.log(vi))**self.theta
            
            # Log-density
            term1 = -(A**(1/self.theta))
            term2 = np.log(A**(1/self.theta - 2))
            term3 = np.log((-np.log(ui))**(-1) * (-np.log(vi))**(-1))
            term4 = np.log((-np.log(ui))**(self.theta - 1) + (-np.log(vi))**(self.theta - 1))
            term5 = np.log(1 + (self.theta - 1) * A**(-1/self.theta))
            
            ll += term1 + term2 + term3 + term4 + term5
            
        return ll
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from Gumbel copula"""
        if not self.fitted:
            raise ValueError("Copula must be fitted before sampling")
            
        # Generate samples using Archimedean generator
        # This is a simplified implementation
        u = np.random.uniform(0, 1, (n_samples, 2))
        
        # Apply Gumbel transformation (simplified)
        # Full implementation would use proper inverse generator
        return u

class ArchimedeanCopula(BaseCopula):
    """General Archimedean Copula framework"""
    
    def __init__(self, family: str, dimension: int = 2):
        super().__init__()
        self.family = family
        self.dimension = dimension
        
        if family == 'clayton':
            self.copula = ClaytonCopula(dimension)
        elif family == 'gumbel':
            self.copula = GumbelCopula(dimension)
        else:
            raise ValueError(f"Unknown Archimedean family: {family}")
            
    def fit(self, data: np.ndarray) -> 'ArchimedeanCopula':
        """Fit Archimedean copula"""
        self.copula.fit(data)
        self.fitted = True
        self.parameters = self.copula.parameters
        self.log_likelihood = self.copula.log_likelihood
        return self
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples"""
        return self.copula.sample(n_samples)

class CopulaModels:
    """Main copula modeling framework"""
    
    def __init__(self):
        self.fitted_copulas = {}
        self.model_selection_results = {}
        
    def fit_copula(
        self,
        data: np.ndarray,
        copula_type: str,
        name: str = None
    ) -> BaseCopula:
        """
        Fit specified copula to data
        
        Args:
            data: Uniform data in [0,1]^d
            copula_type: Type of copula ('gaussian', 't', 'clayton', 'gumbel')
            name: Optional name for the fitted copula
            
        Returns:
            Fitted copula object
        """
        dimension = data.shape[1]
        
        if copula_type == 'gaussian':
            copula = GaussianCopula(dimension)
        elif copula_type == 't':
            copula = TCopula(dimension)
        elif copula_type == 'clayton':
            copula = ClaytonCopula(dimension)
        elif copula_type == 'gumbel':
            copula = GumbelCopula(dimension)
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")
            
        fitted_copula = copula.fit(data)
        
        # Store fitted copula
        copula_name = name or f"{copula_type}_{dimension}d"
        self.fitted_copulas[copula_name] = fitted_copula
        
        return fitted_copula
        
    def model_selection(
        self,
        data: np.ndarray,
        copula_types: List[str] = ['gaussian', 't', 'clayton', 'gumbel']
    ) -> Dict[str, Any]:
        """
        Perform copula model selection using information criteria
        
        Args:
            data: Uniform data
            copula_types: List of copula types to compare
            
        Returns:
            Model selection results
        """
        results = {}
        
        for copula_type in copula_types:
            try:
                logger.info(f"Fitting {copula_type} copula for model selection")
                copula = self.fit_copula(data, copula_type, f"selection_{copula_type}")
                
                results[copula_type] = {
                    'log_likelihood': copula.log_likelihood,
                    'aic': copula.aic(),
                    'bic': copula.bic(len(data)),
                    'parameters': copula.parameters
                }
                
            except Exception as e:
                logger.warning(f"Failed to fit {copula_type} copula: {e}")
                results[copula_type] = {
                    'log_likelihood': -np.inf,
                    'aic': np.inf,
                    'bic': np.inf,
                    'error': str(e)
                }
                
        # Find best model by AIC and BIC
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_models:
            best_aic = min(valid_models.keys(), key=lambda k: valid_models[k]['aic'])
            best_bic = min(valid_models.keys(), key=lambda k: valid_models[k]['bic'])
            
            selection_summary = {
                'best_aic': best_aic,
                'best_bic': best_bic,
                'results': results
            }
        else:
            selection_summary = {
                'best_aic': None,
                'best_bic': None,
                'results': results
            }
            
        self.model_selection_results = selection_summary
        return selection_summary
        
    def transform_to_uniform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform data to uniform margins using empirical CDF
        
        Args:
            data: Original data
            
        Returns:
            Uniform data in [0,1]^d
        """
        uniform_data = np.zeros_like(data.values)
        
        for i, column in enumerate(data.columns):
            # Empirical CDF transformation
            sorted_values = np.sort(data[column].values)
            ranks = np.searchsorted(sorted_values, data[column].values, side='right')
            uniform_data[:, i] = ranks / (len(data) + 1)
            
        return uniform_data
        
    def generate_scenarios(
        self,
        copula_name: str,
        marginal_distributions: List[stats.rv_continuous],
        n_scenarios: int = 1000
    ) -> np.ndarray:
        """
        Generate scenarios using fitted copula and marginal distributions
        
        Args:
            copula_name: Name of fitted copula to use
            marginal_distributions: List of marginal distributions
            n_scenarios: Number of scenarios to generate
            
        Returns:
            Generated scenarios
        """
        if copula_name not in self.fitted_copulas:
            raise ValueError(f"Copula '{copula_name}' not found")
            
        copula = self.fitted_copulas[copula_name]
        
        # Generate uniform samples from copula
        uniform_samples = copula.sample(n_scenarios)
        
        # Transform using marginal distributions
        scenarios = np.zeros_like(uniform_samples)
        
        for i, dist in enumerate(marginal_distributions):
            scenarios[:, i] = dist.ppf(uniform_samples[:, i])
            
        return scenarios
        
    def estimate_tail_dependence(self, data: np.ndarray) -> Dict[str, float]:
        """
        Estimate tail dependence coefficients
        
        Args:
            data: Uniform data
            
        Returns:
            Tail dependence estimates
        """
        if data.shape[1] != 2:
            raise ValueError("Tail dependence only implemented for bivariate data")
            
        u, v = data[:, 0], data[:, 1]
        
        # Upper tail dependence: lambda_U = lim_{t->1-} P(V > t | U > t)
        thresholds = np.linspace(0.9, 0.99, 10)
        upper_estimates = []
        
        for t in thresholds:
            mask_u = u > t
            if np.sum(mask_u) > 0:
                cond_prob = np.mean(v[mask_u] > t)
                upper_estimates.append(cond_prob)
                
        # Lower tail dependence: lambda_L = lim_{t->0+} P(V <= t | U <= t)  
        thresholds_lower = np.linspace(0.01, 0.1, 10)
        lower_estimates = []
        
        for t in thresholds_lower:
            mask_u = u <= t
            if np.sum(mask_u) > 0:
                cond_prob = np.mean(v[mask_u] <= t)
                lower_estimates.append(cond_prob)
                
        return {
            'upper_tail_dependence': np.mean(upper_estimates) if upper_estimates else 0.0,
            'lower_tail_dependence': np.mean(lower_estimates) if lower_estimates else 0.0
        }
        
    def goodness_of_fit_test(
        self,
        data: np.ndarray,
        copula_name: str,
        n_bootstrap: int = 100
    ) -> Dict[str, Any]:
        """
        Goodness-of-fit test using bootstrap
        
        Args:
            data: Uniform data used for fitting
            copula_name: Name of fitted copula to test
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Test results
        """
        if copula_name not in self.fitted_copulas:
            raise ValueError(f"Copula '{copula_name}' not found")
            
        copula = self.fitted_copulas[copula_name]
        
        # Calculate test statistic (simplified Cramér-von Mises)
        def test_statistic(data_sample):
            n = len(data_sample)
            stat = 0.0
            
            for i in range(n):
                empirical_cdf = np.mean(np.all(data_sample <= data_sample[i], axis=1))
                theoretical_cdf = copula.cdf(data_sample[i:i+1])[0] if hasattr(copula, 'cdf') else 0.5
                stat += (empirical_cdf - theoretical_cdf)**2
                
            return stat / n
            
        # Observed test statistic
        observed_stat = test_statistic(data)
        
        # Bootstrap distribution
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = copula.sample(len(data))
                bootstrap_stat = test_statistic(bootstrap_sample)
                bootstrap_stats.append(bootstrap_stat)
            except:
                continue
                
        if bootstrap_stats:
            p_value = np.mean(np.array(bootstrap_stats) >= observed_stat)
        else:
            p_value = np.nan
            
        return {
            'test_statistic': observed_stat,
            'p_value': p_value,
            'bootstrap_statistics': bootstrap_stats
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Copula Models...")
    
    # Generate synthetic data with known dependence
    np.random.seed(42)
    n_samples = 1000
    
    # Generate correlated normal data
    rho = 0.7
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    normal_data = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Transform to uniform using normal CDF
    uniform_data = stats.norm.cdf(normal_data)
    
    print(f"Generated {n_samples} samples with correlation {rho}")
    
    # Initialize copula models
    copula_models = CopulaModels()
    
    # Test individual copulas
    print("\nTesting individual copulas:")
    
    # Gaussian copula
    gaussian_copula = copula_models.fit_copula(uniform_data, 'gaussian', 'test_gaussian')
    print(f"Gaussian copula fitted: correlation = {gaussian_copula.correlation_matrix[0,1]:.3f}")
    
    # t-copula
    t_copula = copula_models.fit_copula(uniform_data, 't', 'test_t')
    print(f"t-copula fitted: nu = {t_copula.degrees_of_freedom:.2f}")
    
    # Clayton copula
    clayton_copula = copula_models.fit_copula(uniform_data, 'clayton', 'test_clayton')
    print(f"Clayton copula fitted: theta = {clayton_copula.theta:.3f}")
    
    # Model selection
    print("\nPerforming model selection:")
    selection_results = copula_models.model_selection(uniform_data)
    
    print(f"Best model by AIC: {selection_results['best_aic']}")
    print(f"Best model by BIC: {selection_results['best_bic']}")
    
    # Show AIC and BIC values
    for model, results in selection_results['results'].items():
        if 'error' not in results:
            print(f"  {model}: AIC={results['aic']:.2f}, BIC={results['bic']:.2f}")
    
    # Test sampling
    print("\nTesting sampling:")
    samples = gaussian_copula.sample(100)
    print(f"Generated {len(samples)} samples from Gaussian copula")
    print(f"Sample correlation: {np.corrcoef(samples.T)[0,1]:.3f}")
    
    # Tail dependence estimation
    print("\nEstimating tail dependence:")
    tail_dep = copula_models.estimate_tail_dependence(uniform_data)
    print(f"Upper tail dependence: {tail_dep['upper_tail_dependence']:.3f}")
    print(f"Lower tail dependence: {tail_dep['lower_tail_dependence']:.3f}")
    
    # Scenario generation
    print("\nGenerating scenarios with marginal distributions:")
    marginal_dists = [stats.norm(0, 1), stats.norm(0, 1)]  # Standard normal margins
    scenarios = copula_models.generate_scenarios('test_gaussian', marginal_dists, 100)
    print(f"Generated {len(scenarios)} scenarios")
    print(f"Scenario correlation: {np.corrcoef(scenarios.T)[0,1]:.3f}")
    
    print("\nCopula models test completed!")
