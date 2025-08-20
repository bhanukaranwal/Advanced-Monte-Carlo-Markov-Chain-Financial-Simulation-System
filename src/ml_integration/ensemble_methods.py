"""
Ensemble methods for combining multiple models and predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.optimize as optimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger(__name__)

class BaseEnsemble(BaseEstimator, RegressorMixin):
    """Base class for ensemble methods"""
    
    def __init__(self):
        self.models = []
        self.weights = None
        self.fitted = False
        
    def add_model(self, model, name: str = None):
        """Add a model to the ensemble"""
        if name is None:
            name = f"model_{len(self.models)}"
        self.models.append({'model': model, 'name': name})
        
    def fit(self, X, y):
        """Fit all models in ensemble"""
        for model_info in self.models:
            model_info['model'].fit(X, y)
        self.fitted = True
        return self
        
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = np.array([
            model_info['model'].predict(X) 
            for model_info in self.models
        ]).T
        
        if self.weights is None:
            # Simple average
            return np.mean(predictions, axis=1)
        else:
            # Weighted average
            return np.average(predictions, axis=1, weights=self.weights)
            
    def get_individual_predictions(self, X):
        """Get predictions from individual models"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
            
        predictions = {}
        for model_info in self.models:
            predictions[model_info['name']] = model_info['model'].predict(X)
            
        return predictions

class ModelStacking(BaseEnsemble):
    """Stacked generalization ensemble"""
    
    def __init__(
        self, 
        meta_learner=None,
        cv_folds: int = 5,
        use_original_features: bool = True
    ):
        super().__init__()
        self.meta_learner = meta_learner or LinearRegression()
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features
        self.meta_fitted = False
        
    def fit(self, X, y):
        """Fit stacked ensemble"""
        logger.info("Training stacked ensemble...")
        
        # Fit base models
        super().fit(X, y)
        
        # Generate out-of-fold predictions for meta-learner
        meta_features = self._generate_meta_features(X, y)
        
        # Include original features if specified
        if self.use_original_features:
            meta_features = np.column_stack([X, meta_features])
            
        # Fit meta-learner
        self.meta_learner.fit(meta_features, y)
        self.meta_fitted = True
        
        logger.info("Stacked ensemble training completed")
        return self
        
    def _generate_meta_features(self, X, y):
        """Generate meta-features using cross-validation"""
        n_models = len(self.models)
        meta_features = np.zeros((len(X), n_models))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Train models on fold training data
            for i, model_info in enumerate(self.models):
                # Clone model for this fold
                fold_model = type(model_info['model'])(**model_info['model'].get_params())
                fold_model.fit(X_train, y_train)
                
                # Predict on validation set
                val_pred = fold_model.predict(X_val)
                meta_features[val_idx, i] = val_pred
                
        return meta_features
        
    def predict(self, X):
        """Make stacked predictions"""
        if not self.fitted or not self.meta_fitted:
            raise ValueError("Stacked ensemble must be fitted before prediction")
            
        # Get base model predictions
        base_predictions = np.array([
            model_info['model'].predict(X) 
            for model_info in self.models
        ]).T
        
        # Create meta-features
        meta_features = base_predictions
        if self.use_original_features:
            meta_features = np.column_stack([X, meta_features])
            
        # Meta-learner prediction
        return self.meta_learner.predict(meta_features)
        
    def get_meta_feature_importance(self):
        """Get feature importance from meta-learner"""
        if hasattr(self.meta_learner, 'feature_importances_'):
            return self.meta_learner.feature_importances_
        elif hasattr(self.meta_learner, 'coef_'):
            return np.abs(self.meta_learner.coef_)
        else:
            return None

class BayesianModelAveraging(BaseEnsemble):
    """Bayesian Model Averaging ensemble"""
    
    def __init__(self, prior_weights=None, likelihood_method='gaussian'):
        super().__init__()
        self.prior_weights = prior_weights
        self.likelihood_method = likelihood_method
        self.posterior_weights = None
        self.model_likelihoods = None
        
    def fit(self, X, y):
        """Fit BMA ensemble"""
        logger.info("Training Bayesian Model Averaging ensemble...")
        
        # Fit base models
        super().fit(X, y)
        
        # Calculate model likelihoods and posterior weights
        self._calculate_model_weights(X, y)
        
        logger.info("BMA ensemble training completed")
        return self
        
    def _calculate_model_weights(self, X, y):
        """Calculate Bayesian model weights"""
        n_models = len(self.models)
        
        # Set uniform prior if not specified
        if self.prior_weights is None:
            self.prior_weights = np.ones(n_models) / n_models
            
        # Calculate likelihoods using cross-validation
        self.model_likelihoods = np.zeros(n_models)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for i, model_info in enumerate(self.models):
            log_likelihoods = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone and fit model
                fold_model = type(model_info['model'])(**model_info['model'].get_params())
                fold_model.fit(X_train, y_train)
                
                # Calculate likelihood
                y_pred = fold_model.predict(X_val)
                
                if self.likelihood_method == 'gaussian':
                    # Gaussian likelihood
                    mse = mean_squared_error(y_val, y_pred)
                    sigma = np.sqrt(mse)
                    log_likelihood = np.sum(norm.logpdf(y_val, y_pred, sigma))
                else:
                    # Laplace likelihood (MAE-based)
                    mae = mean_absolute_error(y_val, y_pred)
                    b = mae  # Scale parameter
                    log_likelihood = np.sum(-np.log(2 * b) - np.abs(y_val - y_pred) / b)
                    
                log_likelihoods.append(log_likelihood)
                
            # Average log-likelihood across folds
            self.model_likelihoods[i] = np.mean(log_likelihoods)
            
        # Calculate posterior weights (up to normalization constant)
        log_posterior = np.log(self.prior_weights) + self.model_likelihoods
        
        # Normalize to get probabilities (softmax)
        max_log_posterior = np.max(log_posterior)
        exp_log_posterior = np.exp(log_posterior - max_log_posterior)
        self.posterior_weights = exp_log_posterior / np.sum(exp_log_posterior)
        
        self.weights = self.posterior_weights
        
    def predict(self, X):
        """Make BMA predictions"""
        if not self.fitted or self.posterior_weights is None:
            raise ValueError("BMA ensemble must be fitted before prediction")
            
        return super().predict(X)
        
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty estimates"""
        if not self.fitted:
            raise ValueError("BMA ensemble must be fitted before prediction")
            
        # Individual model predictions
        predictions = np.array([
            model_info['model'].predict(X) 
            for model_info in self.models
        ]).T
        
        # Weighted mean (BMA prediction)
        bma_prediction = np.average(predictions, axis=1, weights=self.posterior_weights)
        
        # BMA variance (including model uncertainty)
        weighted_variance = np.average(
            (predictions - bma_prediction.reshape(-1, 1))**2, 
            axis=1, 
            weights=self.posterior_weights
        )
        
        return bma_prediction, np.sqrt(weighted_variance)
        
    def get_model_probabilities(self):
        """Get posterior model probabilities"""
        return dict(zip(
            [model_info['name'] for model_info in self.models],
            self.posterior_weights
        ))

class AdaptiveEnsemble(BaseEnsemble):
    """Ensemble with adaptive weights based on recent performance"""
    
    def __init__(self, window_size: int = 100, learning_rate: float = 0.01):
        super().__init__()
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.performance_history = []
        self.adaptive_weights = None
        
    def fit(self, X, y):
        """Fit adaptive ensemble"""
        super().fit(X, y)
        
        # Initialize equal weights
        n_models = len(self.models)
        self.adaptive_weights = np.ones(n_models) / n_models
        
        return self
        
    def partial_fit(self, X, y):
        """Update ensemble with new data"""
        if not self.fitted:
            return self.fit(X, y)
            
        # Get predictions from all models
        predictions = np.array([
            model_info['model'].predict(X) 
            for model_info in self.models
        ]).T
        
        # Calculate individual model errors
        model_errors = np.array([
            mean_squared_error(y, pred) for pred in predictions.T
        ])
        
        # Update performance history
        self.performance_history.append(model_errors)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
            
        # Update adaptive weights
        self._update_weights()
        
        return self
        
    def _update_weights(self):
        """Update weights based on recent performance"""
        if len(self.performance_history) == 0:
            return
            
        # Calculate recent average performance
        recent_errors = np.array(self.performance_history)
        avg_errors = np.mean(recent_errors, axis=0)
        
        # Convert errors to weights (inverse relationship)
        # Add small constant to avoid division by zero
        inv_errors = 1.0 / (avg_errors + 1e-8)
        new_weights = inv_errors / np.sum(inv_errors)
        
        # Exponential moving average update
        if self.adaptive_weights is not None:
            self.adaptive_weights = (
                (1 - self.learning_rate) * self.adaptive_weights + 
                self.learning_rate * new_weights
            )
        else:
            self.adaptive_weights = new_weights
            
        self.weights = self.adaptive_weights

class EnsemblePredictor:
    """Main ensemble predictor orchestrator"""
    
    def __init__(self):
        self.ensembles = {}
        self.base_models = {}
        
    def create_base_models(self, model_configs: List[Dict]) -> Dict[str, Any]:
        """Create base models from configurations"""
        models = {}
        
        for config in model_configs:
            model_type = config['type']
            model_name = config.get('name', model_type)
            model_params = config.get('params', {})
            
            if model_type == 'linear_regression':
                model = LinearRegression(**model_params)
            elif model_type == 'ridge':
                model = Ridge(**model_params)
            elif model_type == 'lasso':
                model = Lasso(**model_params)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(**model_params)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(**model_params)
            elif model_type == 'svm':
                model = SVR(**model_params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            models[model_name] = model
            
        self.base_models = models
        return models
        
    def create_ensemble(
        self,
        ensemble_type: str,
        models: List[Any],
        ensemble_name: str = None,
        **kwargs
    ):
        """Create ensemble from models"""
        if ensemble_name is None:
            ensemble_name = f"{ensemble_type}_{len(self.ensembles)}"
            
        if ensemble_type == 'stacking':
            ensemble = ModelStacking(**kwargs)
        elif ensemble_type == 'bma':
            ensemble = BayesianModelAveraging(**kwargs)
        elif ensemble_type == 'adaptive':
            ensemble = AdaptiveEnsemble(**kwargs)
        elif ensemble_type == 'simple':
            ensemble = BaseEnsemble()
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
            
        # Add models to ensemble
        for i, model in enumerate(models):
            model_name = getattr(model, 'name', f'model_{i}')
            ensemble.add_model(model, model_name)
            
        self.ensembles[ensemble_name] = ensemble
        return ensemble
        
    def train_ensemble(
        self,
        ensemble_name: str,
        X: np.ndarray,
        y: np.ndarray
    ):
        """Train specific ensemble"""
        if ensemble_name not in self.ensembles:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
            
        ensemble = self.ensembles[ensemble_name]
        ensemble.fit(X, y)
        
        logger.info(f"Ensemble '{ensemble_name}' trained successfully")
        
    def predict_ensemble(
        self,
        ensemble_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """Make predictions using specific ensemble"""
        if ensemble_name not in self.ensembles:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
            
        ensemble = self.ensembles[ensemble_name]
        return ensemble.predict(X)
        
    def compare_ensembles(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Compare performance of all ensembles"""
        results = []
        
        for name, ensemble in self.ensembles.items():
            try:
                predictions = ensemble.predict(X_test)
                
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                results.append({
                    'ensemble': name,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                })
                
            except Exception as e:
                logger.error(f"Error evaluating ensemble {name}: {e}")
                results.append({
                    'ensemble': name,
                    'mse': np.nan,
                    'mae': np.nan,
                    'r2': np.nan,
                    'rmse': np.nan
                })
                
        return pd.DataFrame(results).set_index('ensemble')
        
    def get_ensemble_summary(self) -> pd.DataFrame:
        """Get summary of all ensembles"""
        summary = []
        
        for name, ensemble in self.ensembles.items():
            n_models = len(ensemble.models)
            ensemble_type = type(ensemble).__name__
            fitted = ensemble.fitted
            
            model_types = [type(model_info['model']).__name__ 
                          for model_info in ensemble.models]
            
            summary.append({
                'name': name,
                'type': ensemble_type,
                'n_models': n_models,
                'fitted': fitted,
                'model_types': ', '.join(set(model_types))
            })
            
        return pd.DataFrame(summary).set_index('name')

# Example usage and testing
if __name__ == "__main__":
    print("Testing Ensemble Methods...")
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    
    # Non-linear target with noise
    y = (X[:, 0]**2 + X[:, 1] * X[:, 2] + 
         np.sin(X[:, 3]) + 0.5 * X[:, 4] + 
         np.random.normal(0, 0.1, n_samples))
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Generated {n_samples} samples with {n_features} features")
    
    # Initialize ensemble predictor
    ensemble_predictor = EnsemblePredictor()
    
    # Create base models
    model_configs = [
        {'type': 'linear_regression', 'name': 'linear'},
        {'type': 'ridge', 'name': 'ridge', 'params': {'alpha': 1.0}},
        {'type': 'random_forest', 'name': 'rf', 'params': {'n_estimators': 50, 'random_state': 42}},
        {'type': 'gradient_boosting', 'name': 'gb', 'params': {'n_estimators': 50, 'random_state': 42}}
    ]
    
    base_models = ensemble_predictor.create_base_models(model_configs)
    print(f"Created {len(base_models)} base models")
    
    # Create different types of ensembles
    models_list = list(base_models.values())
    
    # Simple averaging ensemble
    simple_ensemble = ensemble_predictor.create_ensemble(
        'simple', models_list, 'simple_average'
    )
    
    # Stacking ensemble
    stacking_ensemble = ensemble_predictor.create_ensemble(
        'stacking', models_list, 'stacked', 
        meta_learner=Ridge(alpha=1.0), cv_folds=5
    )
    
    # Bayesian Model Averaging
    bma_ensemble = ensemble_predictor.create_ensemble(
        'bma', models_list, 'bayesian_avg'
    )
    
    # Adaptive ensemble
    adaptive_ensemble = ensemble_predictor.create_ensemble(
        'adaptive', models_list, 'adaptive_weights',
        window_size=50, learning_rate=0.05
    )
    
    print("Created 4 different ensemble types")
    
    # Train all ensembles
    for ensemble_name in ensemble_predictor.ensembles.keys():
        print(f"Training {ensemble_name} ensemble...")
        ensemble_predictor.train_ensemble(ensemble_name, X_train, y_train)
    
    # Compare performance
    print("\nComparing ensemble performance:")
    comparison = ensemble_predictor.compare_ensembles(X_test, y_test)
    print(comparison.round(4))
    
    # Test individual ensemble features
    print("\nTesting BMA uncertainty estimation:")
    bma_pred, bma_std = bma_ensemble.predict_with_uncertainty(X_test[:10])
    print("BMA predictions with std:", list(zip(bma_pred.round(3), bma_std.round(3))))
    
    # Model probabilities in BMA
    print("\nBMA model probabilities:")
    model_probs = bma_ensemble.get_model_probabilities()
    for model, prob in model_probs.items():
        print(f"  {model}: {prob:.3f}")
    
    # Test adaptive ensemble updates
    print("\nTesting adaptive ensemble updates:")
    print("Initial weights:", adaptive_ensemble.adaptive_weights.round(3))
    
    # Simulate online learning
    for i in range(0, len(X_test), 20):
        batch_X = X_test[i:i+20]
        batch_y = y_test[i:i+20]
        if len(batch_X) > 0:
            adaptive_ensemble.partial_fit(batch_X, batch_y)
            
    print("Final adaptive weights:", adaptive_ensemble.adaptive_weights.round(3))
    
    # Get ensemble summary
    print("\nEnsemble Summary:")
    summary = ensemble_predictor.get_ensemble_summary()
    print(summary)
    
    print("\nEnsemble methods test completed!")
