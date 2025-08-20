"""
Kalman filters for real-time state estimation in financial markets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from scipy.linalg import inv, cholesky
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

@dataclass
class KalmanState:
    """Kalman filter state"""
    x: np.ndarray  # State vector
    P: np.ndarray  # State covariance matrix
    timestamp: float
    log_likelihood: float = 0.0

class KalmanFilter:
    """Standard linear Kalman filter for financial time series"""
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        F: Optional[np.ndarray] = None,  # State transition matrix
        H: Optional[np.ndarray] = None,  # Observation matrix
        Q: Optional[np.ndarray] = None,  # Process noise covariance
        R: Optional[np.ndarray] = None,  # Observation noise covariance
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # State transition matrix (default: random walk)
        self.F = F if F is not None else np.eye(state_dim)
        
        # Observation matrix (default: observe all states)
        self.H = H if H is not None else np.eye(obs_dim, state_dim)
        
        # Process noise covariance (default: small diagonal)
        self.Q = Q if Q is not None else np.eye(state_dim) * 0.01
        
        # Observation noise covariance (default: diagonal)
        self.R = R if R is not None else np.eye(obs_dim) * 0.1
        
        # Initialize state
        self.state = initial_state if initial_state is not None else np.zeros(state_dim)
        self.covariance = initial_covariance if initial_covariance is not None else np.eye(state_dim)
        
        # Filter history
        self.state_history = []
        self.covariance_history = []
        self.innovation_history = []
        self.log_likelihood_history = []
        
        # Current timestamp
        self.current_time = 0.0
        
    def predict(self, dt: float = 1.0) -> KalmanState:
        """
        Prediction step of Kalman filter
        
        Args:
            dt: Time step size
            
        Returns:
            Predicted state
        """
        # Adjust state transition matrix for time step
        F_dt = self._adjust_transition_matrix(dt)
        Q_dt = self._adjust_process_noise(dt)
        
        # Predict state and covariance
        predicted_state = F_dt @ self.state
        predicted_covariance = F_dt @ self.covariance @ F_dt.T + Q_dt
        
        # Update internal state (prediction becomes prior for update)
        self.state = predicted_state
        self.covariance = predicted_covariance
        self.current_time += dt
        
        return KalmanState(
            x=predicted_state.copy(),
            P=predicted_covariance.copy(),
            timestamp=self.current_time
        )
        
    def update(self, observation: np.ndarray, R_obs: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update step of Kalman filter
        
        Args:
            observation: Observation vector
            R_obs: Observation noise covariance for this observation
            
        Returns:
            Updated state
        """
        R_current = R_obs if R_obs is not None else self.R
        
        # Innovation (residual)
        innovation = observation - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.covariance @ self.H.T + R_current
        
        # Kalman gain
        try:
            S_inv = inv(S)
            K = self.covariance @ self.H.T @ S_inv
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance matrix, using pseudoinverse")
            S_inv = np.linalg.pinv(S)
            K = self.covariance @ self.H.T @ S_inv
            
        # Update state and covariance
        self.state = self.state + K @ innovation
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R_current @ K.T
        
        # Calculate log-likelihood
        try:
            log_likelihood = -0.5 * (
                len(observation) * np.log(2 * np.pi) +
                np.log(np.linalg.det(S)) +
                innovation.T @ S_inv @ innovation
            )
        except:
            log_likelihood = 0.0
            
        # Store history
        self.state_history.append(self.state.copy())
        self.covariance_history.append(self.covariance.copy())
        self.innovation_history.append(innovation)
        self.log_likelihood_history.append(log_likelihood)
        
        return KalmanState(
            x=self.state.copy(),
            P=self.covariance.copy(),
            timestamp=self.current_time,
            log_likelihood=log_likelihood
        )
        
    def predict_and_update(
        self, 
        observation: np.ndarray, 
        dt: float = 1.0,
        R_obs: Optional[np.ndarray] = None
    ) -> KalmanState:
        """Combined predict and update step"""
        self.predict(dt)
        return self.update(observation, R_obs)
        
    def _adjust_transition_matrix(self, dt: float) -> np.ndarray:
        """Adjust state transition matrix for time step"""
        # For linear systems, this might involve matrix exponential
        # For simple case, we'll scale the off-diagonal elements
        if dt == 1.0:
            return self.F
            
        # Simple scaling (more sophisticated methods would use matrix exponential)
        F_dt = self.F.copy()
        # Scale off-diagonal elements by dt
        mask = ~np.eye(self.state_dim, dtype=bool)
        F_dt[mask] *= dt
        
        return F_dt
        
    def _adjust_process_noise(self, dt: float) -> np.ndarray:
        """Adjust process noise covariance for time step"""
        # Process noise typically scales with time
        return self.Q * dt
        
    def get_filtered_states(self) -> np.ndarray:
        """Get all filtered states"""
        return np.array(self.state_history) if self.state_history else np.array([])
        
    def get_innovations(self) -> np.ndarray:
        """Get innovation sequence"""
        return np.array(self.innovation_history) if self.innovation_history else np.array([])
        
    def get_log_likelihood(self) -> float:
        """Get total log-likelihood"""
        return sum(self.log_likelihood_history)
        
    def smooth(self) -> List[KalmanState]:
        """Rauch-Tung-Striebel smoother"""
        if not self.state_history:
            return []
            
        n = len(self.state_history)
        smoothed_states = [None] * n
        
        # Initialize with last filtered estimate
        smoothed_states[-1] = KalmanState(
            x=self.state_history[-1].copy(),
            P=self.covariance_history[-1].copy(),
            timestamp=self.current_time
        )
        
        # Backward pass
        for k in range(n - 2, -1, -1):
            # Predict one step ahead from k
            F_dt = self.F  # Assuming dt = 1 for simplicity
            Q_dt = self.Q
            
            x_pred = F_dt @ self.state_history[k]
            P_pred = F_dt @ self.covariance_history[k] @ F_dt.T + Q_dt
            
            # Smoother gain
            try:
                A = self.covariance_history[k] @ F_dt.T @ inv(P_pred)
            except np.linalg.LinAlgError:
                A = self.covariance_history[k] @ F_dt.T @ np.linalg.pinv(P_pred)
                
            # Smoothed estimates
            x_smooth = self.state_history[k] + A @ (smoothed_states[k + 1].x - x_pred)
            P_smooth = self.covariance_history[k] + A @ (smoothed_states[k + 1].P - P_pred) @ A.T
            
            smoothed_states[k] = KalmanState(
                x=x_smooth,
                P=P_smooth,
                timestamp=k  # Simplified timestamp
            )
            
        return smoothed_states

class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems"""
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        f_func: Callable,  # State transition function
        h_func: Callable,  # Observation function
        F_jacobian: Callable,  # Jacobian of f_func
        H_jacobian: Callable,  # Jacobian of h_func
        Q: np.ndarray,
        R: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f_func = f_func
        self.h_func = h_func
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
        
        # Initialize state
        self.state = initial_state if initial_state is not None else np.zeros(state_dim)
        self.covariance = initial_covariance if initial_covariance is not None else np.eye(state_dim)
        
        # History
        self.state_history = []
        self.covariance_history = []
        self.current_time = 0.0
        
    def predict(self, dt: float = 1.0, control: Optional[np.ndarray] = None) -> KalmanState:
        """EKF prediction step"""
        # Nonlinear state prediction
        predicted_state = self.f_func(self.state, dt, control)
        
        # Linearize around current state
        F = self.F_jacobian(self.state, dt, control)
        
        # Predicted covariance
        predicted_covariance = F @ self.covariance @ F.T + self.Q * dt
        
        # Update internal state
        self.state = predicted_state
        self.covariance = predicted_covariance
        self.current_time += dt
        
        return KalmanState(
            x=predicted_state.copy(),
            P=predicted_covariance.copy(),
            timestamp=self.current_time
        )
        
    def update(self, observation: np.ndarray) -> KalmanState:
        """EKF update step"""
        # Nonlinear observation prediction
        h = self.h_func(self.state)
        
        # Linearize observation function
        H = self.H_jacobian(self.state)
        
        # Innovation
        innovation = observation - h
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R
        
        # Kalman gain
        try:
            K = self.covariance @ H.T @ inv(S)
        except np.linalg.LinAlgError:
            K = self.covariance @ H.T @ np.linalg.pinv(S)
            
        # Update
        self.state = self.state + K @ innovation
        I_KH = np.eye(self.state_dim) - K @ H
        self.covariance = I_KH @ self.covariance
        
        # Store history
        self.state_history.append(self.state.copy())
        self.covariance_history.append(self.covariance.copy())
        
        return KalmanState(
            x=self.state.copy(),
            P=self.covariance.copy(),
            timestamp=self.current_time
        )

class UnscentedKalmanFilter:
    """Unscented Kalman Filter for highly nonlinear systems"""
    
    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        f_func: Callable,
        h_func: Callable,
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 0.001,
        beta: float = 2.0,
        kappa: float = 0.0,
        initial_state: Optional[np.ndarray] = None,
        initial_covariance: Optional[np.ndarray] = None
    ):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.f_func = f_func
        self.h_func = h_func
        self.Q = Q
        self.R = R
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Derived parameters
        self.n = state_dim
        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        
        # Sigma point weights
        self.n_sigma = 2 * self.n + 1
        self.Wm = np.zeros(self.n_sigma)  # Weights for means
        self.Wc = np.zeros(self.n_sigma)  # Weights for covariance
        
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        
        for i in range(1, self.n_sigma):
            self.Wm[i] = 1 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1 / (2 * (self.n + self.lambda_))
            
        # Initialize state
        self.state = initial_state if initial_state is not None else np.zeros(state_dim)
        self.covariance = initial_covariance if initial_covariance is not None else np.eye(state_dim)
        
        # History
        self.state_history = []
        self.current_time = 0.0
        
    def _generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generate sigma points"""
        n = len(x)
        sigma_points = np.zeros((2 * n + 1, n))
        
        # First sigma point is the mean
        sigma_points[0] = x
        
        # Calculate square root of (n + lambda) * P
        try:
            sqrt = cholesky((n + self.lambda_) * P).T
        except np.linalg.LinAlgError:
            # Use SVD if Cholesky fails
            U, s, Vh = np.linalg.svd(P)
            sqrt = U @ np.diag(np.sqrt(s * (n + self.lambda_)))
            
        # Generate remaining sigma points
        for i in range(n):
            sigma_points[i + 1] = x + sqrt[i]
            sigma_points[n + i + 1] = x - sqrt[i]
            
        return sigma_points
        
    def predict(self, dt: float = 1.0) -> KalmanState:
        """UKF prediction step"""
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state, self.covariance)
        
        # Propagate sigma points through nonlinear function
        sigma_points_pred = np.array([
            self.f_func(sp, dt) for sp in sigma_points
        ])
        
        # Predicted mean
        x_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Predicted covariance
        P_pred = self.Q * dt
        for i in range(self.n_sigma):
            diff = sigma_points_pred[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
            
        # Update state
        self.state = x_pred
        self.covariance = P_pred
        self.current_time += dt
        
        return KalmanState(
            x=x_pred.copy(),
            P=P_pred.copy(),
            timestamp=self.current_time
        )
        
    def update(self, observation: np.ndarray) -> KalmanState:
        """UKF update step"""
        # Generate sigma points for update
        sigma_points = self._generate_sigma_points(self.state, self.covariance)
        
        # Transform sigma points through observation function
        gamma_points = np.array([
            self.h_func(sp) for sp in sigma_points
        ])
        
        # Predicted observation mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * gamma_points, axis=0)
        
        # Innovation covariance
        S = self.R.copy()
        for i in range(self.n_sigma):
            diff = gamma_points[i] - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
            
        # Cross-covariance
        T = np.zeros((self.state_dim, self.obs_dim))
        for i in range(self.n_sigma):
            x_diff = sigma_points[i] - self.state
            z_diff = gamma_points[i] - z_pred
            T += self.Wc[i] * np.outer(x_diff, z_diff)
            
        # Kalman gain
        try:
            K = T @ inv(S)
        except np.linalg.LinAlgError:
            K = T @ np.linalg.pinv(S)
            
        # Update
        innovation = observation - z_pred
        self.state = self.state + K @ innovation
        self.covariance = self.covariance - K @ S @ K.T
        
        # Store history
        self.state_history.append(self.state.copy())
        
        return KalmanState(
            x=self.state.copy(),
            P=self.covariance.copy(),
            timestamp=self.current_time
        )

class AdaptiveKalmanFilter(KalmanFilter):
    """Adaptive Kalman filter with online parameter estimation"""
    
    def __init__(self, *args, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.innovation_covariance_estimate = None
        
    def update(self, observation: np.ndarray, R_obs: Optional[np.ndarray] = None) -> KalmanState:
        """Adaptive update with covariance estimation"""
        # Standard update
        result = super().update(observation, R_obs)
        
        # Adaptive covariance estimation
        if len(self.innovation_history) > 1:
            innovation = self.innovation_history[-1]
            
            # Estimate innovation covariance
            H = self.H
            S_theoretical = H @ self.covariance @ H.T + self.R
            S_empirical = np.outer(innovation, innovation)
            
            # Adaptive update of R
            if self.innovation_covariance_estimate is None:
                self.innovation_covariance_estimate = S_empirical
            else:
                self.innovation_covariance_estimate = (
                    (1 - self.adaptation_rate) * self.innovation_covariance_estimate +
                    self.adaptation_rate * S_empirical
                )
                
            # Update observation noise covariance
            R_adaptive = self.innovation_covariance_estimate - H @ self.covariance @ H.T
            R_adaptive = np.maximum(R_adaptive, 0.001 * np.eye(self.obs_dim))  # Ensure positive definite
            
            self.R = R_adaptive
            
        return result

# Financial-specific Kalman filter implementations

class TrendFollowingKalman(KalmanFilter):
    """Kalman filter for trend-following in price series"""
    
    def __init__(self, process_noise_std: float = 0.01, obs_noise_std: float = 0.1):
        # State: [price, trend, acceleration]
        state_dim = 3
        obs_dim = 1
        
        # State transition matrix (constant acceleration model)
        F = np.array([
            [1, 1, 0.5],  # price += trend + 0.5*acceleration
            [0, 1, 1.0],  # trend += acceleration
            [0, 0, 1.0]   # acceleration persists
        ])
        
        # Observation matrix (observe price only)
        H = np.array([[1, 0, 0]])
        
        # Process noise
        Q = np.diag([0.1, process_noise_std, process_noise_std]) ** 2
        
        # Observation noise
        R = np.array([[obs_noise_std**2]])
        
        super().__init__(
            state_dim=state_dim,
            obs_dim=obs_dim,
            F=F, H=H, Q=Q, R=R,
            initial_state=np.array([100.0, 0.0, 0.0]),  # price=100, no trend, no acceleration
            initial_covariance=np.diag([1.0, 0.1, 0.01])
        )
        
    def get_price_estimate(self) -> float:
        """Get current price estimate"""
        return self.state[0]
        
    def get_trend_estimate(self) -> float:
        """Get current trend estimate"""
        return self.state[1]
        
    def get_acceleration_estimate(self) -> float:
        """Get current acceleration estimate"""
        return self.state[2]

class VolatilityKalman(KalmanFilter):
    """Kalman filter for volatility estimation"""
    
    def __init__(self, volatility_process_noise: float = 0.01):
        # State: [log_volatility]
        state_dim = 1
        obs_dim = 1
        
        # State transition (AR(1) for log volatility)
        F = np.array([[0.99]])  # Mean reversion
        
        # Observation matrix
        H = np.array([[1.0]])
        
        # Process noise
        Q = np.array([[volatility_process_noise**2]])
        
        # Observation noise (small for log volatility observations)
        R = np.array([[0.01]])
        
        super().__init__(
            state_dim=state_dim,
            obs_dim=obs_dim,
            F=F, H=H, Q=Q, R=R,
            initial_state=np.array([np.log(0.2)]),  # 20% initial volatility
            initial_covariance=np.array([[0.1]])
        )
        
    def get_volatility_estimate(self) -> float:
        """Get current volatility estimate"""
        return np.exp(self.state[0])
        
    def update_with_return(self, return_: float) -> KalmanState:
        """Update filter with return observation"""
        # Use squared return as proxy for instantaneous variance
        log_variance_obs = np.log(max(return_**2, 1e-8))
        return self.update(np.array([log_variance_obs]))

# Example usage and testing
if __name__ == "__main__":
    print("Testing Kalman Filters...")
    
    # Generate synthetic price data with trend and noise
    np.random.seed(42)
    n_points = 200
    true_trend = 0.1
    true_acceleration = 0.001
    noise_std = 1.0
    
    # Generate true prices
    true_prices = []
    current_price = 100.0
    current_trend = true_trend
    
    for i in range(n_points):
        current_price += current_trend + np.random.normal(0, noise_std)
        current_trend += true_acceleration + np.random.normal(0, 0.01)
        true_prices.append(current_price)
        
    # Observed prices with additional noise
    obs_noise = 0.5
    observed_prices = [p + np.random.normal(0, obs_noise) for p in true_prices]
    
    print(f"Generated {n_points} price observations")
    
    # Test Trend Following Kalman Filter
    print("\nTesting Trend Following Kalman Filter:")
    trend_kf = TrendFollowingKalman(process_noise_std=0.1, obs_noise_std=obs_noise)
    
    estimated_prices = []
    estimated_trends = []
    
    for price in observed_prices:
        state = trend_kf.predict_and_update(np.array([price]))
        estimated_prices.append(trend_kf.get_price_estimate())
        estimated_trends.append(trend_kf.get_trend_estimate())
        
    # Calculate errors
    price_errors = np.array(estimated_prices) - np.array(true_prices)
    price_mae = np.mean(np.abs(price_errors))
    price_rmse = np.sqrt(np.mean(price_errors**2))
    
    print(f"Price estimation MAE: {price_mae:.3f}")
    print(f"Price estimation RMSE: {price_rmse:.3f}")
    print(f"Final trend estimate: {estimated_trends[-1]:.4f} (true: {true_trend:.4f})")
    
    # Test Volatility Kalman Filter
    print("\nTesting Volatility Kalman Filter:")
    returns = np.diff(observed_prices) / np.array(observed_prices[:-1])
    vol_kf = VolatilityKalman(volatility_process_noise=0.01)
    
    estimated_volatilities = []
    
    for ret in returns:
        state = vol_kf.update_with_return(ret)
        estimated_volatilities.append(vol_kf.get_volatility_estimate())
        
    print(f"Final volatility estimate: {estimated_volatilities[-1]:.4f}")
    print(f"Empirical volatility: {np.std(returns):.4f}")
    
    # Test Extended Kalman Filter with nonlinear dynamics
    print("\nTesting Extended Kalman Filter:")
    
    def nonlinear_f(x, dt, u=None):
        """Nonlinear state transition (price with nonlinear trend)"""
        price, trend = x
        new_price = price + trend * dt + 0.01 * np.sin(price / 10) * dt
        new_trend = trend * 0.99  # Mean reversion in trend
        return np.array([new_price, new_trend])
        
    def linear_h(x):
        """Linear observation function"""
        return np.array([x[0]])  # Observe price only
        
    def F_jac(x, dt, u=None):
        """Jacobian of state transition"""
        price = x[0]
        return np.array([
            [1 + 0.01 * np.cos(price / 10) * dt / 10, dt],
            [0, 0.99]
        ])
        
    def H_jac(x):
        """Jacobian of observation function"""
        return np.array([[1.0, 0.0]])
        
    ekf = ExtendedKalmanFilter(
        state_dim=2,
        obs_dim=1,
        f_func=nonlinear_f,
        h_func=linear_h,
        F_jacobian=F_jac,
        H_jacobian=H_jac,
        Q=np.diag([0.1, 0.01]),
        R=np.array([[obs_noise**2]]),
        initial_state=np.array([100.0, 0.1])
    )
    
    ekf_estimates = []
    for price in observed_prices[:50]:  # Test on subset for speed
        state = ekf.predict()
        state = ekf.update(np.array([price]))
        ekf_estimates.append(state.x[0])
        
    ekf_errors = np.array(ekf_estimates) - np.array(true_prices[:len(ekf_estimates)])
    ekf_mae = np.mean(np.abs(ekf_errors))
    
    print(f"EKF price estimation MAE: {ekf_mae:.3f}")
    
    print("\nKalman filters test completed!")
