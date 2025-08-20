"""
Reinforcement Learning optimization for trading strategies and parameter tuning
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import logging

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TradingEnvironment(gym.Env):
    """Custom trading environment for RL optimization"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        lookback_window: int = 20
    ):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position (-1 to 1)
        self.total_portfolio_value = initial_balance
        self.trade_history = []
        
        # Action space: continuous action between -1 and 1 (position sizing)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: price features + position + portfolio metrics
        n_features = len(data.columns) + 3  # +3 for position, balance ratio, drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(lookback_window, n_features), dtype=np.float32
        )
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_portfolio_value = self.initial_balance
        self.trade_history = []
        
        return self._get_observation()
        
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
            
        # Current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action (position sizing)
        new_position = np.clip(action[0], -self.max_position_size, self.max_position_size)
        position_change = new_position - self.position
        
        # Calculate transaction cost
        transaction_cost = abs(position_change) * current_price * self.transaction_cost
        
        # Update balance and position
        self.balance -= position_change * current_price + transaction_cost
        self.position = new_position
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value
        next_price = self.data.iloc[self.current_step]['close']
        position_value = self.position * next_price
        self.total_portfolio_value = self.balance + position_value
        
        # Calculate reward (returns with risk adjustment)
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.total_portfolio_value <= 0.1 * self.initial_balance)
        
        # Store trade information
        if abs(position_change) > 1e-6:
            self.trade_history.append({
                'step': self.current_step,
                'action': action[0],
                'position_change': position_change,
                'price': current_price,
                'transaction_cost': transaction_cost,
                'portfolio_value': self.total_portfolio_value
            })
            
        return self._get_observation(), reward, done, self._get_info()
        
    def _get_observation(self):
        """Get current observation"""
        if self.current_step < self.lookback_window:
            # Pad with zeros if not enough history
            market_data = np.zeros((self.lookback_window, len(self.data.columns)))
            available_data = self.data.iloc[:self.current_step + 1].values
            market_data[-len(available_data):] = available_data
        else:
            market_data = self.data.iloc[
                self.current_step - self.lookback_window + 1:self.current_step + 1
            ].values
            
        # Add portfolio state
        portfolio_state = np.array([
            self.position,
            self.balance / self.initial_balance,
            self._calculate_drawdown()
        ])
        
        # Broadcast portfolio state to match time dimension
        portfolio_features = np.tile(portfolio_state, (self.lookback_window, 1))
        
        # Combine market data and portfolio state
        observation = np.concatenate([market_data, portfolio_features], axis=1)
        
        return observation.astype(np.float32)
        
    def _calculate_reward(self):
        """Calculate step reward"""
        if len(self.trade_history) == 0:
            return 0.0
            
        # Portfolio return
        current_return = (self.total_portfolio_value / self.initial_balance) - 1.0
        
        # Risk-adjusted reward (Sharpe-like)
        if len(self.trade_history) > 1:
            returns = [trade['portfolio_value'] / self.initial_balance - 1.0 
                      for trade in self.trade_history[-10:]]  # Last 10 trades
            volatility = np.std(returns) if len(returns) > 1 else 0.01
            sharpe_like = current_return / (volatility + 1e-8)
        else:
            sharpe_like = current_return
            
        # Penalty for large positions (risk management)
        position_penalty = -0.1 * abs(self.position) if abs(self.position) > 0.8 else 0.0
        
        # Penalty for drawdown
        drawdown_penalty = -0.5 * abs(self._calculate_drawdown())
        
        reward = sharpe_like + position_penalty + drawdown_penalty
        
        return reward
        
    def _calculate_drawdown(self):
        """Calculate current drawdown"""
        if not self.trade_history:
            return 0.0
            
        portfolio_values = [trade['portfolio_value'] for trade in self.trade_history]
        peak = max(portfolio_values + [self.initial_balance])
        
        if peak == 0:
            return 0.0
            
        return (self.total_portfolio_value - peak) / peak
        
    def _get_info(self):
        """Get additional info"""
        return {
            'portfolio_value': self.total_portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_return': (self.total_portfolio_value / self.initial_balance) - 1.0,
            'num_trades': len(self.trade_history),
            'drawdown': self._calculate_drawdown()
        }

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int):
        """Sample batch from buffer"""
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for continuous action spaces"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64]
    ):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            self.shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Actor head (policy network)
        self.actor_layers = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Critic head (value network)
        self.critic_layers = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
    def forward(self, state):
        """Forward pass"""
        # Flatten state if needed
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
            
        # Shared features
        x = state
        for layer in self.shared_layers:
            x = layer(x)
            
        # Actor and critic outputs
        action_probs = self.actor_layers(x)
        value = self.critic_layers(x)
        
        return action_probs, value

class PPOAgent:
    """Proximal Policy Optimization agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        # Networks
        self.policy = ActorCriticNetwork(state_dim, action_dim)
        self.policy_old = ActorCriticNetwork(state_dim, action_dim)
        
        # Copy weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_mean, _ = self.policy_old(state_tensor)
            
        if deterministic:
            action = action_mean
        else:
            # Add noise for exploration
            std = 0.1  # Fixed standard deviation
            noise = torch.normal(0, std, action_mean.shape)
            action = action_mean + noise
            action = torch.clamp(action, -1, 1)
            
        # Calculate log probability (assuming normal distribution)
        log_prob = -0.5 * ((action - action_mean) / std) ** 2 - 0.5 * np.log(2 * np.pi * std ** 2)
        log_prob = log_prob.sum(dim=-1)
        
        return action.numpy().flatten(), log_prob.item()
        
    def store_experience(self, state, action, reward, log_prob, value, done):
        """Store experience for training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        
    def update(self):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return {}
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(self.rewards)
        old_log_probs = torch.FloatTensor(self.log_probs)
        values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)
        
        # Flatten states if needed
        if states.dim() > 2:
            states = states.view(states.size(0), -1)
            
        # Calculate discounted rewards
        returns = self._calculate_returns(rewards, dones)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.k_epochs):
            # Forward pass with current policy
            action_means, new_values = self.policy(states)
            
            # Calculate new log probabilities
            std = 0.1
            new_log_probs = -0.5 * ((actions - action_means) / std) ** 2 - 0.5 * np.log(2 * np.pi * std ** 2)
            new_log_probs = new_log_probs.sum(dim=-1)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Entropy loss (for exploration)
            entropy_loss = -torch.mean(-0.5 * np.log(2 * np.pi * std ** 2) - 0.5)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
        
    def _calculate_returns(self, rewards, dones):
        """Calculate discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            
        return returns

class RLOptimizer:
    """Main RL optimization orchestrator"""
    
    def __init__(
        self,
        environment: TradingEnvironment,
        algorithm: str = 'ppo',
        **kwargs
    ):
        self.environment = environment
        self.algorithm = algorithm
        
        # Determine state and action dimensions
        obs_sample = environment.reset()
        state_dim = obs_sample.flatten().shape[0]
        action_dim = environment.action_space.shape
        
        # Initialize agent
        if algorithm == 'ppo':
            self.agent = PPOAgent(state_dim, action_dim, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'policy_losses': [],
            'value_losses': []
        }
        
    def train(
        self,
        n_episodes: int = 1000,
        max_steps_per_episode: int = None,
        eval_frequency: int = 100,
        save_frequency: int = 500
    ):
        """Train the RL agent"""
        logger.info(f"Starting RL training for {n_episodes} episodes")
        
        if max_steps_per_episode is None:
            max_steps_per_episode = len(self.environment.data) - self.environment.lookback_window
            
        for episode in range(n_episodes):
            state = self.environment.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action, log_prob = self.agent.select_action(state.flatten())
                
                # Get value estimate
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                with torch.no_grad():
                    _, value = self.agent.policy(state_tensor)
                    
                # Take action
                next_state, reward, done, info = self.environment.step(action)
                
                # Store experience
                self.agent.store_experience(
                    state.flatten(), action, reward, log_prob, value.item(), done
                )
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
                    
            # Update policy
            if self.algorithm == 'ppo' and len(self.agent.states) > 0:
                losses = self.agent.update()
                
                if losses:
                    self.training_history['policy_losses'].append(losses['policy_loss'])
                    self.training_history['value_losses'].append(losses['value_loss'])
                    
            # Record metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['portfolio_values'].append(
                self.environment.total_portfolio_value
            )
            
            # Logging
            if episode % eval_frequency == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-eval_frequency:])
                avg_portfolio = np.mean(self.training_history['portfolio_values'][-eval_frequency:])
                
                logger.info(
                    f"Episode {episode}: Avg Reward: {avg_reward:.4f}, "
                    f"Avg Portfolio Value: {avg_portfolio:.2f}"
                )
                
        logger.info("RL training completed")
        
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate trained policy"""
        total_rewards = []
        final_portfolio_values = []
        total_returns = []
        max_drawdowns = []
        
        for episode in range(n_episodes):
            state = self.environment.reset()
            episode_reward = 0
            
            while True:
                action, _ = self.agent.select_action(state.flatten(), deterministic=deterministic)
                next_state, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
                    
            total_rewards.append(episode_reward)
            final_portfolio_values.append(info['portfolio_value'])
            total_returns.append(info['total_return'])
            max_drawdowns.append(abs(info['drawdown']))
            
        return {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'avg_portfolio_value': np.mean(final_portfolio_values),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'sharpe_ratio': np.mean(total_returns) / (np.std(total_returns) + 1e-8)
        }
        
    def get_trading_strategy(self):
        """Extract learned trading strategy"""
        def strategy_func(market_data):
            """
            Trading strategy function
            
            Args:
                market_data: DataFrame with OHLCV data
                
            Returns:
                Series of position signals
            """
            positions = []
            
            # Create temporary environment for strategy execution
            temp_env = TradingEnvironment(
                market_data,
                initial_balance=self.environment.initial_balance,
                transaction_cost=self.environment.transaction_cost,
                lookback_window=self.environment.lookback_window
            )
            
            state = temp_env.reset()
            
            while temp_env.current_step < len(market_data) - 1:
                action, _ = self.agent.select_action(state.flatten(), deterministic=True)
                state, _, done, _ = temp_env.step(action)
                positions.append(action[0])
                
                if done:
                    break
                    
            return pd.Series(positions, index=market_data.index[temp_env.lookback_window:temp_env.lookback_window+len(positions)])
            
        return strategy_func

class PolicyGradientOptimizer:
    """Simple policy gradient optimizer"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99
    ):
        self.gamma = gamma
        
        # Simple policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state)
        action_mean = self.policy_net(state_tensor)
        
        # Add noise for exploration
        std = 0.1
        action = action_mean + torch.normal(0, std, action_mean.shape)
        action = torch.clamp(action, -1, 1)
        
        # Calculate log probability
        log_prob = -0.5 * ((action - action_mean) / std) ** 2 - 0.5 * np.log(2 * np.pi * std ** 2)
        log_prob = log_prob.sum()
        
        self.log_probs.append(log_prob)
        
        return action.detach().numpy()
        
    def store_reward(self, reward):
        """Store reward for episode"""
        self.rewards.append(reward)
        
    def update_policy(self):
        """Update policy using REINFORCE"""
        if len(self.rewards) == 0:
            return
            
        # Calculate discounted returns
        returns = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.log_probs.clear()
        self.rewards.clear()

class QLearningOptimizer:
    """Q-Learning optimizer for discrete action spaces"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        
    def update_q_network(self, batch_size=32):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return
            
        # Sample batch
        experiences = self.replay_buffer.sample(batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.BoolTensor(batch.done)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

# Example usage and testing
if __name__ == "__main__":
    print("Testing RL Optimization...")
    
    # Generate synthetic market data
    np.random.seed(42)
    n_days = 1000
    
    # Create synthetic price data with trends and volatility
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [100]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    # Create market data DataFrame
    market_data = pd.DataFrame({
        'close': prices[1:],
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices[1:]],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[1:]],
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # Ensure high >= close >= low
    market_data['high'] = np.maximum(market_data[['open', 'close']].max(axis=1), market_data['high'])
    market_data['low'] = np.minimum(market_data[['open', 'close']].min(axis=1), market_data['low'])
    
    print(f"Generated {len(market_data)} days of synthetic market data")
    
    # Create trading environment
    env = TradingEnvironment(
        data=market_data,
        initial_balance=100000,
        transaction_cost=0.001,
        lookback_window=20
    )
    
    print("Created trading environment")
    
    # Test environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Take a random action
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print(f"Step result - Reward: {reward:.4f}, Done: {done}, Portfolio: {info['portfolio_value']:.2f}")
    
    # Initialize RL optimizer
    rl_optimizer = RLOptimizer(env, algorithm='ppo')
    
    print("Training RL agent...")
    # Quick training for demonstration
    rl_optimizer.train(n_episodes=50, eval_frequency=10)
    
    # Evaluate trained policy
    print("\nEvaluating trained policy...")
    eval_results = rl_optimizer.evaluate(n_episodes=5)
    
    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
        
    # Get trading strategy
    strategy = rl_optimizer.get_trading_strategy()
    print("Extracted trading strategy function")
    
    print("\nRL optimization test completed!")
