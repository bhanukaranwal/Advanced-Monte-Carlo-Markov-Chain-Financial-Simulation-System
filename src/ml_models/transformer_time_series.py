"""
Advanced Transformer models for financial time series forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, Any, List
import logging
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig:
    """Configuration for Transformer model"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 1000
    n_features: int = 1
    n_targets: int = 1

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(context)

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model)
        )
        
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class FinancialTransformer(nn.Module):
    """
    Transformer model for financial time series forecasting
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Input embedding
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.n_targets)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Market regime detection head
        self.regime_head = nn.Linear(config.d_model, 3)  # Bull, Bear, Sideways
        
        # Volatility prediction head
        self.volatility_head = nn.Linear(config.d_model, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
            
        # Generate predictions
        price_pred = self.output_projection(x)
        regime_pred = self.regime_head(x[:, -1, :])  # Use last timestep
        volatility_pred = self.volatility_head(x[:, -1, :])
        
        return {
            'price_prediction': price_pred,
            'regime_prediction': regime_pred,
            'volatility_prediction': volatility_pred,
            'hidden_states': x
        }

class FinancialTimeSeriesDataset(Dataset):
    """Dataset for financial time series"""
    
    def __init__(self, data: np.ndarray, sequence_length: int, prediction_horizon: int):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon]
        return x, y

class TransformerTimeSeriesPredictor:
    """
    Transformer-based time series predictor for financial data
    """
    
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.model = FinancialTransformer(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train(self, train_data: np.ndarray, val_data: np.ndarray,
              sequence_length: int = 60, prediction_horizon: int = 1,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the transformer model"""
        
        # Create datasets
        train_dataset = FinancialTimeSeriesDataset(
            train_data, sequence_length, prediction_horizon
        )
        val_dataset = FinancialTimeSeriesDataset(
            val_data, sequence_length, prediction_horizon
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
        return {'train_losses': train_losses, 'val_losses': val_losses}
        
    def _train_epoch(self, data_loader: DataLoader) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_x)
            
            # Calculate losses
            price_loss = F.mse_loss(outputs['price_prediction'][:, -1, :], batch_y[:, -1, :])
            
            # Add regime consistency loss
            regime_probs = F.softmax(outputs['regime_prediction'], dim=1)
            regime_entropy = -torch.sum(regime_probs * torch.log(regime_probs + 1e-8), dim=1).mean()
            
            total_loss_batch = price_loss + 0.1 * regime_entropy
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def _validate_epoch(self, data_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = F.mse_loss(outputs['price_prediction'][:, -1, :], batch_y[:, -1, :])
                
                total_loss += loss.item()
                num_batches += 1
                
        self.model.train()
        return total_loss / num_batches
        
    def predict(self, data: np.ndarray, sequence_length: int = 60) -> Dict[str, np.ndarray]:
        """Make predictions"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(data[-sequence_length:]).unsqueeze(0).to(self.device)
            outputs = self.model(x)
            
            return {
                'price_prediction': outputs['price_prediction'].cpu().numpy(),
                'regime_prediction': F.softmax(outputs['regime_prediction'], dim=1).cpu().numpy(),
                'volatility_prediction': outputs['volatility_prediction'].cpu().numpy()
            }
            
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
