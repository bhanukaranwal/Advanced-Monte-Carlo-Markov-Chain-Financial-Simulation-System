"""
Neural network surrogates for accelerated Monte Carlo simulation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class NeuralSurrogate:
    """Neural network surrogate for Monte Carlo pricing"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        output_dim: int = 1,
        activation: str = 'relu',
        dropout_rate: float = 0.1,
        framework: str = 'tensorflow'  # 'tensorflow' or 'pytorch'
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.framework = framework
        
        # Scalers for input/output normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # Model storage
        self.model = None
        self.trained = False
        
        # Training history
        self.training_history = {}
        
        # Build model
        self._build_model()
        
    def _build_model(self):
        """Build neural network model"""
        if self.framework == 'tensorflow':
            self._build_tensorflow_model()
        elif self.framework == 'pytorch':
            self._build_pytorch_model()
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
            
    def _build_tensorflow_model(self):
        """Build TensorFlow model"""
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        # Hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = layers.Dense(
                hidden_dim,
                activation=self.activation,
                name=f'hidden_{i}'
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization()(x)
            
            # Dropout
            if self.dropout_rate > 0:
                x = layers.Dropout(self.dropout_rate)(x)
                
        # Output layer
        outputs = layers.Dense(
            self.output_dim,
            activation='linear',
            name='output'
        )(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='neural_surrogate')
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
    def _build_pytorch_model(self):
        """Build PyTorch model"""
        class SurrogateNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim, activation, dropout_rate):
                super(SurrogateNet, self).__init__()
                
                self.layers = nn.ModuleList()
                
                # Input layer
                prev_dim = input_dim
                
                # Hidden layers
                for hidden_dim in hidden_dims:
                    self.layers.append(nn.Linear(prev_dim, hidden_dim))
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
                    
                    if activation == 'relu':
                        self.layers.append(nn.ReLU())
                    elif activation == 'tanh':
                        self.layers.append(nn.Tanh())
                    elif activation == 'sigmoid':
                        self.layers.append(nn.Sigmoid())
                        
                    if dropout_rate > 0:
                        self.layers.append(nn.Dropout(dropout_rate))
                        
                    prev_dim = hidden_dim
                    
                # Output layer
                self.layers.append(nn.Linear(prev_dim, output_dim))
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
                
        self.model = SurrogateNet(
            self.input_dim, self.hidden_dims, self.output_dim, 
            self.activation, self.dropout_rate
        )
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping: bool = True,
        patience: int = 10,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the neural surrogate
        
        Args:
            X: Input features (market parameters, path features, etc.)
            y: Target values (option prices, risk measures, etc.)
            validation_split: Fraction for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        logger.info(f"Training neural surrogate with {len(X)} samples")
        
        # Normalize inputs and outputs
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train-test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        if self.framework == 'tensorflow':
            history = self._train_tensorflow(
                X_train, y_train, X_val, y_val, epochs, batch_size, 
                learning_rate, early_stopping, patience, **kwargs
            )
        else:
            history = self._train_pytorch(
                X_train, y_train, X_val, y_val, epochs, batch_size,
                learning_rate, early_stopping, patience, **kwargs
            )
            
        self.trained = True
        self.training_history = history
        
        logger.info("Neural surrogate training completed")
        return history
        
    def _train_tensorflow(
        self, X_train, y_train, X_val, y_val, epochs, batch_size,
        learning_rate, early_stopping, patience, **kwargs
    ):
        """Train TensorFlow model"""
        # Update learning rate
        self.model.optimizer.learning_rate = learning_rate
        
        # Callbacks
        callbacks = []
        
        if early_stopping:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )
            callbacks.append(early_stop)
            
        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6
        )
        callbacks.append(lr_scheduler)
        
        # Training
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return history.history
        
    def _train_pytorch(
        self, X_train, y_train, X_val, y_val, epochs, batch_size,
        learning_rate, early_stopping, patience, **kwargs
    ):
        """Train PyTorch model"""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
        return history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained surrogate"""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        # Scale inputs
        X_scaled = self.input_scaler.transform(X)
        
        if self.framework == 'tensorflow':
            predictions_scaled = self.model.predict(X_scaled, verbose=0)
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                predictions_scaled = self.model(X_tensor).numpy()
                
        # Inverse scale outputs
        predictions = self.output_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
        
    def get_feature_importance(self, X: np.ndarray, method: str = 'permutation') -> np.ndarray:
        """
        Calculate feature importance
        
        Args:
            X: Input features
            method: 'permutation' or 'gradient'
            
        Returns:
            Feature importance scores
        """
        if not self.trained:
            raise ValueError("Model must be trained before feature importance calculation")
            
        if method == 'permutation':
            return self._permutation_importance(X)
        elif method == 'gradient':
            return self._gradient_importance(X)
        else:
            raise ValueError(f"Unknown importance method: {method}")
            
    def _permutation_importance(self, X: np.ndarray) -> np.ndarray:
        """Calculate permutation-based feature importance"""
        baseline_predictions = self.predict(X)
        baseline_error = np.mean((baseline_predictions - baseline_predictions)**2)  # Dummy calculation
        
        importance_scores = np.zeros(self.input_dim)
        
        for i in range(self.input_dim):
            # Permute feature i
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get predictions with permuted feature
            permuted_predictions = self.predict(X_permuted)
            permuted_error = np.mean((permuted_predictions - baseline_predictions)**2)
            
            # Importance is increase in error
            importance_scores[i] = permuted_error - baseline_error
            
        return importance_scores
        
    def _gradient_importance(self, X: np.ndarray) -> np.ndarray:
        """Calculate gradient-based feature importance"""
        if self.framework == 'tensorflow':
            return self._tensorflow_gradients(X)
        else:
            return self._pytorch_gradients(X)
            
    def _tensorflow_gradients(self, X: np.ndarray) -> np.ndarray:
        """TensorFlow gradient calculation"""
        X_scaled = self.input_scaler.transform(X)
        X_tensor = tf.Variable(X_scaled, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            predictions = self.model(X_tensor)
            
        gradients = tape.gradient(predictions, X_tensor)
        
        # Average absolute gradients
        importance_scores = np.mean(np.abs(gradients.numpy()), axis=0)
        
        return importance_scores
        
    def _pytorch_gradients(self, X: np.ndarray) -> np.ndarray:
        """PyTorch gradient calculation"""
        X_scaled = self.input_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        X_tensor.requires_grad_(True)
        
        self.model.eval()
        predictions = self.model(X_tensor)
        
        # Calculate gradients
        predictions.sum().backward()
        gradients = X_tensor.grad
        
        # Average absolute gradients
        importance_scores = np.mean(np.abs(gradients.numpy()), axis=0)
        
        return importance_scores
        
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.trained:
            raise ValueError("No trained model to save")
            
        if self.framework == 'tensorflow':
            self.model.save(filepath)
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_scaler': self.input_scaler,
                'output_scaler': self.output_scaler,
                'model_config': {
                    'input_dim': self.input_dim,
                    'hidden_dims': self.hidden_dims,
                    'output_dim': self.output_dim,
                    'activation': self.activation,
                    'dropout_rate': self.dropout_rate
                }
            }, filepath)
            
    def load_model(self, filepath: str):
        """Load trained model"""
        if self.framework == 'tensorflow':
            self.model = keras.models.load_model(filepath)
        else:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.input_scaler = checkpoint['input_scaler']
            self.output_scaler = checkpoint['output_scaler']
            
        self.trained = True

class DeepPathGenerator:
    """Deep learning-based path generator using GANs or VAEs"""
    
    def __init__(
        self,
        latent_dim: int = 100,
        path_length: int = 252,
        n_assets: int = 1,
        architecture: str = 'gan',  # 'gan' or 'vae'
        framework: str = 'tensorflow'
    ):
        self.latent_dim = latent_dim
        self.path_length = path_length
        self.n_assets = n_assets
        self.architecture = architecture
        self.framework = framework
        
        self.generator = None
        self.discriminator = None
        self.vae = None
        self.trained = False
        
        self._build_models()
        
    def _build_models(self):
        """Build generator/discriminator or VAE models"""
        if self.architecture == 'gan':
            self._build_gan()
        elif self.architecture == 'vae':
            self._build_vae()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
            
    def _build_gan(self):
        """Build GAN for path generation"""
        if self.framework != 'tensorflow':
            raise NotImplementedError("GAN implementation only available for TensorFlow")
            
        # Generator
        generator_input = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(generator_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(self.path_length * self.n_assets, activation='tanh')(x)
        generator_output = layers.Reshape((self.path_length, self.n_assets))(x)
        
        self.generator = Model(generator_input, generator_output, name='generator')
        
        # Discriminator
        discriminator_input = keras.Input(shape=(self.path_length, self.n_assets))
        x = layers.Flatten()(discriminator_input)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        discriminator_output = layers.Dense(1, activation='sigmoid')(x)
        
        self.discriminator = Model(discriminator_input, discriminator_output, name='discriminator')
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def _build_vae(self):
        """Build VAE for path generation"""
        if self.framework != 'tensorflow':
            raise NotImplementedError("VAE implementation only available for TensorFlow")
            
        input_dim = self.path_length * self.n_assets
        
        # Encoder
        encoder_inputs = keras.Input(shape=(input_dim,))
        h = layers.Dense(512, activation='relu')(encoder_inputs)
        h = layers.Dense(256, activation='relu')(h)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(h)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(h)
        
        # Sampling layer
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        h = layers.Dense(256, activation='relu')(latent_inputs)
        h = layers.Dense(512, activation='relu')(h)
        decoder_outputs = layers.Dense(input_dim, activation='tanh')(h)
        
        decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE model
        vae_outputs = decoder(encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, vae_outputs, name='vae')
        
        # VAE loss
        reconstruction_loss = keras.losses.mse(encoder_inputs, vae_outputs)
        reconstruction_loss *= input_dim
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        
        self.generator = decoder
        
    def train(
        self,
        real_paths: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 32,
        **kwargs
    ):
        """Train the deep path generator"""
        logger.info(f"Training deep path generator with {len(real_paths)} real paths")
        
        if self.architecture == 'gan':
            self._train_gan(real_paths, epochs, batch_size, **kwargs)
        else:
            self._train_vae(real_paths, epochs, batch_size, **kwargs)
            
        self.trained = True
        
    def _train_gan(self, real_paths, epochs, batch_size, **kwargs):
        """Train GAN"""
        # Normalize real paths to [-1, 1]
        real_paths_normalized = 2 * (real_paths - real_paths.min()) / (real_paths.max() - real_paths.min()) - 1
        
        for epoch in range(epochs):
            # Train discriminator
            batch_indices = np.random.randint(0, real_paths_normalized.shape[0], batch_size)
            real_batch = real_paths_normalized[batch_indices]
            
            # Generate fake paths
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = self.generator.predict(noise, verbose=0)
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_batch, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_batch, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Make discriminator non-trainable
            self.discriminator.trainable = False
            
            # Combined model for generator training
            gan_input = keras.Input(shape=(self.latent_dim,))
            gan_output = self.discriminator(self.generator(gan_input))
            combined = Model(gan_input, gan_output)
            combined.compile(loss='binary_crossentropy', optimizer='adam')
            
            g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
            
            # Make discriminator trainable again
            self.discriminator.trainable = True
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")
                
    def _train_vae(self, real_paths, epochs, batch_size, **kwargs):
        """Train VAE"""
        # Flatten paths for VAE
        real_paths_flat = real_paths.reshape(real_paths.shape[0], -1)
        
        # Normalize
        real_paths_normalized = 2 * (real_paths_flat - real_paths_flat.min()) / (real_paths_flat.max() - real_paths_flat.min()) - 1
        
        self.vae.fit(
            real_paths_normalized,
            real_paths_normalized,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
    def generate_paths(self, n_paths: int) -> np.ndarray:
        """Generate synthetic paths"""
        if not self.trained:
            raise ValueError("Model must be trained before path generation")
            
        # Generate from latent space
        noise = np.random.normal(0, 1, (n_paths, self.latent_dim))
        
        if self.architecture == 'gan':
            generated_paths = self.generator.predict(noise, verbose=0)
        else:
            generated_flat = self.generator.predict(noise, verbose=0)
            generated_paths = generated_flat.reshape(n_paths, self.path_length, self.n_assets)
            
        return generated_paths

class VariationalAutoencoder:
    """Variational Autoencoder for dimensionality reduction and generation"""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 50,
        hidden_dims: List[int] = [256, 128],
        framework: str = 'tensorflow'
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.framework = framework
        
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.trained = False
        
        self._build_vae()
        
    def _build_vae(self):
        """Build VAE architecture"""
        if self.framework == 'tensorflow':
            self._build_tensorflow_vae()
        else:
            self._build_pytorch_vae()
            
    def _build_tensorflow_vae(self):
        """Build TensorFlow VAE"""
        # Encoder
        encoder_inputs = keras.Input(shape=(self.input_dim,))
        x = encoder_inputs
        
        for hidden_dim in self.hidden_dims:
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = latent_inputs
        
        for hidden_dim in reversed(self.hidden_dims):
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
        decoder_outputs = layers.Dense(self.input_dim, activation='linear')(x)
        
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE
        vae_outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = Model(encoder_inputs, vae_outputs, name='vae')
        
        # Loss function
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(encoder_inputs, vae_outputs)
        ) * self.input_dim
        
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        
        total_loss = reconstruction_loss + kl_loss
        self.vae.add_loss(total_loss)
        self.vae.compile(optimizer='adam')
        
    def _build_pytorch_vae(self):
        """Build PyTorch VAE"""
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim, hidden_dims):
                super(VAE, self).__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                    
                self.encoder = nn.Sequential(*encoder_layers)
                self.fc_mu = nn.Linear(prev_dim, latent_dim)
                self.fc_logvar = nn.Linear(prev_dim, latent_dim)
                
                # Decoder
                decoder_layers = []
                prev_dim = latent_dim
                
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                    
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
                
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)
                
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
                
            def decode(self, z):
                return self.decoder(z)
                
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar
                
        self.vae = VAE(self.input_dim, self.latent_dim, self.hidden_dims)
        
    def train(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs
    ):
        """Train the VAE"""
        if self.framework == 'tensorflow':
            self.vae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
        else:
            self._train_pytorch_vae(X, epochs, batch_size, learning_rate)
            
        self.trained = True
        
    def _train_pytorch_vae(self, X, epochs, batch_size, learning_rate):
        """Train PyTorch VAE"""
        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                x = batch[0]
                
                optimizer.zero_grad()
                
                recon_x, mu, logvar = self.vae(x)
                
                # Loss function
                recon_loss = nn.MSELoss()(recon_x, x) * self.input_dim
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                loss = recon_loss + kl_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
                
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data to latent space"""
        if not self.trained:
            raise ValueError("VAE must be trained before encoding")
            
        if self.framework == 'tensorflow':
            return self.encoder.predict(X, verbose=0)[0]  # Return mean
        else:
            self.vae.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                mu, _ = self.vae.encode(X_tensor)
                return mu.numpy()
                
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space"""
        if not self.trained:
            raise ValueError("VAE must be trained before decoding")
            
        if self.framework == 'tensorflow':
            return self.decoder.predict(z, verbose=0)
        else:
            self.vae.eval()
            with torch.no_grad():
                z_tensor = torch.FloatTensor(z)
                return self.vae.decode(z_tensor).numpy()
                
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate new samples"""
        if not self.trained:
            raise ValueError("VAE must be trained before generation")
            
        # Sample from latent space
        z = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.decode(z)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Surrogates...")
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 10000
    n_features = 10
    
    # Synthetic market parameters (volatility, drift, time to maturity, etc.)
    X = np.random.randn(n_samples, n_features)
    
    # Synthetic option prices (Black-Scholes-like function)
    def synthetic_option_price(params):
        # Simplified BS formula approximation
        vol, drift, ttm = params[0], params[1], params
        return 100 * np.exp(-0.5 * vol**2 * ttm + vol * np.sqrt(ttm) * np.random.normal(0, 1))
        
    y = np.array([synthetic_option_price(x[:3]) for x in X])
    
    print(f"Generated {n_samples} training samples")
    
    # Test Neural Surrogate
    print("\nTesting Neural Surrogate:")
    surrogate = NeuralSurrogate(
        input_dim=n_features,
        hidden_dims=[64, 32, 16],
        framework='tensorflow'
    )
    
    # Train
    history = surrogate.train(X, y, epochs=50, batch_size=64)
    print(f"Training completed. Final loss: {history['loss'][-1]:.4f}")
    
    # Predict
    test_X = X[:100]
    predictions = surrogate.predict(test_X)
    actual = y[:100]
    
    mae = np.mean(np.abs(predictions - actual))
    print(f"Test MAE: {mae:.4f}")
    
    # Feature importance
    importance = surrogate.get_feature_importance(test_X[:10])
    print(f"Feature importance: {importance[:5]}")
    
    # Test VAE
    print("\nTesting Variational Autoencoder:")
    vae = VariationalAutoencoder(
        input_dim=n_features,
        latent_dim=5,
        hidden_dims=[32, 16]
    )
    
    vae.train(X, epochs=50)
    
    # Encode and decode
    encoded = vae.encode(X[:100])
    decoded = vae.decode(encoded)
    
    reconstruction_error = np.mean((X[:100] - decoded)**2)
    print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # Generate new samples
    generated = vae.generate(10)
    print(f"Generated samples shape: {generated.shape}")
    
    print("\nNeural surrogates test completed!")
