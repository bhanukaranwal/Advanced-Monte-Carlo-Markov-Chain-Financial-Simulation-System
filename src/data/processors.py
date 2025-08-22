"""
Data processing utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing and cleaning utilities"""
    
    def __init__(self):
        self.missing_data_threshold = 0.05  # 5% missing data threshold
        
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean price data"""
        
        if df.empty:
            return df
            
        # Sort by date
        df = df.sort_values('date').copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'symbol'])
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Detect and handle outliers
        df = self._handle_outliers(df)
        
        # Validate data consistency
        df = self._validate_price_data(df)
        
        return df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in price data"""
        
        # Check missing data percentage
        missing_pct = df.isnull().sum() / len(df)
        
        # Columns with too much missing data
        high_missing_cols = missing_pct[missing_pct > self.missing_data_threshold].index
        
        if len(high_missing_cols) > 0:
            logger.warning(f"Columns with high missing data: {list(high_missing_cols)}")
            
        # Forward fill price data (common practice in finance)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                
        # Volume can be filled with 0 or median
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
            
        return df
        
    def _handle_outliers(self, df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        """Detect and handle outliers using z-score method"""
        
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col not in df.columns:
                continue
                
            # Calculate returns for outlier detection
            returns = df[col].pct_change()
            
            # Z-score method
            z_scores = np.abs(stats.zscore(returns.dropna()))
            
            # Identify outliers
            outliers = z_scores > z_threshold
            
            if outliers.sum() > 0:
                logger.warning(f"Found {outliers.sum()} outliers in {col}")
                
                # Cap outliers at threshold percentiles
                lower_bound = returns.quantile(0.01)
                upper_bound = returns.quantile(0.99)
                
                # Apply to original prices (not returns)
                outlier_indices = returns[outliers].index
                
                for idx in outlier_indices:
                    if idx > 0:
                        prev_price = df.loc[idx-1, col]
                        current_return = returns.loc[idx]
                        
                        if current_return > upper_bound:
                            df.loc[idx, col] = prev_price * (1 + upper_bound)
                        elif current_return < lower_bound:
                            df.loc[idx, col] = prev_price * (1 + lower_bound)
                            
        return df
        
    def _validate_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate price data consistency"""
        
        # Check that high >= low
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                logger.warning(f"Found {invalid_hl.sum()} rows where high < low")
                # Swap values
                df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
                
        # Check that open/close are between high/low
        for price_col in ['open', 'close']:
            if price_col in df.columns and 'high' in df.columns and 'low' in df.columns:
                # Ensure price is within high/low range
                df[price_col] = np.clip(df[price_col], df['low'], df['high'])
                
        # Remove rows with zero or negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                invalid_prices = df[col] <= 0
                if invalid_prices.any():
                    logger.warning(f"Removing {invalid_prices.sum()} rows with invalid {col} prices")
                    df = df[~invalid_prices]
                    
        return df
        
    def resample_data(self, df: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """Resample data to different frequency"""
        
        if df.empty or 'date' not in df.columns:
            return df
            
        df = df.set_index('date')
        
        # Define aggregation rules
        agg_rules = {
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Apply only to columns that exist
        existing_agg_rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
        
        resampled = df.resample(frequency).agg(existing_agg_rules)
        resampled = resampled.dropna()
        resampled = resampled.reset_index()
        
        return resampled

class ReturnCalculator:
    """Calculate various types of returns"""
    
    def calculate_simple_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate simple returns"""
        return prices.pct_change()
        
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1))
        
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate realized volatility"""
        return returns.rolling(window=window).std() * np.sqrt(252)
        
    def calculate_returns_matrix(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns matrix for multiple assets"""
        
        returns_dict = {}
        
        for symbol, data in price_data.items():
            if not data.empty and 'close' in data.columns:
                returns = self.calculate_log_returns(data['close'])
                returns_dict[symbol] = returns
                
        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            return returns_df.dropna()
        else:
            return pd.DataFrame()
            
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        return returns_df.corr()
        
    def calculate_covariance_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix"""
        return returns_df.cov() * 252  # Annualized
        
    def calculate_rolling_correlations(self, returns_df: pd.DataFrame, window: int = 60) -> Dict[str, pd.DataFrame]:
        """Calculate rolling correlations"""
        
        rolling_corrs = {}
        symbols = returns_df.columns
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                pair = f"{symbols[i]}_{symbols[j]}"
                rolling_corrs[pair] = returns_df[symbols[i]].rolling(window).corr(returns_df[symbols[j]])
                
        return rolling_corrs

class TechnicalIndicators:
    """Technical analysis indicators"""
    
    def sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
        
    def ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window).mean()
        
    def rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return pd.DataFrame({
            'middle': sma,
            'upper': upper_band,
            'lower': lower_band
        })
        
    def macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD indicator"""
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
