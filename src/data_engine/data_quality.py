"""
Comprehensive data quality control and validation system
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report"""
    symbol: str
    total_records: int
    date_range: Tuple[datetime, datetime]
    
    # Missing data analysis
    missing_data_pct: float
    missing_by_column: Dict[str, float]
    gaps_in_dates: List[Tuple[datetime, datetime]]
    
    # Data integrity checks
    negative_prices: int
    zero_volume_days: int
    price_inconsistencies: int  # high < low, etc.
    
    # Outlier detection
    price_outliers: int
    volume_outliers: int
    return_outliers: int
    
    # Statistical properties
    price_statistics: Dict[str, float]
    volume_statistics: Dict[str, float]
    return_statistics: Dict[str, float]
    
    # Data quality score (0-100)
    overall_quality_score: float
    
    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class DataQualityController:
    """Advanced data quality control system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 3.0)
        self.min_trading_days_per_month = self.config.get('min_trading_days_per_month', 15)
        
    def assess_data_quality(self, data: pd.DataFrame, symbol: str) -> DataQualityReport:
        """
        Comprehensive data quality assessment
        
        Args:
            data: Market data DataFrame
            symbol: Symbol identifier
            
        Returns:
            DataQualityReport with detailed analysis
        """
        logger.info(f"Assessing data quality for {symbol}")
        
        # Basic information
        total_records = len(data)
        date_range = (data.index.min(), data.index.max()) if len(data) > 0 else (None, None)
        
        # Missing data analysis
        missing_analysis = self._analyze_missing_data(data)
        
        # Data integrity checks
        integrity_issues = self._check_data_integrity(data)
        
        # Outlier detection
        outliers = self._detect_outliers(data)
        
        # Statistical analysis
        statistics = self._calculate_statistics(data)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            missing_analysis, integrity_issues, outliers, total_records
        )
        
        # Generate issues and recommendations
        issues, warnings_list, recommendations = self._generate_recommendations(
            missing_analysis, integrity_issues, outliers, statistics
        )
        
        return DataQualityReport(
            symbol=symbol,
            total_records=total_records,
            date_range=date_range,
            missing_data_pct=missing_analysis['overall_missing_pct'],
            missing_by_column=missing_analysis['by_column'],
            gaps_in_dates=missing_analysis['date_gaps'],
            negative_prices=integrity_issues['negative_prices'],
            zero_volume_days=integrity_issues['zero_volume'],
            price_inconsistencies=integrity_issues['price_inconsistencies'],
            price_outliers=outliers['price_outliers'],
            volume_outliers=outliers['volume_outliers'], 
            return_outliers=outliers['return_outliers'],
            price_statistics=statistics['price'],
            volume_statistics=statistics['volume'],
            return_statistics=statistics['returns'],
            overall_quality_score=quality_score,
            critical_issues=issues,
            warnings=warnings_list,
            recommendations=recommendations
        )
        
    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        if data.empty:
            return {
                'overall_missing_pct': 100.0,
                'by_column': {},
                'date_gaps': []
            }
            
        # Missing data by column
        missing_by_column = {}
        for col in data.columns:
            missing_pct = (data[col].isna().sum() / len(data)) * 100
            missing_by_column[col] = missing_pct
            
        # Overall missing percentage
        overall_missing_pct = data.isna().any(axis=1).sum() / len(data) * 100
        
        # Detect date gaps (missing trading days)
        date_gaps = []
        if isinstance(data.index, pd.DatetimeIndex):
            # Expected business days
            full_date_range = pd.date_range(
                start=data.index.min(),
                end=data.index.max(),
                freq='B'  # Business days
            )
            
            missing_dates = full_date_range.difference(data.index)
            
            # Group consecutive missing dates into gaps
            if len(missing_dates) > 0:
                gap_start = missing_dates[0]
                gap_end = gap_start
                
                for i in range(1, len(missing_dates)):
                    if (missing_dates[i] - missing_dates[i-1]).days == 1:
                        gap_end = missing_dates[i]
                    else:
                        if (gap_end - gap_start).days >= 2:  # Gaps of 3+ days
                            date_gaps.append((gap_start, gap_end))
                        gap_start = missing_dates[i]
                        gap_end = gap_start
                        
                # Don't forget the last gap
                if (gap_end - gap_start).days >= 2:
                    date_gaps.append((gap_start, gap_end))
        
        return {
            'overall_missing_pct': overall_missing_pct,
            'by_column': missing_by_column,
            'date_gaps': date_gaps
        }
        
    def _check_data_integrity(self, data: pd.DataFrame) -> Dict[str, int]:
        """Check for data integrity issues"""
        issues = {
            'negative_prices': 0,
            'zero_volume': 0,
            'price_inconsistencies': 0
        }
        
        if data.empty:
            return issues
            
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                issues['negative_prices'] += (data[col] <= 0).sum()
                
        # Check for zero volume
        if 'volume' in data.columns:
            issues['zero_volume'] = (data['volume'] == 0).sum()
            
        # Check price relationships (high >= low, etc.)
        if all(col in data.columns for col in ['high', 'low', 'open', 'close']):
            # High should be >= Low
            issues['price_inconsistencies'] += (data['high'] < data['low']).sum()
            
            # High should be >= Open and Close
            issues['price_inconsistencies'] += (data['high'] < data['open']).sum()
            issues['price_inconsistencies'] += (data['high'] < data['close']).sum()
            
            # Low should be <= Open and Close
            issues['price_inconsistencies'] += (data['low'] > data['open']).sum()
            issues['price_inconsistencies'] += (data['low'] > data['close']).sum()
            
        return issues
        
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers in price and volume data"""
        outliers = {
            'price_outliers': 0,
            'volume_outliers': 0,
            'return_outliers': 0
        }
        
        if data.empty:
            return outliers
            
        # Price outliers (using close prices)
        if 'close' in data.columns:
            price_outliers = self._detect_outliers_column(data['close'])
            outliers['price_outliers'] = len(price_outliers)
            
        # Volume outliers
        if 'volume' in data.columns:
            volume_outliers = self._detect_outliers_column(data['volume'])
            outliers['volume_outliers'] = len(volume_outliers)
            
        # Return outliers
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            return_outliers = self._detect_outliers_column(returns)
            outliers['return_outliers'] = len(return_outliers)
            
        return outliers
        
    def _detect_outliers_column(self, series: pd.Series) -> np.ndarray:
        """Detect outliers in a single column"""
        clean_data = series.dropna()
        
        if len(clean_data) == 0:
            return np.array([])
            
        if self.outlier_method == 'iqr':
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            
        elif self.outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(clean_data))
            outliers = clean_data[z_scores > self.outlier_threshold]
            
        elif self.outlier_method == 'modified_zscore':
            median = np.median(clean_data)
            mad = np.median(np.abs(clean_data - median))
            modified_z_scores = 0.6745 * (clean_data - median) / mad
            outliers = clean_data[np.abs(modified_z_scores) > self.outlier_threshold]
            
        else:
            outliers = pd.Series([], dtype=clean_data.dtype)
            
        return outliers.index.values
        
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistical properties"""
        statistics = {
            'price': {},
            'volume': {},
            'returns': {}
        }
        
        if data.empty:
            return statistics
            
        # Price statistics (using close price)
        if 'close' in data.columns:
            prices = data['close'].dropna()
            if len(prices) > 0:
                statistics['price'] = {
                    'mean': float(prices.mean()),
                    'std': float(prices.std()),
                    'min': float(prices.min()),
                    'max': float(prices.max()),
                    'skewness': float(prices.skew()),
                    'kurtosis': float(prices.kurtosis())
                }
                
        # Volume statistics  
        if 'volume' in data.columns:
            volumes = data['volume'].dropna()
            if len(volumes) > 0:
                statistics['volume'] = {
                    'mean': float(volumes.mean()),
                    'std': float(volumes.std()),
                    'min': float(volumes.min()),
                    'max': float(volumes.max()),
                    'skewness': float(volumes.skew()),
                    'kurtosis': float(volumes.kurtosis())
                }
                
        # Return statistics
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            if len(returns) > 0:
                statistics['returns'] = {
                    'mean': float(returns.mean()),
                    'std': float(returns.std()),
                    'min': float(returns.min()),
                    'max': float(returns.max()),
                    'skewness': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                }
                
        return statistics
        
    def _calculate_quality_score(
        self, 
        missing_analysis: Dict, 
        integrity_issues: Dict, 
        outliers: Dict, 
        total_records: int
    ) -> float:
        """Calculate overall data quality score (0-100)"""
        if total_records == 0:
            return 0.0
            
        score = 100.0
        
        # Deduct for missing data
        score -= missing_analysis['overall_missing_pct'] * 0.5
        
        # Deduct for integrity issues
        integrity_penalty = (
            integrity_issues['negative_prices'] + 
            integrity_issues['price_inconsistencies']
        ) / total_records * 100
        score -= integrity_penalty * 0.8
        
        # Deduct for outliers (less penalty)
        outlier_penalty = (
            outliers['price_outliers'] + 
            outliers['volume_outliers']
        ) / total_records * 100
        score -= outlier_penalty * 0.3
        
        # Deduct for date gaps
        gap_penalty = len(missing_analysis['date_gaps']) * 2
        score -= gap_penalty
        
        return max(0.0, min(100.0, score))
        
    def _generate_recommendations(
        self, 
        missing_analysis: Dict, 
        integrity_issues: Dict, 
        outliers: Dict, 
        statistics: Dict
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate issues, warnings, and recommendations"""
        critical_issues = []
        warnings_list = []
        recommendations = []
        
        # Critical issues
        if missing_analysis['overall_missing_pct'] > 20:
            critical_issues.append(f"High missing data rate: {missing_analysis['overall_missing_pct']:.1f}%")
            
        if integrity_issues['negative_prices'] > 0:
            critical_issues.append(f"Found {integrity_issues['negative_prices']} negative price records")
            
        if integrity_issues['price_inconsistencies'] > 0:
            critical_issues.append(f"Found {integrity_issues['price_inconsistencies']} price inconsistencies")
            
        # Warnings
        if len(missing_analysis['date_gaps']) > 5:
            warnings_list.append(f"Multiple date gaps detected: {len(missing_analysis['date_gaps'])}")
            
        if outliers['price_outliers'] > len(statistics.get('price', {})) * 0.05:
            warnings_list.append("High number of price outliers detected")
            
        if integrity_issues['zero_volume'] > 0:
            warnings_list.append(f"Found {integrity_issues['zero_volume']} zero volume days")
            
        # Recommendations
        if missing_analysis['overall_missing_pct'] > 5:
            recommendations.append("Consider data imputation or alternative data sources")
            
        if outliers['return_outliers'] > 0:
            recommendations.append("Review return outliers for data quality or market events")
            
        if len(missing_analysis['date_gaps']) > 0:
            recommendations.append("Fill date gaps with appropriate methods")
            
        return critical_issues, warnings_list, recommendations

class DataCleaner:
    """Data cleaning and preprocessing utilities"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def clean_data(self, data: pd.DataFrame, quality_report: DataQualityReport) -> pd.DataFrame:
        """
        Clean data based on quality assessment
        
        Args:
            data: Raw market data
            quality_report: Data quality assessment
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data for {quality_report.symbol}")
        
        cleaned_data = data.copy()
        
        # Fix price inconsistencies
        cleaned_data = self._fix_price_inconsistencies(cleaned_data)
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # Remove or adjust outliers
        cleaned_data = self._handle_outliers(cleaned_data)
        
        # Ensure data types
        cleaned_data = self._ensure_data_types(cleaned_data)
        
        # Sort by date
        if isinstance(cleaned_data.index, pd.DatetimeIndex):
            cleaned_data = cleaned_data.sort_index()
            
        logger.info(f"Data cleaning completed. Records: {len(data)} -> {len(cleaned_data)}")
        
        return cleaned_data
        
    def _fix_price_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix basic price relationship inconsistencies"""
        fixed_data = data.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in fixed_data.columns for col in required_cols):
            return fixed_data
            
        # Ensure high is the maximum of all prices
        fixed_data['high'] = np.maximum.reduce([
            fixed_data['high'], 
            fixed_data['open'], 
            fixed_data['close']
        ])
        
        # Ensure low is the minimum of all prices
        fixed_data['low'] = np.minimum.reduce([
            fixed_data['low'], 
            fixed_data['open'], 
            fixed_data['close']
        ])
        
        # Remove rows with negative or zero prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            fixed_data = fixed_data[fixed_data[col] > 0]
            
        return fixed_data
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using various imputation methods"""
        filled_data = data.copy()
        
        # Forward fill for price data (carry last observation forward)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in filled_data.columns:
                filled_data[col] = filled_data[col].fillna(method='ffill')
                
        # For volume, use median imputation
        if 'volume' in filled_data.columns:
            median_volume = filled_data['volume'].median()
            filled_data['volume'] = filled_data['volume'].fillna(median_volume)
            
        # Drop rows that still have missing critical data
        critical_columns = ['close']  # At minimum, we need close prices
        filled_data = filled_data.dropna(subset=critical_columns)
        
        return filled_data
        
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using various methods"""
        adjusted_data = data.copy()
        
        # For extreme outliers in returns, cap them
        if 'close' in adjusted_data.columns:
            returns = adjusted_data['close'].pct_change()
            
            # Cap returns at 3 standard deviations
            return_std = returns.std()
            return_mean = returns.mean()
            
            upper_bound = return_mean + 3 * return_std
            lower_bound = return_mean - 3 * return_std
            
            # Adjust close prices based on capped returns
            for i in range(1, len(adjusted_data)):
                if not pd.isna(returns.iloc[i]):
                    if returns.iloc[i] > upper_bound:
                        adjusted_data.iloc[i, adjusted_data.columns.get_loc('close')] = \
                            adjusted_data.iloc[i-1, adjusted_data.columns.get_loc('close')] * (1 + upper_bound)
                    elif returns.iloc[i] < lower_bound:
                        adjusted_data.iloc[i, adjusted_data.columns.get_loc('close')] = \
                            adjusted_data.iloc[i-1, adjusted_data.columns.get_loc('close')] * (1 + lower_bound)
        
        return adjusted_data
        
    def _ensure_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types"""
        typed_data = data.copy()
        
        # Price columns should be float
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in typed_data.columns:
                typed_data[col] = pd.to_numeric(typed_data[col], errors='coerce')
                
        # Volume should be integer
        if 'volume' in typed_data.columns:
            typed_data['volume'] = pd.to_numeric(typed_data['volume'], errors='coerce').astype('Int64')
            
        return typed_data

# Example usage and testing
if __name__ == "__main__":
    # Create sample data with quality issues
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate base prices
    base_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    sample_data = pd.DataFrame({
        'open': base_prices + np.random.randn(len(dates)) * 0.1,
        'high': base_prices + np.abs(np.random.randn(len(dates))) * 0.5,
        'low': base_prices - np.abs(np.random.randn(len(dates))) * 0.5,
        'close': base_prices,
        'volume': np.random.randint(100000, 5000000, len(dates))
    }, index=dates)
    
    # Introduce quality issues
    # Missing data
    sample_data.iloc[100:110, :] = np.nan
    sample_data.iloc[500:505, 1] = np.nan  # Missing high prices
    
    # Price inconsistencies
    sample_data.iloc[200, sample_data.columns.get_loc('low')] = sample_data.iloc[200, sample_data.columns.get_loc('high')] + 1
    
    # Negative price
    sample_data.iloc[300, sample_data.columns.get_loc('close')] = -10
    
    # Zero volume
    sample_data.iloc[400:402, sample_data.columns.get_loc('volume')] = 0
    
    # Outliers
    sample_data.iloc[600, sample_data.columns.get_loc('close')] *= 10  # Price spike
    
    # Test quality control
    quality_controller = DataQualityController()
    quality_report = quality_controller.assess_data_quality(sample_data, 'TEST')
    
    print("=== Data Quality Report ===")
    print(f"Symbol: {quality_report.symbol}")
    print(f"Total Records: {quality_report.total_records}")
    print(f"Overall Quality Score: {quality_report.overall_quality_score:.1f}")
    print(f"Missing Data: {quality_report.missing_data_pct:.1f}%")
    print(f"Negative Prices: {quality_report.negative_prices}")
    print(f"Price Inconsistencies: {quality_report.price_inconsistencies}")
    print(f"Price Outliers: {quality_report.price_outliers}")
    print(f"Volume Outliers: {quality_report.volume_outliers}")
    
    if quality_report.critical_issues:
        print("\nCritical Issues:")
        for issue in quality_report.critical_issues:
            print(f"  - {issue}")
            
    if quality_report.warnings:
        print("\nWarnings:")
        for warning in quality_report.warnings:
            print(f"  - {warning}")
            
    if quality_report.recommendations:
        print("\nRecommendations:")
        for rec in quality_report.recommendations:
            print(f"  - {rec}")
    
    # Test data cleaning
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_data(sample_data, quality_report)
    
    print(f"\nData cleaned: {len(sample_data)} -> {len(cleaned_data)} records")
    
    # Verify cleaning
    post_clean_report = quality_controller.assess_data_quality(cleaned_data, 'TEST_CLEANED')
    print(f"Quality score after cleaning: {post_clean_report.overall_quality_score:.1f}")
