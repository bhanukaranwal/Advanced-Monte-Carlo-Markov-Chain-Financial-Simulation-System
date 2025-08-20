"""
Comprehensive data validation system for financial data
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
import logging
from scipy import stats
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    severity: ValidationSeverity
    rule_name: str
    message: str
    affected_records: int = 0
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationSummary:
    """Summary of all validation results"""
    total_validations: int
    passed: int
    failed: int
    results: List[ValidationResult]
    overall_valid: bool
    
    def get_by_severity(self, severity: ValidationSeverity) -> List[ValidationResult]:
        """Get validation results by severity level"""
        return [r for r in self.results if r.severity == severity]

class ValidationRule:
    """Base class for validation rules"""
    
    def __init__(self, name: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.severity = severity
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        """Validate data against this rule"""
        raise NotImplementedError

class SchemaValidationRule(ValidationRule):
    """Validate data schema (columns, types, etc.)"""
    
    def __init__(self, required_columns: List[str], optional_columns: List[str] = None):
        super().__init__("schema_validation", ValidationSeverity.CRITICAL)
        self.required_columns = required_columns
        self.optional_columns = optional_columns or []
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        missing_columns = []
        for col in self.required_columns:
            if col not in data.columns:
                missing_columns.append(col)
                
        if missing_columns:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"Missing required columns: {missing_columns}",
                affected_columns=missing_columns
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Schema validation passed"
        )

class DataTypeValidationRule(ValidationRule):
    """Validate column data types"""
    
    def __init__(self, column_types: Dict[str, str]):
        super().__init__("data_type_validation", ValidationSeverity.ERROR)
        self.column_types = column_types
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        type_issues = []
        
        for col, expected_type in self.column_types.items():
            if col not in data.columns:
                continue
                
            actual_type = str(data[col].dtype)
            
            # Check type compatibility
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(data[col]):
                    type_issues.append(f"{col}: expected numeric, got {actual_type}")
            elif expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    type_issues.append(f"{col}: expected datetime, got {actual_type}")
            elif expected_type not in actual_type.lower():
                type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")
                
        if type_issues:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"Data type issues: {type_issues}",
                details={'issues': type_issues}
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Data type validation passed"
        )

class RangeValidationRule(ValidationRule):
    """Validate data is within expected ranges"""
    
    def __init__(self, column_ranges: Dict[str, Tuple[float, float]]):
        super().__init__("range_validation", ValidationSeverity.WARNING)
        self.column_ranges = column_ranges
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        range_violations = {}
        total_violations = 0
        
        for col, (min_val, max_val) in self.column_ranges.items():
            if col not in data.columns:
                continue
                
            violations = ((data[col] < min_val) | (data[col] > max_val)).sum()
            if violations > 0:
                range_violations[col] = {
                    'violations': int(violations),
                    'min_expected': min_val,
                    'max_expected': max_val,
                    'min_actual': float(data[col].min()),
                    'max_actual': float(data[col].max())
                }
                total_violations += violations
                
        if range_violations:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"Range violations in {len(range_violations)} columns",
                affected_records=total_violations,
                affected_columns=list(range_violations.keys()),
                details=range_violations
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Range validation passed"
        )

class ConsistencyValidationRule(ValidationRule):
    """Validate internal data consistency"""
    
    def __init__(self):
        super().__init__("consistency_validation", ValidationSeverity.ERROR)
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        issues = []
        total_issues = 0
        
        # Price consistency checks
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in data.columns for col in price_cols):
            
            # High >= Low
            high_low_issues = (data['high'] < data['low']).sum()
            if high_low_issues > 0:
                issues.append(f"High < Low: {high_low_issues} records")
                total_issues += high_low_issues
                
            # High >= Open, Close
            high_open_issues = (data['high'] < data['open']).sum()
            high_close_issues = (data['high'] < data['close']).sum()
            if high_open_issues > 0:
                issues.append(f"High < Open: {high_open_issues} records")
                total_issues += high_open_issues
            if high_close_issues > 0:
                issues.append(f"High < Close: {high_close_issues} records")
                total_issues += high_close_issues
                
            # Low <= Open, Close
            low_open_issues = (data['low'] > data['open']).sum()
            low_close_issues = (data['low'] > data['close']).sum()
            if low_open_issues > 0:
                issues.append(f"Low > Open: {low_open_issues} records")
                total_issues += low_open_issues
            if low_close_issues > 0:
                issues.append(f"Low > Close: {low_close_issues} records")
                total_issues += low_close_issues
        
        # Volume consistency (non-negative)
        if 'volume' in data.columns:
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume: {negative_volume} records")
                total_issues += negative_volume
                
        if issues:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"Consistency issues found: {'; '.join(issues)}",
                affected_records=total_issues,
                details={'issues': issues}
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Consistency validation passed"
        )

class CompletenessValidationRule(ValidationRule):
    """Validate data completeness"""
    
    def __init__(self, max_missing_pct: float = 5.0):
        super().__init__("completeness_validation", ValidationSeverity.WARNING)
        self.max_missing_pct = max_missing_pct
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        if data.empty:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                rule_name=self.name,
                message="Dataset is empty"
            )
            
        missing_analysis = {}
        critical_columns = ['close']  # Columns that cannot be missing
        
        for col in data.columns:
            missing_count = data[col].isna().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            missing_analysis[col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
            
        # Check critical columns
        critical_issues = []
        for col in critical_columns:
            if col in missing_analysis and missing_analysis[col]['percentage'] > 0:
                critical_issues.append(f"{col}: {missing_analysis[col]['percentage']:.1f}% missing")
                
        # Check overall missing percentage
        overall_missing = data.isna().any(axis=1).sum()
        overall_pct = (overall_missing / len(data)) * 100
        
        if critical_issues:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                rule_name=self.name,
                message=f"Critical columns missing data: {'; '.join(critical_issues)}",
                details=missing_analysis
            )
            
        if overall_pct > self.max_missing_pct:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"High missing data rate: {overall_pct:.1f}% (threshold: {self.max_missing_pct}%)",
                affected_records=int(overall_missing),
                details=missing_analysis
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Completeness validation passed",
            details=missing_analysis
        )

class UniquenessValidationRule(ValidationRule):
    """Validate data uniqueness (no duplicate timestamps)"""
    
    def __init__(self):
        super().__init__("uniqueness_validation", ValidationSeverity.ERROR)
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        if not isinstance(data.index, pd.DatetimeIndex):
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                rule_name=self.name,
                message="No timestamp index to validate uniqueness"
            )
            
        duplicate_dates = data.index.duplicated().sum()
        
        if duplicate_dates > 0:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"Found {duplicate_dates} duplicate timestamps",
                affected_records=duplicate_dates
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Uniqueness validation passed"
        )

class OutlierValidationRule(ValidationRule):
    """Validate for excessive outliers"""
    
    def __init__(self, max_outlier_pct: float = 5.0, method: str = 'iqr'):
        super().__init__("outlier_validation", ValidationSeverity.WARNING)
        self.max_outlier_pct = max_outlier_pct
        self.method = method
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        outlier_analysis = {}
        total_outliers = 0
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            clean_data = data[col].dropna()
            if len(clean_data) == 0:
                continue
                
            outliers = self._detect_outliers(clean_data)
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(clean_data)) * 100
            
            outlier_analysis[col] = {
                'count': outlier_count,
                'percentage': float(outlier_pct),
                'threshold_exceeded': outlier_pct > self.max_outlier_pct
            }
            
            total_outliers += outlier_count
            
        excessive_outlier_columns = [
            col for col, analysis in outlier_analysis.items() 
            if analysis['threshold_exceeded']
        ]
        
        if excessive_outlier_columns:
            return ValidationResult(
                is_valid=False,
                severity=self.severity,
                rule_name=self.name,
                message=f"Excessive outliers in columns: {excessive_outlier_columns}",
                affected_records=total_outliers,
                affected_columns=excessive_outlier_columns,
                details=outlier_analysis
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name=self.name,
            message="Outlier validation passed",
            details=outlier_analysis
        )
        
    def _detect_outliers(self, series: pd.Series) -> pd.Index:
        """Detect outliers using specified method"""
        if self.method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
        elif self.method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            outliers = series[z_scores > 3]
            
        else:
            outliers = pd.Series([], dtype=series.dtype)
            
        return outliers.index

class CustomValidationRule(ValidationRule):
    """Custom validation rule with user-defined function"""
    
    def __init__(self, name: str, validation_func: Callable, severity: ValidationSeverity = ValidationSeverity.WARNING):
        super().__init__(name, severity)
        self.validation_func = validation_func
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationResult:
        try:
            result = self.validation_func(data, symbol)
            if isinstance(result, ValidationResult):
                return result
            elif isinstance(result, bool):
                return ValidationResult(
                    is_valid=result,
                    severity=self.severity if not result else ValidationSeverity.INFO,
                    rule_name=self.name,
                    message=f"Custom validation {'passed' if result else 'failed'}"
                )
            else:
                raise ValueError("Validation function must return ValidationResult or bool")
                
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_name=self.name,
                message=f"Custom validation failed with error: {str(e)}"
            )

class DataValidator:
    """Main data validation orchestrator"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rules: List[ValidationRule] = []
        
        # Add default rules
        self._add_default_rules()
        
    def _add_default_rules(self):
        """Add standard validation rules"""
        # Schema validation
        required_columns = ['close']  # Minimum required
        optional_columns = ['open', 'high', 'low', 'volume']
        self.add_rule(SchemaValidationRule(required_columns, optional_columns))
        
        # Data type validation
        column_types = {
            'open': 'numeric',
            'high': 'numeric', 
            'low': 'numeric',
            'close': 'numeric',
            'volume': 'numeric'
        }
        self.add_rule(DataTypeValidationRule(column_types))
        
        # Range validation (reasonable bounds)
        column_ranges = {
            'open': (0.01, 1000000),
            'high': (0.01, 1000000),
            'low': (0.01, 1000000),
            'close': (0.01, 1000000),
            'volume': (0, 1e12)
        }
        self.add_rule(RangeValidationRule(column_ranges))
        
        # Add other default rules
        self.add_rule(ConsistencyValidationRule())
        self.add_rule(CompletenessValidationRule())
        self.add_rule(UniquenessValidationRule())
        self.add_rule(OutlierValidationRule())
        
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule"""
        self.rules.append(rule)
        
    def remove_rule(self, rule_name: str):
        """Remove a validation rule by name"""
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        
    def validate(self, data: pd.DataFrame, symbol: str = None) -> ValidationSummary:
        """
        Run all validation rules on data
        
        Args:
            data: DataFrame to validate
            symbol: Symbol identifier for context
            
        Returns:
            ValidationSummary with all results
        """
        logger.info(f"Validating data for {symbol or 'unknown symbol'}")
        
        results = []
        
        for rule in self.rules:
            try:
                result = rule.validate(data, symbol)
                results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_name=rule.name,
                    message=f"Validation rule failed: {str(e)}"
                )
                results.append(error_result)
                
        # Calculate summary
        total_validations = len(results)
        passed = sum(1 for r in results if r.is_valid)
        failed = total_validations - passed
        
        # Overall validity (no critical or error failures)
        critical_errors = [r for r in results if not r.is_valid and r.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        overall_valid = len(critical_errors) == 0
        
        summary = ValidationSummary(
            total_validations=total_validations,
            passed=passed,
            failed=failed,
            results=results,
            overall_valid=overall_valid
        )
        
        logger.info(f"Validation complete: {passed}/{total_validations} passed, overall valid: {overall_valid}")
        
        return summary
        
    def validate_symbol_format(self, symbol: str) -> bool:
        """Validate symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
            
        # Basic symbol format validation
        symbol_pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'
        return bool(re.match(symbol_pattern, symbol.upper()))

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')  # Business days
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    prices = np.array(prices[1:])  # Remove initial price
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.randint(100000, 5000000, len(dates))
    }, index=dates)
    
    # Introduce some data issues for testing
    # Missing data
    sample_data.iloc[10:15, :] = np.nan
    
    # Price inconsistency
    sample_data.iloc[100, sample_data.columns.get_loc('low')] = sample_data.iloc[100, sample_data.columns.get_loc('high')] + 1
    
    # Outlier
    sample_data.iloc[200, sample_data.columns.get_loc('close')] *= 5
    
    # Duplicate index
    sample_data = pd.concat([sample_data, sample_data.iloc[[50]]])
    sample_data = sample_data.sort_index()
    
    # Test validation
    validator = DataValidator()
    
    # Add custom rule
    def custom_volume_rule(data: pd.DataFrame, symbol: str) -> ValidationResult:
        if 'volume' not in data.columns:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_name="custom_volume_check",
                message="Volume column missing"
            )
            
        avg_volume = data['volume'].mean()
        if avg_volume < 100000:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_name="custom_volume_check",
                message=f"Low average volume: {avg_volume:,.0f}"
            )
            
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_name="custom_volume_check",
            message="Volume validation passed"
        )
    
    validator.add_rule(CustomValidationRule("custom_volume_check", custom_volume_rule))
    
    # Run validation
    validation_summary = validator.validate(sample_data, "TEST")
    
    print("=== Validation Summary ===")
    print(f"Total Validations: {validation_summary.total_validations}")
    print(f"Passed: {validation_summary.passed}")
    print(f"Failed: {validation_summary.failed}")
    print(f"Overall Valid: {validation_summary.overall_valid}")
    
    print("\n=== Validation Results ===")
    for result in validation_summary.results:
        status = "✓" if result.is_valid else "✗"
        print(f"{status} {result.rule_name} ({result.severity.value}): {result.message}")
        if not result.is_valid and result.affected_records > 0:
            print(f"    Affected records: {result.affected_records}")
        if result.affected_columns:
            print(f"    Affected columns: {result.affected_columns}")
            
    # Show critical issues
    critical_issues = validation_summary.get_by_severity(ValidationSeverity.CRITICAL)
    if critical_issues:
        print("\n=== Critical Issues ===")
        for issue in critical_issues:
            print(f"CRITICAL: {issue.message}")
            
    # Show errors
    errors = validation_summary.get_by_severity(ValidationSeverity.ERROR)
    if errors:
        print("\n=== Errors ===")
        for error in errors:
            print(f"ERROR: {error.message}")
    
    print("\nValidation test completed!")
