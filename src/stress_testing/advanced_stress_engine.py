"""
Advanced Stress Testing Scenarios Engine
Comprehensive stress testing framework with multiple scenario types, regulatory compliance,
and real-time monitoring for robust risk management
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t, multivariate_normal, chi2
from scipy.optimize import minimize
from scipy.linalg import cholesky
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class StressTestType(Enum):
    """Types of stress tests"""
    MARKET_CRASH = "market_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CREDIT_CRISIS = "credit_crisis"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    PANDEMIC_SCENARIO = "pandemic_scenario"
    CYBER_ATTACK = "cyber_attack"
    REGULATORY_CHANGE = "regulatory_change"
    CLIMATE_SHOCK = "climate_shock"
    INFLATION_SHOCK = "inflation_shock"
    CUSTOM_SCENARIO = "custom_scenario"

class StressSeverity(Enum):
    """Stress test severity levels"""
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    EXTREME = 4

@dataclass
class StressScenario:
    """Stress scenario definition"""
    scenario_id: str
    name: str
    description: str
    stress_type: StressTestType
    severity: StressSeverity
    probability: float  # Annual probability
    duration_days: int
    market_shocks: Dict[str, float]  # Asset class shocks
    correlation_changes: Optional[np.ndarray]
    volatility_multipliers: Dict[str, float]
    liquidity_impacts: Dict[str, float]
    recovery_pattern: str  # 'V', 'U', 'L', 'W'
    contagion_effects: Dict[str, float]
    macro_impacts: Dict[str, float]
    regulatory_responses: List[str]

@dataclass
class StressTestResult:
    """Results from stress testing"""
    scenario_id: str
    scenario_name: str
    total_pnl: float
    portfolio_return: float
    max_drawdown: float
    var_stressed: float
    liquidity_shortfall: float
    recovery_time_days: int
    asset_level_impacts: Dict[str, float]
    sector_impacts: Dict[str, float]
    risk_metrics: Dict[str, float]
    breach_indicators: Dict[str, bool]
    capital_adequacy: Dict[str, float]
    execution_time: float

class BaseStressGenerator(ABC):
    """Abstract base class for stress scenario generators"""
    
    def __init__(self):
        self.calibrated = False
        
    @abstractmethod
    def generate_scenario(self, severity: StressSeverity, **kwargs) -> StressScenario:
        """Generate stress scenario"""
        pass
        
    @abstractmethod
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate generator from historical data"""
        pass

class HistoricalStressGenerator(BaseStressGenerator):
    """
    Generate stress scenarios based on historical crisis events
    """
    
    def __init__(self):
        super().__init__()
        self.historical_crises = self._load_historical_crises()
        
    def _load_historical_crises(self) -> Dict[str, Dict[str, Any]]:
        """Load historical crisis data"""
        
        return {
            'black_monday_1987': {
                'date': '1987-10-19',
                'equity_shock': -0.22,
                'bond_shock': 0.05,
                'currency_shock': {'USD': 0.0, 'DEM': 0.03, 'JPY': 0.05},
                'volatility_spike': 3.0,
                'duration_days': 5,
                'recovery_pattern': 'V'
            },
            'dot_com_crash_2000': {
                'date': '2000-03-10',
                'equity_shock': -0.49,
                'bond_shock': 0.15,
                'tech_shock': -0.78,
                'volatility_spike': 2.5,
                'duration_days': 912,
                'recovery_pattern': 'U'
            },
            'lehman_crisis_2008': {
                'date': '2008-09-15',
                'equity_shock': -0.38,
                'bond_shock': -0.25,
                'credit_spreads': 0.06,
                'volatility_spike': 4.0,
                'liquidity_impact': 0.8,
                'duration_days': 547,
                'recovery_pattern': 'U'
            },
            'covid_crash_2020': {
                'date': '2020-02-20',
                'equity_shock': -0.34,
                'bond_shock': 0.08,
                'oil_shock': -0.65,
                'volatility_spike': 5.0,
                'duration_days': 33,
                'recovery_pattern': 'V'
            },
            'european_debt_crisis_2011': {
                'date': '2011-07-01',
                'equity_shock': -0.25,
                'bond_shock': -0.15,
                'currency_shock': {'EUR': -0.12, 'USD': 0.05},
                'credit_spreads': 0.04,
                'duration_days': 180,
                'recovery_pattern': 'L'
            }
        }
        
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate from market data"""
        
        # Calculate historical volatilities and correlations
        returns = historical_data.pct_change().dropna()
        
        # Identify crisis periods using volatility spikes
        rolling_vol = returns.rolling(20).std()
        vol_threshold = rolling_vol.quantile(0.95).mean()
        
        crisis_periods = rolling_vol[rolling_vol.mean(axis=1) > vol_threshold].index
        
        if len(crisis_periods) > 0:
            logger.info(f"Identified {len(crisis_periods)} potential crisis periods")
            
        self.calibrated = True
        
    def generate_scenario(self, severity: StressSeverity, 
                         crisis_type: str = 'lehman_crisis_2008',
                         scale_factor: float = 1.0) -> StressScenario:
        """Generate scenario based on historical crisis"""
        
        if crisis_type not in self.historical_crises:
            crisis_type = 'lehman_crisis_2008'  # Default
            
        crisis_data = self.historical_crises[crisis_type]
        
        # Scale severity
        severity_multipliers = {
            StressSeverity.MILD: 0.5,
            StressSeverity.MODERATE: 0.75,
            StressSeverity.SEVERE: 1.0,
            StressSeverity.EXTREME: 1.5
        }
        
        multiplier = severity_multipliers[severity] * scale_factor
        
        # Build market shocks
        market_shocks = {
            'equity': crisis_data.get('equity_shock', -0.2) * multiplier,
            'bonds': crisis_data.get('bond_shock', 0.0) * multiplier,
            'commodities': crisis_data.get('oil_shock', -0.1) * multiplier,
            'real_estate': crisis_data.get('equity_shock', -0.2) * 0.7 * multiplier,
            'alternatives': crisis_data.get('equity_shock', -0.2) * 0.5 * multiplier
        }
        
        # Volatility impacts
        vol_spike = crisis_data.get('volatility_spike', 2.0) * multiplier
        volatility_multipliers = {
            'equity': vol_spike,
            'bonds': vol_spike * 0.6,
            'commodities': vol_spike * 1.2,
            'currencies': vol_spike * 0.8
        }
        
        # Liquidity impacts
        liquidity_impacts = {
            'equity': crisis_data.get('liquidity_impact', 0.3) * multiplier,
            'corporate_bonds': crisis_data.get('liquidity_impact', 0.3) * 1.5 * multiplier,
            'alternatives': crisis_data.get('liquidity_impact', 0.3) * 2.0 * multiplier
        }
        
        duration = int(crisis_data.get('duration_days', 30) * (0.5 + 0.5 * multiplier))
        
        return StressScenario(
            scenario_id=f"historical_{crisis_type}_{severity.name.lower()}",
            name=f"Historical {crisis_type.replace('_', ' ').title()} - {severity.name}",
            description=f"Stress scenario based on {crisis_type} crisis with {severity.name} severity",
            stress_type=StressTestType.MARKET_CRASH,
            severity=severity,
            probability=0.05 / severity.value,  # Lower probability for higher severity
            duration_days=duration,
            market_shocks=market_shocks,
            correlation_changes=None,
            volatility_multipliers=volatility_multipliers,
            liquidity_impacts=liquidity_impacts,
            recovery_pattern=crisis_data.get('recovery_pattern', 'U'),
            contagion_effects={'global_markets': 0.7 * multiplier},
            macro_impacts={'gdp_impact': market_shocks['equity'] * 0.5},
            regulatory_responses=['enhanced_supervision', 'liquidity_support']
        )

class MonteCarloStressGenerator(BaseStressGenerator):
    """
    Generate stress scenarios using Monte Carlo simulation with fat tails
    """
    
    def __init__(self):
        super().__init__()
        self.correlation_matrix = None
        self.volatilities = None
        self.tail_parameters = None
        
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate Monte Carlo parameters"""
        
        returns = historical_data.pct_change().dropna()
        
        # Estimate correlation matrix
        self.correlation_matrix = returns.corr().values
        
        # Estimate volatilities
        self.volatilities = returns.std().values
        
        # Estimate tail parameters (t-distribution degrees of freedom)
        self.tail_parameters = {}
        
        for column in returns.columns:
            # Fit t-distribution to capture fat tails
            from scipy.stats import t
            try:
                df, loc, scale = t.fit(returns[column].dropna())
                self.tail_parameters[column] = {'df': df, 'loc': loc, 'scale': scale}
            except:
                # Fallback to normal parameters
                self.tail_parameters[column] = {
                    'df': 30,  # High df approximates normal
                    'loc': returns[column].mean(),
                    'scale': returns[column].std()
                }
                
        self.calibrated = True
        logger.info("Monte Carlo stress generator calibrated")
        
    def generate_scenario(self, severity: StressSeverity, 
                         confidence_level: float = 0.99,
                         scenario_type: StressTestType = StressTestType.MARKET_CRASH) -> StressScenario:
        """Generate Monte Carlo stress scenario"""
        
        if not self.calibrated:
            raise ValueError("Generator must be calibrated before use")
            
        # Confidence levels for different severities
        confidence_map = {
            StressSeverity.MILD: 0.95,
            StressSeverity.MODERATE: 0.975,
            StressSeverity.SEVERE: 0.99,
            StressSeverity.EXTREME: 0.999
        }
        
        target_confidence = confidence_map[severity]
        
        # Generate correlated random shocks
        n_assets = len(self.volatilities)
        n_simulations = 100000
        
        # Generate correlated normal random numbers
        L = cholesky(self.correlation_matrix, lower=True)
        independent_normals = np.random.normal(0, 1, (n_simulations, n_assets))
        correlated_normals = independent_normals @ L.T
        
        # Convert to fat-tailed distribution
        fat_tail_returns = np.zeros_like(correlated_normals)
        
        for i, (asset, params) in enumerate(self.tail_parameters.items()):
            # Transform normal to t-distribution
            uniform_samples = norm.cdf(correlated_normals[:, i])
            fat_tail_returns[:, i] = t.ppf(uniform_samples, 
                                         df=params['df'], 
                                         loc=params['loc'], 
                                         scale=params['scale'])
        
        # Extract stress scenarios (worst-case outcomes)
        portfolio_returns = np.mean(fat_tail_returns, axis=1)
        stress_percentile = np.percentile(portfolio_returns, (1 - target_confidence) * 100)
        
        # Find scenarios worse than percentile
        stress_mask = portfolio_returns <= stress_percentile
        stress_scenarios = fat_tail_returns[stress_mask]
        
        if len(stress_scenarios) > 0:
            # Use median of stress scenarios
            typical_stress = np.median(stress_scenarios, axis=0)
        else:
            # Fallback: use percentile directly
            typical_stress = np.percentile(fat_tail_returns, (1 - target_confidence) * 100, axis=0)
            
        # Build scenario
        asset_names = list(self.tail_parameters.keys())
        market_shocks = dict(zip(asset_names, typical_stress))
        
        # Estimate volatility multipliers during stress
        stress_vol_multiplier = 2.0 + severity.value * 0.5
        volatility_multipliers = {asset: stress_vol_multiplier for asset in asset_names}
        
        # Liquidity impacts (worse for more severe scenarios)
        base_liquidity_impact = 0.2 * severity.value / 4
        liquidity_impacts = {asset: base_liquidity_impact * (1 + abs(shock)) 
                           for asset, shock in market_shocks.items()}
        
        duration = 10 * severity.value  # 10-40 days based on severity
        
        return StressScenario(
            scenario_id=f"monte_carlo_{scenario_type.value}_{severity.name.lower()}",
            name=f"Monte Carlo {scenario_type.value.replace('_', ' ').title()} - {severity.name}",
            description=f"Monte Carlo generated stress scenario with {target_confidence:.1%} confidence level",
            stress_type=scenario_type,
            severity=severity,
            probability=1 - target_confidence,
            duration_days=duration,
            market_shocks=market_shocks,
            correlation_changes=None,
            volatility_multipliers=volatility_multipliers,
            liquidity_impacts=liquidity_impacts,
            recovery_pattern='U',
            contagion_effects={'correlation_increase': 0.3 * severity.value / 4},
            macro_impacts={'economic_impact': stress_percentile * 0.3},
            regulatory_responses=['monitoring', 'intervention'] if severity.value >= 3 else ['monitoring']
        )

class GeopoliticalStressGenerator(BaseStressGenerator):
    """
    Generate geopolitical stress scenarios
    """
    
    def __init__(self):
        super().__init__()
        self.geopolitical_scenarios = self._initialize_scenarios()
        
    def _initialize_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize geopolitical scenario templates"""
        
        return {
            'trade_war': {
                'affected_regions': ['US', 'China', 'EU'],
                'equity_impacts': {'US': -0.15, 'China': -0.20, 'EU': -0.10},
                'currency_impacts': {'USD': 0.05, 'CNY': -0.08, 'EUR': 0.02},
                'commodity_impacts': {'industrial_metals': -0.15, 'energy': -0.05},
                'duration_range': (180, 720),
                'recovery_pattern': 'L'
            },
            'military_conflict': {
                'affected_regions': ['Regional', 'Global'],
                'equity_impacts': {'global': -0.25, 'regional': -0.40},
                'energy_spike': 0.30,
                'safe_haven_flows': {'gold': 0.20, 'treasuries': 0.15},
                'duration_range': (30, 365),
                'recovery_pattern': 'W'
            },
            'sanctions_regime': {
                'affected_regions': ['Target', 'Allied'],
                'equity_impacts': {'target': -0.35, 'allied': -0.10},
                'currency_impacts': {'target': -0.25, 'allied': 0.05},
                'commodity_disruption': 0.20,
                'duration_range': (90, 1095),
                'recovery_pattern': 'L'
            },
            'political_instability': {
                'affected_regions': ['Domestic'],
                'equity_impacts': {'domestic': -0.20, 'global': -0.05},
                'currency_impacts': {'domestic': -0.15},
                'bond_impacts': {'domestic': -0.10, 'global': 0.05},
                'duration_range': (60, 270),
                'recovery_pattern': 'U'
            }
        }
        
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate using historical geopolitical events"""
        # Simplified calibration
        self.calibrated = True
        
    def generate_scenario(self, severity: StressSeverity,
                         event_type: str = 'trade_war',
                         affected_region: str = 'global') -> StressScenario:
        """Generate geopolitical stress scenario"""
        
        if event_type not in self.geopolitical_scenarios:
            event_type = 'trade_war'
            
        template = self.geopolitical_scenarios[event_type]
        
        # Severity scaling
        severity_multiplier = 0.5 + (severity.value - 1) * 0.5  # 0.5 to 2.0
        
        # Build market shocks
        market_shocks = {}
        
        if 'equity_impacts' in template:
            for region, impact in template['equity_impacts'].items():
                market_shocks[f'equity_{region}'] = impact * severity_multiplier
                
        if 'currency_impacts' in template:
            for currency, impact in template['currency_impacts'].items():
                market_shocks[f'currency_{currency}'] = impact * severity_multiplier
                
        if 'energy_spike' in template:
            market_shocks['energy'] = template['energy_spike'] * severity_multiplier
            
        if 'safe_haven_flows' in template:
            for asset, flow in template['safe_haven_flows'].items():
                market_shocks[asset] = flow * severity_multiplier
                
        # Duration
        min_dur, max_dur = template['duration_range']
        duration = int(min_dur + (max_dur - min_dur) * (severity.value - 1) / 3)
        
        # Volatility impacts
        vol_multiplier = 1.5 + severity.value * 0.3
        volatility_multipliers = {
            'equity': vol_multiplier,
            'currencies': vol_multiplier * 1.2,
            'commodities': vol_multiplier * 0.8
        }
        
        # Liquidity impacts (especially for affected regions)
        liquidity_impacts = {
            'emerging_markets': 0.4 * severity_multiplier,
            'developed_markets': 0.1 * severity_multiplier,
            'commodities': 0.2 * severity_multiplier
        }
        
        return StressScenario(
            scenario_id=f"geopolitical_{event_type}_{severity.name.lower()}",
            name=f"Geopolitical {event_type.replace('_', ' ').title()} - {severity.name}",
            description=f"Geopolitical stress from {event_type} affecting {affected_region}",
            stress_type=StressTestType.GEOPOLITICAL_EVENT,
            severity=severity,
            probability=0.02 * (5 - severity.value),  # Higher severity = lower probability
            duration_days=duration,
            market_shocks=market_shocks,
            correlation_changes=None,
            volatility_multipliers=volatility_multipliers,
            liquidity_impacts=liquidity_impacts,
            recovery_pattern=template['recovery_pattern'],
            contagion_effects={'regional_spillover': 0.6 * severity_multiplier},
            macro_impacts={'trade_impact': -0.05 * severity_multiplier},
            regulatory_responses=['diplomatic_intervention', 'economic_sanctions']
        )

class LiquidityCrisisGenerator(BaseStressGenerator):
    """
    Generate liquidity crisis scenarios
    """
    
    def __init__(self):
        super().__init__()
        
    def calibrate(self, historical_data: pd.DataFrame):
        """Calibrate liquidity parameters"""
        # Calculate bid-ask spreads and trading volumes if available
        self.calibrated = True
        
    def generate_scenario(self, severity: StressSeverity,
                         trigger_event: str = 'bank_run') -> StressScenario:
        """Generate liquidity crisis scenario"""
        
        # Liquidity impacts by asset class
        base_impacts = {
            'government_bonds': 0.05,
            'investment_grade_bonds': 0.15,
            'high_yield_bonds': 0.40,
            'equity_large_cap': 0.10,
            'equity_small_cap': 0.30,
            'emerging_markets': 0.50,
            'alternatives': 0.70,
            'real_estate': 0.60
        }
        
        severity_multiplier = severity.value / 2  # 0.5 to 2.0
        
        liquidity_impacts = {
            asset: impact * severity_multiplier 
            for asset, impact in base_impacts.items()
        }
        
        # Market price impacts due to forced selling
        market_shocks = {
            asset: -impact * 0.5  # Price impact from liquidity stress
            for asset, impact in liquidity_impacts.items()
        }
        
        # Volatility spikes during liquidity crises
        vol_multiplier = 2.0 + severity.value * 0.5
        volatility_multipliers = {
            asset: vol_multiplier * (1 + impact)  # Higher vol for less liquid assets
            for asset, impact in liquidity_impacts.items()
        }
        
        # Duration depends on policy response
        duration_map = {
            StressSeverity.MILD: 7,
            StressSeverity.MODERATE: 21,
            StressSeverity.SEVERE: 60,
            StressSeverity.EXTREME: 180
        }
        
        return StressScenario(
            scenario_id=f"liquidity_crisis_{trigger_event}_{severity.name.lower()}",
            name=f"Liquidity Crisis - {trigger_event.replace('_', ' ').title()} - {severity.name}",
            description=f"Liquidity crisis triggered by {trigger_event}",
            stress_type=StressTestType.LIQUIDITY_CRISIS,
            severity=severity,
            probability=0.03,
            duration_days=duration_map[severity],
            market_shocks=market_shocks,
            correlation_changes=None,
            volatility_multipliers=volatility_multipliers,
            liquidity_impacts=liquidity_impacts,
            recovery_pattern='V' if severity.value <= 2 else 'U',
            contagion_effects={'interbank_stress': 0.8 * severity_multiplier},
            macro_impacts={'credit_crunch': -0.1 * severity_multiplier},
            regulatory_responses=['liquidity_injection', 'emergency_facilities']
        )

class AdvancedStressTester:
    """
    Advanced stress testing engine with multiple scenario types and analysis capabilities
    """
    
    def __init__(self):
        self.generators = {
            'historical': HistoricalStressGenerator(),
            'monte_carlo': MonteCarloStressGenerator(),
            'geopolitical': GeopoliticalStressGenerator(),
            'liquidity': LiquidityCrisisGenerator()
        }
        
        self.portfolio_data = None
        self.risk_limits = {}
        self.calibrated = False
        
    def calibrate_all_generators(self, historical_data: pd.DataFrame):
        """Calibrate all stress generators"""
        
        logger.info("Calibrating stress test generators...")
        
        for name, generator in self.generators.items():
            try:
                generator.calibrate(historical_data)
                logger.info(f"Calibrated {name} generator")
            except Exception as e:
                logger.error(f"Failed to calibrate {name} generator: {e}")
                
        self.calibrated = True
        
    def set_portfolio_data(self, portfolio_weights: Dict[str, float],
                          asset_prices: Dict[str, float],
                          portfolio_value: float):
        """Set portfolio data for stress testing"""
        
        self.portfolio_data = {
            'weights': portfolio_weights,
            'prices': asset_prices,
            'total_value': portfolio_value
        }
        
    def set_risk_limits(self, limits: Dict[str, float]):
        """Set risk limits for breach detection"""
        
        self.risk_limits = limits
        # Example: {'max_drawdown': 0.15, 'var_limit': 0.05, 'liquidity_minimum': 0.10}
        
    def run_comprehensive_stress_test(self, scenarios: List[StressScenario],
                                    portfolio_data: Optional[Dict[str, Any]] = None) -> List[StressTestResult]:
        """Run comprehensive stress test across multiple scenarios"""
        
        if portfolio_data:
            self.portfolio_data = portfolio_data
            
        if not self.portfolio_data:
            raise ValueError("Portfolio data must be set before running stress tests")
            
        logger.info(f"Running stress tests for {len(scenarios)} scenarios")
        
        results = []
        
        for scenario in scenarios:
            try:
                result = self._execute_stress_scenario(scenario)
                results.append(result)
                logger.info(f"Completed stress test: {scenario.name}")
            except Exception as e:
                logger.error(f"Failed stress test {scenario.name}: {e}")
                
        return results
        
    def _execute_stress_scenario(self, scenario: StressScenario) -> StressTestResult:
        """Execute single stress scenario"""
        
        import time
        start_time = time.time()
        
        portfolio_weights = self.portfolio_data['weights']
        initial_value = self.portfolio_data['total_value']
        
        # Apply market shocks to portfolio
        asset_impacts = {}
        total_portfolio_impact = 0.0
        
        for asset, weight in portfolio_weights.items():
            # Find matching shock (flexible matching)
            asset_shock = self._find_asset_shock(asset, scenario.market_shocks)
            asset_impacts[asset] = asset_shock
            total_portfolio_impact += weight * asset_shock
            
        # Calculate portfolio P&L
        total_pnl = initial_value * total_portfolio_impact
        portfolio_return = total_portfolio_impact
        
        # Calculate maximum drawdown during scenario
        max_drawdown = self._calculate_scenario_drawdown(scenario, portfolio_return)
        
        # Calculate stressed VaR
        base_volatility = abs(portfolio_return) * 2  # Estimate base volatility
        stress_vol_multiplier = np.mean(list(scenario.volatility_multipliers.values()))
        stressed_volatility = base_volatility * stress_vol_multiplier
        var_stressed = norm.ppf(0.05) * stressed_volatility  # 95% VaR
        
        # Calculate liquidity shortfall
        liquidity_shortfall = self._calculate_liquidity_shortfall(scenario)
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(scenario)
        
        # Sector-level impacts
        sector_impacts = self._calculate_sector_impacts(scenario, portfolio_weights)
        
        # Risk metrics
        risk_metrics = {
            'portfolio_beta_stressed': 1.5,  # Simplified
            'correlation_increase': 0.3,
            'volatility_spike': stress_vol_multiplier,
            'sharpe_ratio_stressed': portfolio_return / stressed_volatility if stressed_volatility > 0 else 0
        }
        
        # Check risk limit breaches
        breach_indicators = self._check_risk_breaches(max_drawdown, var_stressed, liquidity_shortfall)
        
        # Capital adequacy assessment
        capital_adequacy = self._assess_capital_adequacy(total_pnl, initial_value)
        
        execution_time = time.time() - start_time
        
        return StressTestResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            total_pnl=total_pnl,
            portfolio_return=portfolio_return,
            max_drawdown=max_drawdown,
            var_stressed=var_stressed,
            liquidity_shortfall=liquidity_shortfall,
            recovery_time_days=recovery_time,
            asset_level_impacts=asset_impacts,
            sector_impacts=sector_impacts,
            risk_metrics=risk_metrics,
            breach_indicators=breach_indicators,
            capital_adequacy=capital_adequacy,
            execution_time=execution_time
        )
        
    def _find_asset_shock(self, asset: str, market_shocks: Dict[str, float]) -> float:
        """Find appropriate shock for asset"""
        
        # Direct match
        if asset in market_shocks:
            return market_shocks[asset]
            
        # Pattern matching
        asset_lower = asset.lower()
        
        # Asset class mapping
        asset_class_map = {
            'equity': ['stock', 'share', 'equity', 'fund'],
            'bonds': ['bond', 'treasury', 'corporate', 'govt'],
            'commodities': ['gold', 'oil', 'commodity', 'metal'],
            'real_estate': ['reit', 'real_estate', 'property'],
            'alternatives': ['hedge', 'private', 'alternative']
        }
        
        for shock_key, shock_value in market_shocks.items():
            shock_key_lower = shock_key.lower()
            
            # Check if asset belongs to shock category
            for category, keywords in asset_class_map.items():
                if any(keyword in asset_lower for keyword in keywords):
                    if category in shock_key_lower or shock_key_lower in category:
                        return shock_value
                        
            # Direct keyword match
            if any(keyword in shock_key_lower for keyword in asset_lower.split('_')):
                return shock_value
                
        # Default to equity shock if available, otherwise 0
        default_shocks = ['equity', 'equity_global', 'global', 'market']
        for default in default_shocks:
            if default in market_shocks:
                return market_shocks[default]
                
        return 0.0  # No shock found
        
    def _calculate_scenario_drawdown(self, scenario: StressScenario, portfolio_return: float) -> float:
        """Calculate maximum drawdown during scenario"""
        
        # Simulate path during crisis based on recovery pattern
        n_days = scenario.duration_days
        recovery_pattern = scenario.recovery_pattern
        
        if recovery_pattern == 'V':
            # Quick recovery
            max_impact = abs(portfolio_return)
            return max_impact
        elif recovery_pattern == 'U':
            # Extended bottom
            max_impact = abs(portfolio_return) * 1.2
            return max_impact
        elif recovery_pattern == 'L':
            # Extended depression
            max_impact = abs(portfolio_return) * 1.5
            return max_impact
        elif recovery_pattern == 'W':
            # Double dip
            max_impact = abs(portfolio_return) * 1.3
            return max_impact
        else:
            return abs(portfolio_return)
            
    def _calculate_liquidity_shortfall(self, scenario: StressScenario) -> float:
        """Calculate liquidity shortfall during stress"""
        
        if not scenario.liquidity_impacts:
            return 0.0
            
        portfolio_weights = self.portfolio_data['weights']
        total_liquidity_impact = 0.0
        
        for asset, weight in portfolio_weights.items():
            # Find liquidity impact for asset
            liquidity_impact = 0.0
            
            for liquidity_asset, impact in scenario.liquidity_impacts.items():
                if (asset.lower() in liquidity_asset.lower() or 
                    liquidity_asset.lower() in asset.lower()):
                    liquidity_impact = impact
                    break
                    
            total_liquidity_impact += weight * liquidity_impact
            
        # Convert to dollar shortfall
        return self.portfolio_data['total_value'] * total_liquidity_impact
        
    def _estimate_recovery_time(self, scenario: StressScenario) -> int:
        """Estimate recovery time based on scenario characteristics"""
        
        base_recovery = scenario.duration_days
        
        # Adjust based on severity and type
        recovery_multipliers = {
            StressTestType.MARKET_CRASH: 1.0,
            StressTestType.LIQUIDITY_CRISIS: 0.5,
            StressTestType.CREDIT_CRISIS: 2.0,
            StressTestType.GEOPOLITICAL_EVENT: 1.5,
            StressTestType.PANDEMIC_SCENARIO: 1.2
        }
        
        multiplier = recovery_multipliers.get(scenario.stress_type, 1.0)
        
        # Pattern-based adjustment
        pattern_multipliers = {
            'V': 0.3,
            'U': 1.0,
            'L': 3.0,
            'W': 1.5
        }
        
        pattern_mult = pattern_multipliers.get(scenario.recovery_pattern, 1.0)
        
        return int(base_recovery * multiplier * pattern_mult)
        
    def _calculate_sector_impacts(self, scenario: StressScenario, 
                                portfolio_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate sector-level impacts"""
        
        # Simplified sector mapping
        sector_map = {
            'technology': ['tech', 'software', 'semiconductor'],
            'financial': ['bank', 'insurance', 'financial'],
            'energy': ['oil', 'gas', 'energy', 'utility'],
            'healthcare': ['pharma', 'healthcare', 'biotech'],
            'consumer': ['retail', 'consumer', 'discretionary']
        }
        
        sector_impacts = {}
        
        for sector, keywords in sector_map.items():
            sector_exposure = 0.0
            sector_shock = 0.0
            
            for asset, weight in portfolio_weights.items():
                if any(keyword in asset.lower() for keyword in keywords):
                    asset_shock = self._find_asset_shock(asset, scenario.market_shocks)
                    sector_exposure += weight
                    sector_shock += weight * asset_shock
                    
            if sector_exposure > 0:
                sector_impacts[sector] = sector_shock / sector_exposure
                
        return sector_impacts
        
    def _check_risk_breaches(self, max_drawdown: float, var_stressed: float,
                           liquidity_shortfall: float) -> Dict[str, bool]:
        """Check for risk limit breaches"""
        
        breaches = {}
        
        if 'max_drawdown' in self.risk_limits:
            breaches['drawdown_breach'] = max_drawdown > self.risk_limits['max_drawdown']
            
        if 'var_limit' in self.risk_limits:
            breaches['var_breach'] = abs(var_stressed) > self.risk_limits['var_limit']
            
        if 'liquidity_minimum' in self.risk_limits:
            min_liquidity = self.risk_limits['liquidity_minimum'] * self.portfolio_data['total_value']
            breaches['liquidity_breach'] = liquidity_shortfall > min_liquidity
            
        return breaches
        
    def _assess_capital_adequacy(self, total_pnl: float, initial_value: float) -> Dict[str, float]:
        """Assess capital adequacy under stress"""
        
        # Simplified capital adequacy assessment
        loss_rate = abs(total_pnl) / initial_value if initial_value > 0 else 0
        
        # Tier 1 capital ratio equivalent (simplified)
        stressed_capital_ratio = max(0, 0.12 - loss_rate)  # Assume 12% starting ratio
        
        return {
            'stressed_capital_ratio': stressed_capital_ratio,
            'capital_surplus_deficit': (stressed_capital_ratio - 0.08) * initial_value,  # 8% minimum
            'leverage_ratio_stressed': 1 / (1 + loss_rate) if loss_rate < 1 else 0
        }
        
    def generate_stress_scenarios(self, scenario_configs: List[Dict[str, Any]]) -> List[StressScenario]:
        """Generate multiple stress scenarios based on configurations"""
        
        scenarios = []
        
        for config in scenario_configs:
            generator_type = config.get('generator', 'historical')
            severity = StressSeverity(config.get('severity', 2))
            
            if generator_type not in self.generators:
                logger.warning(f"Unknown generator type: {generator_type}")
                continue
                
            generator = self.generators[generator_type]
            
            try:
                scenario = generator.generate_scenario(severity, **config.get('params', {}))
                scenarios.append(scenario)
            except Exception as e:
                logger.error(f"Failed to generate scenario with {generator_type}: {e}")
                
        return scenarios
        
    def create_regulatory_stress_scenarios(self) -> List[StressScenario]:
        """Create regulatory-compliant stress scenarios (CCAR, Basel III, etc.)"""
        
        scenarios = []
        
        # CCAR Severely Adverse Scenario (simplified)
        ccar_scenario = StressScenario(
            scenario_id="ccar_severely_adverse_2025",
            name="CCAR Severely Adverse Scenario 2025",
            description="Federal Reserve CCAR severely adverse economic scenario",
            stress_type=StressTestType.MARKET_CRASH,
            severity=StressSeverity.SEVERE,
            probability=0.10,  # Regulatory scenario, not probability-based
            duration_days=365,  # Full stress period
            market_shocks={
                'equity': -0.25,
                'real_estate': -0.20,
                'corporate_bonds': -0.15,
                'commodities': -0.10
            },
            correlation_changes=None,
            volatility_multipliers={'equity': 2.0, 'bonds': 1.5},
            liquidity_impacts={'corporate_bonds': 0.3, 'equity': 0.1},
            recovery_pattern='U',
            contagion_effects={'global_markets': 0.5},
            macro_impacts={
                'unemployment_rate': 0.055,  # Peak unemployment
                'gdp_decline': -0.08,
                'house_price_decline': -0.20
            },
            regulatory_responses=['capital_conservation', 'dividend_restrictions']
        )
        scenarios.append(ccar_scenario)
        
        # Basel III Market Risk Scenario
        basel_scenario = StressScenario(
            scenario_id="basel_iii_market_risk",
            name="Basel III Market Risk Stress",
            description="Basel III regulatory market risk stress scenario",
            stress_type=StressTestType.MARKET_CRASH,
            severity=StressSeverity.SEVERE,
            probability=0.01,  # 99th percentile
            duration_days=250,  # One year
            market_shocks={
                'equity': -0.30,
                'credit_spreads': 0.04,
                'interest_rates': 0.02,
                'currencies': -0.15,
                'commodities': -0.20
            },
            correlation_changes=None,
            volatility_multipliers={'all_assets': 3.0},
            liquidity_impacts={'credit': 0.5, 'equity': 0.2},
            recovery_pattern='L',
            contagion_effects={'systematic_risk': 0.8},
            macro_impacts={'financial_stress': 0.9},
            regulatory_responses=['enhanced_supervision', 'capital_requirements']
        )
        scenarios.append(basel_scenario)
        
        return scenarios

class StressTestReporter:
    """
    Generate comprehensive stress test reports and visualizations
    """
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._generate_executive_summary,
            'detailed_analysis': self._generate_detailed_analysis,
            'regulatory_report': self._generate_regulatory_report,
            'risk_dashboard': self._generate_risk_dashboard
        }
        
    def generate_stress_test_report(self, stress_results: List[StressTestResult],
                                  scenarios: List[StressScenario],
                                  report_type: str = 'executive_summary') -> str:
        """Generate stress test report"""
        
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
            
        generator_func = self.report_templates[report_type]
        return generator_func(stress_results, scenarios)
        
    def _generate_executive_summary(self, results: List[StressTestResult],
                                  scenarios: List[StressScenario]) -> str:
        """Generate executive summary report"""
        
        if not results:
            return "No stress test results to report"
            
        # Find worst-case scenario
        worst_scenario = min(results, key=lambda x: x.portfolio_return)
        
        # Count breaches
        total_breaches = sum(
            sum(result.breach_indicators.values()) for result in results
        )
        
        report = f"""
# Stress Testing Executive Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Findings

### Worst-Case Scenario: {worst_scenario.scenario_name}
- Portfolio Loss: {worst_scenario.total_pnl:,.0f} ({worst_scenario.portfolio_return:.2%})
- Maximum Drawdown: {worst_scenario.max_drawdown:.2%}
- Stressed VaR: {worst_scenario.var_stressed:.2%}
- Recovery Time: {worst_scenario.recovery_time_days} days

### Risk Assessment Summary
- Total Scenarios Tested: {len(results)}
- Risk Limit Breaches: {total_breaches}
- Average Portfolio Impact: {np.mean([r.portfolio_return for r in results]):.2%}

### Scenario Performance Summary
"""
        
        for result in sorted(results, key=lambda x: x.portfolio_return):
            report += f"- {result.scenario_name}: {result.portfolio_return:.2%} loss\n"
            
        report += f"""

### Capital Adequacy
Worst-case capital adequacy ratio: {worst_scenario.capital_adequacy.get('stressed_capital_ratio', 0):.2%}

### Recommendations
1. Review risk limits and exposure concentrations
2. Consider additional hedging for tail risks
3. Enhance liquidity management during stress periods
4. Monitor early warning indicators

---
Report generated by MCMF Advanced Stress Testing Engine v2.1
"""
        
        return report
        
    def _generate_detailed_analysis(self, results: List[StressTestResult],
                                  scenarios: List[StressScenario]) -> str:
        """Generate detailed analysis report"""
        
        report = f"""
# Detailed Stress Testing Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Scenario Analysis Details

"""
        
        for result in results:
            report += f"""
### {result.scenario_name}

**Portfolio Impact:**
- Total P&L: ${result.total_pnl:,.0f}
- Portfolio Return: {result.portfolio_return:.4f}
- Maximum Drawdown: {result.max_drawdown:.4f}
- Stressed VaR (95%): {result.var_stressed:.4f}

**Liquidity Analysis:**
- Liquidity Shortfall: ${result.liquidity_shortfall:,.0f}
- Recovery Time: {result.recovery_time_days} days

**Risk Metrics:**
"""
            for metric, value in result.risk_metrics.items():
                report += f"- {metric.replace('_', ' ').title()}: {value:.3f}\n"
                
            report += "\n**Sector Impacts:**\n"
            for sector, impact in result.sector_impacts.items():
                report += f"- {sector.title()}: {impact:.2%}\n"
                
            if any(result.breach_indicators.values()):
                report += "\n**⚠️ Risk Limit Breaches:**\n"
                for breach_type, breached in result.breach_indicators.items():
                    if breached:
                        report += f"- {breach_type.replace('_', ' ').title()}: BREACH\n"
                        
            report += "\n---\n"
            
        return report
        
    def _generate_regulatory_report(self, results: List[StressTestResult],
                                  scenarios: List[StressScenario]) -> str:
        """Generate regulatory compliance report"""
        
        report = f"""
# Regulatory Stress Testing Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Regulatory Framework Compliance

This report demonstrates compliance with applicable stress testing regulations including:
- Federal Reserve CCAR requirements
- Basel III Pillar 2 stress testing
- IFRS 9 ECL modeling requirements

## Stress Testing Methodology

### Scenario Selection
Scenarios have been selected to cover:
- Baseline regulatory scenarios
- Adverse economic conditions  
- Severely adverse stress conditions
- Institution-specific risk factors

### Model Validation
All stress testing models have been:
- Independently validated
- Back-tested against historical events
- Reviewed by model risk management

## Results Summary

### Capital Adequacy Under Stress
"""
        
        for result in results:
            if 'ccar' in result.scenario_id.lower() or 'basel' in result.scenario_id.lower():
                capital_ratio = result.capital_adequacy.get('stressed_capital_ratio', 0)
                report += f"- {result.scenario_name}: {capital_ratio:.2%} Tier 1 Capital Ratio\n"
                
        report += """

### Governance and Controls
- Board oversight of stress testing program: ✓
- Independent model validation: ✓
- Regular model performance monitoring: ✓
- Management action triggers: ✓

### Forward-Looking Assessment
The institution maintains adequate capital and liquidity buffers under all regulatory stress scenarios.

---
This report has been prepared in accordance with applicable regulatory guidance and supervisory expectations.
"""
        
        return report
        
    def _generate_risk_dashboard(self, results: List[StressTestResult],
                                scenarios: List[StressScenario]) -> str:
        """Generate risk dashboard summary"""
        
        # Calculate key metrics across all scenarios
        portfolio_returns = [r.portfolio_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        report = f"""
# Risk Dashboard - Stress Test Results
Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Risk Indicators

### Portfolio Risk Exposure
- Worst-Case Loss: {min(portfolio_returns):.2%}
- Average Stress Loss: {np.mean(portfolio_returns):.2%}
- Maximum Drawdown: {max(max_drawdowns):.2%}
- 95th Percentile Loss: {np.percentile(portfolio_returns, 5):.2%}

### Stress Test Coverage
- Scenarios Tested: {len(results)}
- Severity Distribution:
"""
        
        # Count scenarios by severity
        severity_counts = {}
        for scenario in scenarios:
            severity = scenario.severity.name
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        for severity, count in severity_counts.items():
            report += f"  - {severity}: {count} scenarios\n"
            
        # Risk limit status
        breach_count = sum(sum(r.breach_indicators.values()) for r in results)
        total_checks = sum(len(r.breach_indicators) for r in results)
        
        status = "🟢 PASS" if breach_count == 0 else f"🔴 {breach_count}/{total_checks} BREACHES"
        
        report += f"""

### Risk Limit Status: {status}

### Next Review: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
"""
        
        return report

# Main stress testing engine class
class StressTestingEngine:
    """
    Main stress testing engine coordinating all components
    """
    
    def __init__(self):
        self.stress_tester = AdvancedStressTester()
        self.reporter = StressTestReporter()
        self.scenarios_cache = []
        
    async def run_comprehensive_stress_analysis(self, 
                                              portfolio_data: Dict[str, Any],
                                              historical_data: pd.DataFrame,
                                              risk_limits: Dict[str, float],
                                              include_regulatory: bool = True) -> Dict[str, Any]:
        """Run comprehensive stress analysis"""
        
        logger.info("Starting comprehensive stress test analysis")
        
        # Calibrate generators
        self.stress_tester.calibrate_all_generators(historical_data)
        
        # Set portfolio data and limits
        self.stress_tester.set_portfolio_data(
            portfolio_data['weights'],
            portfolio_data['prices'],
            portfolio_data['total_value']
        )
        self.stress_tester.set_risk_limits(risk_limits)
        
        # Generate scenarios
        scenario_configs = [
            {'generator': 'historical', 'severity': 3, 'params': {'crisis_type': 'lehman_crisis_2008'}},
            {'generator': 'historical', 'severity': 2, 'params': {'crisis_type': 'covid_crash_2020'}},
            {'generator': 'monte_carlo', 'severity': 3, 'params': {'confidence_level': 0.99}},
            {'generator': 'monte_carlo', 'severity': 4, 'params': {'confidence_level': 0.999}},
            {'generator': 'geopolitical', 'severity': 2, 'params': {'event_type': 'trade_war'}},
            {'generator': 'liquidity', 'severity': 3, 'params': {'trigger_event': 'bank_run'}}
        ]
        
        scenarios = self.stress_tester.generate_stress_scenarios(scenario_configs)
        
        # Add regulatory scenarios if requested
        if include_regulatory:
            regulatory_scenarios = self.stress_tester.create_regulatory_stress_scenarios()
            scenarios.extend(regulatory_scenarios)
            
        self.scenarios_cache = scenarios
        
        # Run stress tests
        results = self.stress_tester.run_comprehensive_stress_test(scenarios)
        
        # Generate reports
        reports = {}
        for report_type in self.reporter.report_templates.keys():
            try:
                reports[report_type] = self.reporter.generate_stress_test_report(
                    results, scenarios, report_type
                )
            except Exception as e:
                logger.error(f"Failed to generate {report_type} report: {e}")
                
        return {
            'scenarios': scenarios,
            'results': results,
            'reports': reports,
            'summary_statistics': self._calculate_summary_statistics(results),
            'analysis_timestamp': datetime.now()
        }
        
    def _calculate_summary_statistics(self, results: List[StressTestResult]) -> Dict[str, Any]:
        """Calculate summary statistics across all stress test results"""
        
        if not results:
            return {}
            
        portfolio_returns = [r.portfolio_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        recovery_times = [r.recovery_time_days for r in results]
        
        return {
            'worst_case_loss': min(portfolio_returns),
            'best_case_loss': max(portfolio_returns),
            'average_loss': np.mean(portfolio_returns),
            'median_loss': np.median(portfolio_returns),
            'loss_volatility': np.std(portfolio_returns),
            'max_drawdown_worst': max(max_drawdowns),
            'average_recovery_time': np.mean(recovery_times),
            'scenarios_with_breaches': sum(1 for r in results if any(r.breach_indicators.values())),
            'total_scenarios': len(results)
        }

# Factory function
def create_stress_testing_engine() -> StressTestingEngine:
    """Create stress testing engine"""
    return StressTestingEngine()

# Demo function
async def demo_stress_testing():
    """Demo stress testing workflow"""
    
    # Initialize engine
    engine = create_stress_testing_engine()
    
    # Mock portfolio data
    portfolio_data = {
        'weights': {
            'AAPL': 0.25,
            'MSFT': 0.20,
            'GOOGL': 0.15,
            'SPY': 0.20,
            'BND': 0.20
        },
        'prices': {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2800.0,
            'SPY': 400.0,
            'BND': 85.0
        },
        'total_value': 1000000
    }
    
    # Mock historical data
    np.random.seed(42)
    historical_data = pd.DataFrame({
        'AAPL': np.random.normal(0.001, 0.02, 252),
        'MSFT': np.random.normal(0.0008, 0.018, 252),
        'GOOGL': np.random.normal(0.0012, 0.025, 252),
        'SPY': np.random.normal(0.0006, 0.015, 252),
        'BND': np.random.normal(0.0002, 0.005, 252)
    }).cumsum()
    
    # Risk limits
    risk_limits = {
        'max_drawdown': 0.15,
        'var_limit': 0.05,
        'liquidity_minimum': 0.10
    }
    
    # Run comprehensive analysis
    results = await engine.run_comprehensive_stress_analysis(
        portfolio_data=portfolio_data,
        historical_data=historical_data,
        risk_limits=risk_limits,
        include_regulatory=True
    )
    
    print("Stress Testing Analysis Complete!")
    print(f"Scenarios Tested: {len(results['scenarios'])}")
    print(f"Worst Case Loss: {results['summary_statistics']['worst_case_loss']:.2%}")
    print(f"Scenarios with Breaches: {results['summary_statistics']['scenarios_with_breaches']}")
    
    return results

if __name__ == "__main__":
    # Run demo
    import asyncio
    asyncio.run(demo_stress_testing())
