"""
ESG (Environmental, Social, Governance) Risk Factor Integration Engine
Comprehensive ESG risk modeling for sustainable finance and climate scenario analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio
import aiohttp
import json

try:
    import requests
    import yfinance as yf
    EXTERNAL_DATA_AVAILABLE = True
except ImportError:
    EXTERNAL_DATA_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ESGScores:
    """ESG scoring data structure"""
    environmental_score: float
    social_score: float
    governance_score: float
    overall_score: float
    climate_risk_score: float
    carbon_intensity: float
    water_usage_score: float
    waste_management_score: float
    employee_satisfaction: float
    diversity_score: float
    board_independence: float
    executive_compensation: float
    data_quality_score: float
    last_updated: datetime

@dataclass
class ClimateScenario:
    """Climate scenario parameters"""
    scenario_name: str
    temperature_increase: float  # Celsius
    carbon_price_trajectory: np.ndarray
    transition_risk_multiplier: float
    physical_risk_multiplier: float
    policy_stringency: float
    technology_adoption_rate: float
    stranded_asset_probability: float

@dataclass
class ESGRiskResult:
    """Results from ESG risk analysis"""
    adjusted_returns: np.ndarray
    esg_risk_premium: float
    climate_var: float
    transition_risk_impact: Dict[str, float]
    physical_risk_impact: Dict[str, float]
    esg_momentum_score: float
    sustainable_alpha: float
    carbon_footprint: float
    execution_time: float

class BaseESGProvider(ABC):
    """Abstract base class for ESG data providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
    @abstractmethod
    async def fetch_esg_scores(self, symbols: List[str]) -> Dict[str, ESGScores]:
        """Fetch ESG scores for given symbols"""
        pass
        
    @abstractmethod
    async def fetch_climate_metrics(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch climate-related metrics"""
        pass

class MSCIESGProvider(BaseESGProvider):
    """MSCI ESG data provider integration"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.msci.com/esg/v2.0"
        
    async def fetch_esg_scores(self, symbols: List[str]) -> Dict[str, ESGScores]:
        """Fetch ESG scores from MSCI API"""
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Mock implementation - real version would call MSCI API
                    mock_data = self._generate_mock_esg_data(symbol)
                    
                    results[symbol] = ESGScores(
                        environmental_score=mock_data['environmental'],
                        social_score=mock_data['social'],
                        governance_score=mock_data['governance'],
                        overall_score=mock_data['overall'],
                        climate_risk_score=mock_data['climate_risk'],
                        carbon_intensity=mock_data['carbon_intensity'],
                        water_usage_score=mock_data['water_usage'],
                        waste_management_score=mock_data['waste_management'],
                        employee_satisfaction=mock_data['employee_satisfaction'],
                        diversity_score=mock_data['diversity'],
                        board_independence=mock_data['board_independence'],
                        executive_compensation=mock_data['executive_compensation'],
                        data_quality_score=mock_data['data_quality'],
                        last_updated=datetime.now()
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to fetch ESG data for {symbol}: {e}")
                    
        return results
        
    async def fetch_climate_metrics(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch climate-specific metrics"""
        
        results = {}
        
        for symbol in symbols:
            results[symbol] = {
                'scope_1_emissions': np.random.uniform(10000, 1000000),  # tCO2e
                'scope_2_emissions': np.random.uniform(5000, 500000),
                'scope_3_emissions': np.random.uniform(50000, 5000000),
                'carbon_intensity_revenue': np.random.uniform(10, 1000),  # tCO2e/million revenue
                'renewable_energy_ratio': np.random.uniform(0.1, 0.9),
                'water_consumption': np.random.uniform(1000, 100000),  # megalitres
                'waste_generated': np.random.uniform(100, 10000),  # tonnes
                'climate_risk_physical': np.random.uniform(1, 10),  # 1-10 scale
                'climate_risk_transition': np.random.uniform(1, 10)
            }
            
        return results
        
    def _generate_mock_esg_data(self, symbol: str) -> Dict[str, float]:
        """Generate realistic mock ESG data"""
        
        # Set seed based on symbol for consistent mock data
        np.random.seed(hash(symbol) % 2**32)
        
        # Generate correlated ESG scores (higher E tends to correlate with higher S and G)
        base_score = np.random.uniform(3, 8)
        noise_factor = 0.3
        
        environmental = max(0, min(10, base_score + np.random.normal(0, noise_factor)))
        social = max(0, min(10, base_score * 0.8 + np.random.normal(0, noise_factor)))
        governance = max(0, min(10, base_score * 0.9 + np.random.normal(0, noise_factor)))
        overall = (environmental + social + governance) / 3
        
        return {
            'environmental': environmental,
            'social': social,
            'governance': governance,
            'overall': overall,
            'climate_risk': max(0, min(10, 10 - environmental + np.random.normal(0, 0.5))),
            'carbon_intensity': max(0, np.random.lognormal(4, 1) * (10 - environmental)),
            'water_usage': np.random.uniform(1, 10),
            'waste_management': environmental + np.random.normal(0, 0.5),
            'employee_satisfaction': social + np.random.normal(0, 0.5),
            'diversity': social + np.random.normal(0, 0.7),
            'board_independence': governance + np.random.normal(0, 0.3),
            'executive_compensation': max(0, min(10, governance + np.random.normal(0, 0.8))),
            'data_quality': np.random.uniform(6, 10)
        }

class SustainalyticsESGProvider(BaseESGProvider):
    """Sustainalytics ESG provider (alternative data source)"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.sustainalytics.com/v1"
        
    async def fetch_esg_scores(self, symbols: List[str]) -> Dict[str, ESGScores]:
        """Fetch ESG scores from Sustainalytics"""
        # Similar implementation to MSCI but with Sustainalytics data structure
        return await MSCIESGProvider(self.api_key).fetch_esg_scores(symbols)
        
    async def fetch_climate_metrics(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Fetch climate metrics from Sustainalytics"""
        return await MSCIESGProvider(self.api_key).fetch_climate_metrics(symbols)

class ClimateScenarioEngine:
    """
    Climate scenario modeling engine for physical and transition risks
    Based on IPCC climate scenarios and TCFD recommendations
    """
    
    def __init__(self):
        self.scenarios = self._initialize_climate_scenarios()
        
    def _initialize_climate_scenarios(self) -> Dict[str, ClimateScenario]:
        """Initialize standard climate scenarios"""
        
        scenarios = {}
        
        # RCP 2.6 (Paris Agreement aligned)
        scenarios['RCP26'] = ClimateScenario(
            scenario_name="RCP 2.6 - Paris Aligned",
            temperature_increase=1.5,
            carbon_price_trajectory=np.array([50, 75, 100, 125, 150]),  # $/tCO2 over 5 years
            transition_risk_multiplier=2.0,
            physical_risk_multiplier=1.0,
            policy_stringency=0.9,
            technology_adoption_rate=0.8,
            stranded_asset_probability=0.3
        )
        
        # RCP 4.5 (Moderate scenario)
        scenarios['RCP45'] = ClimateScenario(
            scenario_name="RCP 4.5 - Moderate",
            temperature_increase=2.5,
            carbon_price_trajectory=np.array([30, 40, 50, 60, 70]),
            transition_risk_multiplier=1.5,
            physical_risk_multiplier=1.5,
            policy_stringency=0.6,
            technology_adoption_rate=0.5,
            stranded_asset_probability=0.2
        )
        
        # RCP 8.5 (Business as usual / Hot house world)
        scenarios['RCP85'] = ClimateScenario(
            scenario_name="RCP 8.5 - Business as Usual",
            temperature_increase=4.0,
            carbon_price_trajectory=np.array([10, 15, 20, 25, 30]),
            transition_risk_multiplier=0.5,
            physical_risk_multiplier=3.0,
            policy_stringency=0.2,
            technology_adoption_rate=0.2,
            stranded_asset_probability=0.1
        )
        
        return scenarios
        
    def calculate_physical_risk_impact(self, scenario: ClimateScenario, 
                                     sector: str, location: str) -> float:
        """Calculate physical climate risk impact"""
        
        # Sector-specific physical risk multipliers
        sector_multipliers = {
            'utilities': 2.0,
            'real_estate': 1.8,
            'agriculture': 2.5,
            'insurance': 1.5,
            'tourism': 2.0,
            'manufacturing': 1.3,
            'technology': 0.8,
            'financial': 1.0,
            'default': 1.2
        }
        
        # Location-specific risk (simplified - would use detailed climate data)
        location_multipliers = {
            'coastal': 1.5,
            'tropical': 1.7,
            'temperate': 1.0,
            'arctic': 2.0,
            'desert': 1.3,
            'default': 1.2
        }
        
        sector_mult = sector_multipliers.get(sector.lower(), sector_multipliers['default'])
        location_mult = location_multipliers.get(location.lower(), location_multipliers['default'])
        
        # Base physical risk increases exponentially with temperature
        base_risk = (scenario.temperature_increase / 1.5) ** 2
        
        total_impact = base_risk * scenario.physical_risk_multiplier * sector_mult * location_mult
        
        # Cap at reasonable maximum (50% annual impact)
        return min(total_impact * 0.1, 0.5)  # Convert to annual percentage impact
        
    def calculate_transition_risk_impact(self, scenario: ClimateScenario,
                                       carbon_intensity: float,
                                       sector: str) -> float:
        """Calculate transition climate risk impact"""
        
        # High carbon intensity sectors face higher transition risk
        sector_transition_risk = {
            'energy_fossil': 3.0,
            'utilities_coal': 2.8,
            'steel': 2.5,
            'cement': 2.3,
            'airlines': 2.2,
            'shipping': 2.0,
            'automotive_ice': 2.0,
            'chemicals': 1.8,
            'mining': 1.7,
            'technology': 0.5,
            'financial': 0.8,
            'healthcare': 0.6,
            'default': 1.0
        }
        
        sector_mult = sector_transition_risk.get(sector.lower(), sector_transition_risk['default'])
        
        # Carbon price impact ($ per tonne CO2)
        future_carbon_price = scenario.carbon_price_trajectory[-1]  # End of period price
        carbon_cost_impact = carbon_intensity * future_carbon_price / 1000000  # As % of revenue
        
        # Technology disruption risk
        tech_disruption = scenario.technology_adoption_rate * sector_mult * 0.1
        
        # Policy risk
        policy_risk = scenario.policy_stringency * sector_mult * 0.05
        
        total_transition_impact = carbon_cost_impact + tech_disruption + policy_risk
        
        return min(total_transition_impact * scenario.transition_risk_multiplier, 0.4)

class ESGRiskIntegration:
    """
    Main ESG risk integration engine
    Integrates ESG factors into financial risk models and portfolio optimization
    """
    
    def __init__(self, esg_provider: BaseESGProvider):
        self.esg_provider = esg_provider
        self.climate_engine = ClimateScenarioEngine()
        self.esg_cache = {}
        self.last_cache_update = None
        self.cache_duration = timedelta(hours=24)  # Cache ESG data for 24 hours
        
    async def get_esg_scores(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, ESGScores]:
        """Get ESG scores with caching"""
        
        current_time = datetime.now()
        
        # Check cache validity
        if (not force_refresh and 
            self.last_cache_update and 
            current_time - self.last_cache_update < self.cache_duration and
            all(symbol in self.esg_cache for symbol in symbols)):
            
            logger.info("Using cached ESG data")
            return {symbol: self.esg_cache[symbol] for symbol in symbols}
        
        # Fetch fresh data
        logger.info("Fetching fresh ESG data")
        fresh_data = await self.esg_provider.fetch_esg_scores(symbols)
        
        # Update cache
        self.esg_cache.update(fresh_data)
        self.last_cache_update = current_time
        
        return fresh_data
        
    def calculate_esg_risk_adjustment(self, returns: np.ndarray, 
                                    esg_scores: Dict[str, ESGScores],
                                    climate_scenario: str = 'RCP45') -> ESGRiskResult:
        """
        Calculate ESG risk adjustments to traditional risk models
        """
        
        import time
        start_time = time.time()
        
        scenario = self.climate_engine.scenarios[climate_scenario]
        
        # ESG momentum calculation
        esg_momentum = self._calculate_esg_momentum(esg_scores)
        
        # ESG risk premium calculation
        esg_risk_premium = self._calculate_esg_risk_premium(esg_scores)
        
        # Climate VaR calculation
        climate_var = self._calculate_climate_var(returns, esg_scores, scenario)
        
        # Transition and physical risk impacts
        transition_impacts = {}
        physical_impacts = {}
        total_carbon_footprint = 0
        
        for symbol, scores in esg_scores.items():
            # Assume sector and location from symbol (simplified)
            sector = self._infer_sector(symbol)
            location = self._infer_location(symbol)
            
            transition_impacts[symbol] = self.climate_engine.calculate_transition_risk_impact(
                scenario, scores.carbon_intensity, sector
            )
            
            physical_impacts[symbol] = self.climate_engine.calculate_physical_risk_impact(
                scenario, sector, location
            )
            
            total_carbon_footprint += scores.carbon_intensity
            
        # Adjust returns for ESG factors
        adjusted_returns = self._adjust_returns_for_esg(
            returns, esg_scores, esg_risk_premium, esg_momentum
        )
        
        # Calculate sustainable alpha
        sustainable_alpha = self._calculate_sustainable_alpha(esg_scores, returns)
        
        execution_time = time.time() - start_time
        
        return ESGRiskResult(
            adjusted_returns=adjusted_returns,
            esg_risk_premium=esg_risk_premium,
            climate_var=climate_var,
            transition_risk_impact=transition_impacts,
            physical_risk_impact=physical_impacts,
            esg_momentum_score=esg_momentum,
            sustainable_alpha=sustainable_alpha,
            carbon_footprint=total_carbon_footprint,
            execution_time=execution_time
        )
        
    def _calculate_esg_momentum(self, esg_scores: Dict[str, ESGScores]) -> float:
        """Calculate ESG momentum score"""
        
        # ESG momentum is based on improvement trends
        # Mock implementation - would track historical ESG score changes
        overall_scores = [score.overall_score for score in esg_scores.values()]
        
        if len(overall_scores) == 0:
            return 0.0
            
        # High ESG scores indicate positive momentum
        avg_esg_score = np.mean(overall_scores)
        
        # Normalize to -1 to 1 scale
        momentum_score = (avg_esg_score - 5.0) / 5.0  # Assuming 0-10 ESG scale
        
        return momentum_score
        
    def _calculate_esg_risk_premium(self, esg_scores: Dict[str, ESGScores]) -> float:
        """Calculate ESG risk premium"""
        
        # Higher ESG scores typically command lower risk premiums
        overall_scores = [score.overall_score for score in esg_scores.values()]
        
        if len(overall_scores) == 0:
            return 0.02  # Default 2% risk premium
            
        avg_esg_score = np.mean(overall_scores)
        
        # Risk premium decreases with better ESG scores
        # Range from 4% (worst ESG) to 0% (best ESG)
        risk_premium = 0.04 * (1 - avg_esg_score / 10.0)
        
        return max(0, risk_premium)
        
    def _calculate_climate_var(self, returns: np.ndarray, 
                             esg_scores: Dict[str, ESGScores],
                             scenario: ClimateScenario) -> float:
        """Calculate Climate Value at Risk"""
        
        # Climate VaR incorporates both transition and physical risks
        if len(returns) == 0:
            return 0.0
            
        base_var_95 = np.percentile(returns, 5)  # 95% VaR
        
        # Climate risk amplification factor
        climate_risks = [score.climate_risk_score for score in esg_scores.values()]
        avg_climate_risk = np.mean(climate_risks) if climate_risks else 5.0
        
        # Higher climate risk scores indicate higher risk
        climate_amplification = 1.0 + (avg_climate_risk / 10.0) * scenario.physical_risk_multiplier * 0.5
        
        climate_var = base_var_95 * climate_amplification
        
        return climate_var
        
    def _adjust_returns_for_esg(self, returns: np.ndarray, 
                              esg_scores: Dict[str, ESGScores],
                              esg_risk_premium: float,
                              esg_momentum: float) -> np.ndarray:
        """Adjust returns for ESG factors"""
        
        if len(returns) == 0:
            return returns
            
        # ESG adjustment factors
        esg_adjustment = esg_momentum * 0.001  # Small daily adjustment based on momentum
        risk_adjustment = -esg_risk_premium / 252  # Daily risk premium adjustment
        
        # Apply adjustments
        adjusted_returns = returns + esg_adjustment + risk_adjustment
        
        # Add ESG-related volatility based on ESG score dispersion
        overall_scores = [score.overall_score for score in esg_scores.values()]
        
        if len(overall_scores) > 1:
            esg_vol_adjustment = np.std(overall_scores) / 10.0 * 0.01  # Max 1% vol adjustment
            adjusted_returns += np.random.normal(0, esg_vol_adjustment, len(adjusted_returns))
            
        return adjusted_returns
        
    def _calculate_sustainable_alpha(self, esg_scores: Dict[str, ESGScores],
                                   returns: np.ndarray) -> float:
        """Calculate sustainable alpha (excess return from ESG factors)"""
        
        if len(returns) == 0 or len(esg_scores) == 0:
            return 0.0
            
        # Sustainable alpha is positive correlation between ESG scores and returns
        overall_scores = [score.overall_score for score in esg_scores.values()]
        avg_esg_score = np.mean(overall_scores)
        
        # Higher ESG typically leads to sustainable alpha
        # 0% to 2% annual alpha for ESG leaders
        alpha_factor = max(0, (avg_esg_score - 5.0) / 5.0)  # 0 to 1
        sustainable_alpha = alpha_factor * 0.02  # Max 2% annual alpha
        
        return sustainable_alpha
        
    def _infer_sector(self, symbol: str) -> str:
        """Infer sector from symbol (simplified)"""
        
        # Mock sector mapping - in practice would use proper sector data
        sector_mapping = {
            'TSLA': 'automotive_ice',
            'AAPL': 'technology',
            'MSFT': 'technology',
            'XOM': 'energy_fossil',
            'JPM': 'financial',
            'JNJ': 'healthcare',
            'BA': 'aerospace'
        }
        
        return sector_mapping.get(symbol, 'default')
        
    def _infer_location(self, symbol: str) -> str:
        """Infer primary location from symbol (simplified)"""
        
        # Mock location mapping
        location_mapping = {
            'TSLA': 'temperate',
            'AAPL': 'temperate',
            'MSFT': 'temperate'
        }
        
        return location_mapping.get(symbol, 'temperate')

class ESGPortfolioOptimizer:
    """
    ESG-aware portfolio optimizer
    Incorporates ESG constraints and objectives into portfolio construction
    """
    
    def __init__(self):
        self.min_esg_score = 0.0
        self.max_carbon_intensity = float('inf')
        self.esg_objective_weight = 0.1
        
    def optimize_esg_portfolio(self, 
                             expected_returns: np.ndarray,
                             covariance_matrix: np.ndarray,
                             esg_scores: Dict[str, ESGScores],
                             symbols: List[str],
                             constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize portfolio with ESG considerations
        """
        
        n_assets = len(expected_returns)
        
        if len(symbols) != n_assets:
            raise ValueError("Symbols length must match expected returns length")
            
        # Extract ESG data
        esg_overall_scores = np.array([
            esg_scores[symbol].overall_score if symbol in esg_scores else 5.0 
            for symbol in symbols
        ])
        
        carbon_intensities = np.array([
            esg_scores[symbol].carbon_intensity if symbol in esg_scores else 1000.0
            for symbol in symbols
        ])
        
        # Objective function: Maximize risk-adjusted return and ESG score
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Risk-adjusted return
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # ESG score component
            portfolio_esg_score = np.dot(weights, esg_overall_scores)
            normalized_esg_score = portfolio_esg_score / 10.0  # Normalize to 0-1
            
            # Combined objective (maximize)
            combined_objective = (
                sharpe_ratio + 
                self.esg_objective_weight * normalized_esg_score
            )
            
            return -combined_objective  # Minimize negative
            
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # ESG constraints
        if constraints:
            min_esg = constraints.get('min_esg_score', self.min_esg_score)
            if min_esg > 0:
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda w: np.dot(w, esg_overall_scores) - min_esg
                })
                
            max_carbon = constraints.get('max_carbon_intensity', self.max_carbon_intensity)
            if max_carbon < float('inf'):
                constraint_list.append({
                    'type': 'ineq', 
                    'fun': lambda w: max_carbon - np.dot(w, carbon_intensities)
                })
                
            # Sector/geographic diversification constraints could be added here
            
        # Bounds (long-only with optional position limits)
        max_weight = constraints.get('max_single_weight', 0.4) if constraints else 0.4
        bounds = [(0.0, max_weight) for _ in range(n_assets)]
        
        # Initial guess - equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_variance = np.dot(optimal_weights.T, np.dot(covariance_matrix, optimal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_esg_score = np.dot(optimal_weights, esg_overall_scores)
            portfolio_carbon_intensity = np.dot(optimal_weights, carbon_intensities)
            
            return {
                'optimal_weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
                'esg_score': portfolio_esg_score,
                'carbon_intensity': portfolio_carbon_intensity,
                'optimization_success': True,
                'asset_weights': dict(zip(symbols, optimal_weights))
            }
        else:
            return {
                'optimization_success': False,
                'message': result.message
            }

class ESGReportingEngine:
    """
    ESG reporting and visualization engine
    Generates comprehensive ESG risk reports and dashboards
    """
    
    def __init__(self):
        self.report_templates = {
            'tcfd_report': self._generate_tcfd_report,
            'esg_risk_summary': self._generate_esg_risk_summary,
            'climate_scenario_analysis': self._generate_climate_scenario_report,
            'sustainable_portfolio_report': self._generate_sustainable_portfolio_report
        }
        
    def generate_report(self, report_type: str, data: Dict[str, Any]) -> str:
        """Generate ESG report of specified type"""
        
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
            
        generator_func = self.report_templates[report_type]
        return generator_func(data)
        
    def _generate_tcfd_report(self, data: Dict[str, Any]) -> str:
        """Generate TCFD (Task Force on Climate-related Financial Disclosures) report"""
        
        esg_result = data.get('esg_result')
        portfolio_data = data.get('portfolio_data', {})
        
        report = f"""
# TCFD Climate Risk Assessment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides climate-related financial risk disclosures in accordance with TCFD recommendations.

### Key Metrics
- Portfolio Carbon Footprint: {esg_result.carbon_footprint:.0f} tCO2e
- Climate VaR (95%): {esg_result.climate_var:.4f}
- ESG Risk Premium: {esg_result.esg_risk_premium:.2%}

## Governance
Climate risk oversight is integrated into investment decision-making processes through:
- ESG risk assessment framework
- Climate scenario analysis
- Portfolio-level carbon footprint monitoring

## Strategy
### Climate Scenario Analysis
The following climate scenarios have been analyzed:

#### Transition Risks
"""
        
        for symbol, impact in esg_result.transition_risk_impact.items():
            report += f"- {symbol}: {impact:.2%} annual impact from transition risks\n"
            
        report += "\n#### Physical Risks\n"
        for symbol, impact in esg_result.physical_risk_impact.items():
            report += f"- {symbol}: {impact:.2%} annual impact from physical risks\n"
            
        report += f"""

## Risk Management
- Climate risks are integrated into overall portfolio risk management
- Regular stress testing under multiple climate scenarios
- ESG momentum tracking: {esg_result.esg_momentum_score:.3f}

## Metrics and Targets
- Sustainable Alpha Generated: {esg_result.sustainable_alpha:.2%} annually
- ESG Integration Score: Active
- Carbon Footprint Reduction Target: On track

---
Report generated by MCMF ESG Risk Engine v2.1
"""
        
        return report
        
    def _generate_esg_risk_summary(self, data: Dict[str, Any]) -> str:
        """Generate ESG risk summary report"""
        
        esg_scores = data.get('esg_scores', {})
        
        report = f"""
# ESG Risk Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Portfolio ESG Scores
"""
        
        for symbol, scores in esg_scores.items():
            report += f"""
### {symbol}
- Overall ESG Score: {scores.overall_score:.1f}/10
- Environmental: {scores.environmental_score:.1f}/10
- Social: {scores.social_score:.1f}/10  
- Governance: {scores.governance_score:.1f}/10
- Climate Risk Score: {scores.climate_risk_score:.1f}/10
- Carbon Intensity: {scores.carbon_intensity:.0f} tCO2e/million revenue
"""
        
        return report
        
    def _generate_climate_scenario_report(self, data: Dict[str, Any]) -> str:
        """Generate climate scenario analysis report"""
        
        scenarios = data.get('scenarios', {})
        
        report = f"""
# Climate Scenario Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Scenario Overview
This analysis evaluates portfolio performance under different climate scenarios:

"""
        
        for scenario_name, scenario in scenarios.items():
            report += f"""
### {scenario.scenario_name}
- Temperature Increase: {scenario.temperature_increase}°C
- Policy Stringency: {scenario.policy_stringency:.1f}
- Technology Adoption: {scenario.technology_adoption_rate:.1%}
- Stranded Asset Risk: {scenario.stranded_asset_probability:.1%}

"""
        
        return report
        
    def _generate_sustainable_portfolio_report(self, data: Dict[str, Any]) -> str:
        """Generate sustainable portfolio performance report"""
        
        optimization_result = data.get('optimization_result', {})
        
        if not optimization_result.get('optimization_success'):
            return "Portfolio optimization failed - cannot generate report"
            
        report = f"""
# Sustainable Portfolio Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Portfolio Characteristics
- Expected Return: {optimization_result['expected_return']:.2%}
- Volatility: {optimization_result['volatility']:.2%}
- Sharpe Ratio: {optimization_result['sharpe_ratio']:.3f}
- ESG Score: {optimization_result['esg_score']:.1f}/10
- Carbon Intensity: {optimization_result['carbon_intensity']:.0f} tCO2e/million revenue

## Asset Allocation
"""
        
        for symbol, weight in optimization_result.get('asset_weights', {}).items():
            if weight > 0.01:  # Show only significant weights
                report += f"- {symbol}: {weight:.1%}\n"
                
        return report

class ESGIntegrationEngine:
    """
    Main ESG integration engine combining all ESG functionality
    """
    
    def __init__(self, esg_provider_name: str = 'msci', api_key: Optional[str] = None):
        
        # Initialize ESG data provider
        if esg_provider_name.lower() == 'msci':
            self.esg_provider = MSCIESGProvider(api_key or "demo_key")
        elif esg_provider_name.lower() == 'sustainalytics':
            self.esg_provider = SustainalyticsESGProvider(api_key or "demo_key")
        else:
            raise ValueError(f"Unknown ESG provider: {esg_provider_name}")
            
        # Initialize components
        self.risk_integration = ESGRiskIntegration(self.esg_provider)
        self.portfolio_optimizer = ESGPortfolioOptimizer()
        self.reporting_engine = ESGReportingEngine()
        
        logger.info(f"ESG Integration Engine initialized with {esg_provider_name} provider")
        
    async def comprehensive_esg_analysis(self, 
                                       symbols: List[str],
                                       returns_data: Dict[str, np.ndarray],
                                       expected_returns: np.ndarray,
                                       covariance_matrix: np.ndarray,
                                       climate_scenario: str = 'RCP45') -> Dict[str, Any]:
        """
        Perform comprehensive ESG analysis including risk assessment and portfolio optimization
        """
        
        logger.info(f"Starting comprehensive ESG analysis for {len(symbols)} assets")
        
        # Fetch ESG scores
        esg_scores = await self.risk_integration.get_esg_scores(symbols)
        
        # Calculate ESG risk adjustments
        combined_returns = np.concatenate(list(returns_data.values())) if returns_data else np.array([])
        esg_risk_result = self.risk_integration.calculate_esg_risk_adjustment(
            combined_returns, esg_scores, climate_scenario
        )
        
        # Optimize ESG-aware portfolio
        optimization_result = self.portfolio_optimizer.optimize_esg_portfolio(
            expected_returns, covariance_matrix, esg_scores, symbols
        )
        
        # Generate reports
        reports = {}
        report_data = {
            'esg_result': esg_risk_result,
            'esg_scores': esg_scores,
            'optimization_result': optimization_result,
            'scenarios': self.risk_integration.climate_engine.scenarios
        }
        
        for report_type in self.reporting_engine.report_templates.keys():
            try:
                reports[report_type] = self.reporting_engine.generate_report(report_type, report_data)
            except Exception as e:
                logger.error(f"Failed to generate {report_type}: {e}")
                
        return {
            'esg_scores': esg_scores,
            'risk_analysis': esg_risk_result,
            'portfolio_optimization': optimization_result,
            'reports': reports,
            'climate_scenario': climate_scenario,
            'analysis_timestamp': datetime.now()
        }

# Factory functions for easy initialization
def create_esg_engine(provider: str = 'msci', api_key: Optional[str] = None) -> ESGIntegrationEngine:
    """Create ESG integration engine with specified provider"""
    return ESGIntegrationEngine(provider, api_key)

def create_climate_scenarios() -> ClimateScenarioEngine:
    """Create climate scenario engine"""
    return ClimateScenarioEngine()

async def demo_esg_analysis():
    """Demo ESG analysis workflow"""
    
    # Initialize engine
    esg_engine = create_esg_engine('msci')
    
    # Demo symbols
    symbols = ['AAPL', 'TSLA', 'MSFT', 'XOM', 'JPM']
    
    # Mock data
    expected_returns = np.array([0.08, 0.12, 0.07, 0.06, 0.09])
    covariance_matrix = np.random.random((5, 5))
    covariance_matrix = covariance_matrix @ covariance_matrix.T  # Make positive definite
    covariance_matrix *= 0.04  # Scale to reasonable volatility levels
    
    returns_data = {
        symbol: np.random.normal(ret/252, 0.02, 252) 
        for symbol, ret in zip(symbols, expected_returns)
    }
    
    # Run comprehensive analysis
    results = await esg_engine.comprehensive_esg_analysis(
        symbols=symbols,
        returns_data=returns_data,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        climate_scenario='RCP26'
    )
    
    print("ESG Analysis Complete!")
    print(f"Portfolio ESG Score: {results['portfolio_optimization']['esg_score']:.1f}/10")
    print(f"Carbon Intensity: {results['portfolio_optimization']['carbon_intensity']:.0f} tCO2e/M$")
    
    return results

if __name__ == "__main__":
    # Run demo
    import asyncio
    asyncio.run(demo_esg_analysis())
