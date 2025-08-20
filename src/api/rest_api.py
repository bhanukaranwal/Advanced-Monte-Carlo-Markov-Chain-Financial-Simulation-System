"""
REST API endpoints for Monte Carlo-Markov Finance System
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields, validate, ValidationError
import logging

# MCMF imports
from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from monte_carlo_engine.multi_asset import MultiAssetEngine
from monte_carlo_engine.path_dependent import PathDependentEngine
from markov_models.hidden_markov import HiddenMarkovModel
from markov_models.regime_switching import RegimeSwitchingModel
from analytics_engine.risk_analytics import RiskAnalytics, PortfolioRiskAnalyzer
from analytics_engine.copula_models import CopulaModels
from validation.backtesting import BacktestEngine
from real_time_engine.stream_processor import StreamProcessor
from visualization.report_generator import PDFReportGenerator, HTMLReportGenerator
from .auth import token_required, get_user_from_token
from .rate_limiting import limiter

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mcmf-secret-key-change-in-production'
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Enable CORS
CORS(app, origins=['http://localhost:3000', 'http://localhost:8501'])

# Initialize rate limiter
limiter.init_app(app)

# Request/Response Schemas
class MonteCarloRequestSchema(Schema):
    n_simulations = fields.Integer(required=True, validate=validate.Range(min=100, max=1000000))
    n_steps = fields.Integer(required=True, validate=validate.Range(min=10, max=10000))
    initial_price = fields.Float(required=True, validate=validate.Range(min=0.01))
    drift = fields.Float(required=True, validate=validate.Range(min=-1.0, max=1.0))
    volatility = fields.Float(required=True, validate=validate.Range(min=0.001, max=2.0))
    random_seed = fields.Integer(missing=None)
    use_gpu = fields.Boolean(missing=False)
    antithetic_variates = fields.Boolean(missing=False)

class MultiAssetRequestSchema(Schema):
    n_simulations = fields.Integer(required=True, validate=validate.Range(min=100, max=500000))
    n_steps = fields.Integer(required=True, validate=validate.Range(min=10, max=5000))
    initial_prices = fields.List(fields.Float(validate=validate.Range(min=0.01)), required=True)
    drifts = fields.List(fields.Float(validate=validate.Range(min=-1.0, max=1.0)), required=True)
    volatilities = fields.List(fields.Float(validate=validate.Range(min=0.001, max=2.0)), required=True)
    correlation_matrix = fields.List(fields.List(fields.Float()), required=True)
    random_seed = fields.Integer(missing=None)

class OptionPricingRequestSchema(Schema):
    strike = fields.Float(required=True, validate=validate.Range(min=0.01))
    option_type = fields.String(required=True, validate=validate.OneOf(['call', 'put']))
    risk_free_rate = fields.Float(required=True, validate=validate.Range(min=-0.1, max=0.3))
    time_to_maturity = fields.Float(required=True, validate=validate.Range(min=0.001, max=10.0))
    barrier = fields.Float(missing=None, validate=validate.Range(min=0.01))
    barrier_type = fields.String(missing=None, validate=validate.OneOf(['up-and-out', 'up-and-in', 'down-and-out', 'down-and-in']))

class RiskAnalysisRequestSchema(Schema):
    returns_data = fields.List(fields.Float(), required=True)
    confidence_levels = fields.List(fields.Float(validate=validate.Range(min=0.8, max=0.999)), missing=[0.95, 0.99])
    method = fields.String(missing='historical', validate=validate.OneOf(['historical', 'parametric', 'monte_carlo']))
    risk_free_rate = fields.Float(missing=0.02)

class BacktestRequestSchema(Schema):
    strategy_code = fields.String(required=True)
    start_date = fields.Date(required=True)
    end_date = fields.Date(required=True)
    initial_capital = fields.Float(required=True, validate=validate.Range(min=1000))
    commission_rate = fields.Float(missing=0.001, validate=validate.Range(min=0, max=0.1))
    slippage_rate = fields.Float(missing=0.0005, validate=validate.Range(min=0, max=0.01))

# Error handlers
@app.errorhandler(ValidationError)
def handle_validation_error(e):
    return jsonify({'error': 'Validation error', 'messages': e.messages}), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def handle_internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = datetime.utcnow()

@app.after_request
def after_request(response):
    duration = (datetime.utcnow() - g.start_time).total_seconds()
    logger.info(f"Request {g.request_id} completed in {duration:.3f}s")
    response.headers['X-Request-ID'] = g.request_id
    return response

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

# API Documentation endpoint
@app.route('/api/v1', methods=['GET'])
def api_documentation():
    """API documentation endpoint"""
    endpoints = {
        'monte_carlo': {
            'url': '/api/v1/simulations/monte-carlo',
            'method': 'POST',
            'description': 'Run Monte Carlo simulation'
        },
        'multi_asset': {
            'url': '/api/v1/simulations/multi-asset',
            'method': 'POST',
            'description': 'Run multi-asset Monte Carlo simulation'
        },
        'option_pricing': {
            'url': '/api/v1/pricing/options',
            'method': 'POST',
            'description': 'Price path-dependent options'
        },
        'risk_analysis': {
            'url': '/api/v1/analytics/risk',
            'method': 'POST',
            'description': 'Calculate risk metrics'
        },
        'backtesting': {
            'url': '/api/v1/backtesting/run',
            'method': 'POST',
            'description': 'Run strategy backtesting'
        }
    }
    return jsonify({
        'api_version': 'v1',
        'endpoints': endpoints,
        'documentation': 'https://docs.mcmf-system.com'
    })

# Monte Carlo Simulation Endpoints
@app.route('/api/v1/simulations/monte-carlo', methods=['POST'])
@limiter.limit("10 per minute")
@token_required
def monte_carlo_simulation():
    """Run single-asset Monte Carlo simulation"""
    try:
        schema = MonteCarloRequestSchema()
        params = schema.load(request.json)
        
        logger.info(f"Starting Monte Carlo simulation with {params['n_simulations']} paths")
        
        # Create engine
        engine = GeometricBrownianMotionEngine(
            n_simulations=params['n_simulations'],
            n_steps=params['n_steps'],
            initial_price=params['initial_price'],
            drift=params['drift'],
            volatility=params['volatility'],
            random_seed=params.get('random_seed'),
            use_gpu=params.get('use_gpu', False),
            antithetic_variates=params.get('antithetic_variates', False)
        )
        
        # Run simulation
        paths = engine.simulate_paths()
        
        # Calculate statistics
        final_prices = paths[:, -1]
        results = {
            'simulation_id': str(uuid.uuid4()),
            'parameters': params,
            'statistics': {
                'mean_final_price': float(np.mean(final_prices)),
                'std_final_price': float(np.std(final_prices)),
                'min_final_price': float(np.min(final_prices)),
                'max_final_price': float(np.max(final_prices)),
                'percentiles': {
                    '5th': float(np.percentile(final_prices, 5)),
                    '25th': float(np.percentile(final_prices, 25)),
                    '50th': float(np.percentile(final_prices, 50)),
                    '75th': float(np.percentile(final_prices, 75)),
                    '95th': float(np.percentile(final_prices, 95))
                }
            },
            'paths_sample': paths[:min(10, len(paths))].tolist(),  # Return first 10 paths
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/simulations/multi-asset', methods=['POST'])
@limiter.limit("5 per minute")
@token_required
def multi_asset_simulation():
    """Run multi-asset Monte Carlo simulation"""
    try:
        schema = MultiAssetRequestSchema()
        params = schema.load(request.json)
        
        logger.info(f"Starting multi-asset simulation with {len(params['initial_prices'])} assets")
        
        # Create engine
        engine = MultiAssetEngine(
            n_simulations=params['n_simulations'],
            n_steps=params['n_steps'],
            initial_prices=params['initial_prices'],
            drifts=params['drifts'],
            volatilities=params['volatilities'],
            correlation_matrix=np.array(params['correlation_matrix']),
            random_seed=params.get('random_seed')
        )
        
        # Run simulation
        paths = engine.simulate_correlated_paths()
        
        # Calculate statistics for each asset
        asset_statistics = []
        for i in range(len(params['initial_prices'])):
            final_prices = paths[:, -1, i]
            asset_statistics.append({
                'asset_index': i,
                'mean_final_price': float(np.mean(final_prices)),
                'std_final_price': float(np.std(final_prices)),
                'var_95': float(np.percentile(final_prices, 5))
            })
        
        results = {
            'simulation_id': str(uuid.uuid4()),
            'parameters': params,
            'asset_statistics': asset_statistics,
            'correlation_matrix': params['correlation_matrix'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Multi-asset simulation error: {e}")
        return jsonify({'error': str(e)}), 500

# Option Pricing Endpoints
@app.route('/api/v1/pricing/options', methods=['POST'])
@limiter.limit("20 per minute")
@token_required
def price_options():
    """Price path-dependent options"""
    try:
        # Get Monte Carlo parameters and option parameters
        request_data = request.json
        mc_params = request_data.get('monte_carlo_params', {})
        option_schema = OptionPricingRequestSchema()
        option_params = option_schema.load(request_data.get('option_params', {}))
        
        # Default Monte Carlo parameters
        n_simulations = mc_params.get('n_simulations', 50000)
        n_steps = mc_params.get('n_steps', 252)
        initial_price = mc_params.get('initial_price', 100.0)
        drift = mc_params.get('drift', 0.05)
        volatility = mc_params.get('volatility', 0.2)
        
        logger.info(f"Pricing {option_params['option_type']} option with strike {option_params['strike']}")
        
        # Create path-dependent engine
        engine = PathDependentEngine(
            n_simulations=n_simulations,
            n_steps=n_steps,
            initial_price=initial_price,
            drift=drift,
            volatility=volatility
        )
        
        # Price option based on type
        if option_params.get('barrier') is not None:
            # Barrier option
            result = engine.price_barrier_option(
                strike=option_params['strike'],
                barrier=option_params['barrier'],
                option_type=option_params['option_type'],
                barrier_type=option_params.get('barrier_type', 'up-and-out'),
                risk_free_rate=option_params['risk_free_rate'],
                time_to_maturity=option_params['time_to_maturity']
            )
        else:
            # European option
            result = engine.price_european_option(
                strike=option_params['strike'],
                option_type=option_params['option_type'],
                risk_free_rate=option_params['risk_free_rate'],
                time_to_maturity=option_params['time_to_maturity']
            )
        
        # Add metadata
        result['pricing_id'] = str(uuid.uuid4())
        result['parameters'] = {
            'monte_carlo': mc_params,
            'option': option_params
        }
        result['timestamp'] = datetime.utcnow().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Option pricing error: {e}")
        return jsonify({'error': str(e)}), 500

# Risk Analytics Endpoints
@app.route('/api/v1/analytics/risk', methods=['POST'])
@limiter.limit("30 per minute")
@token_required
def risk_analysis():
    """Calculate comprehensive risk metrics"""
    try:
        schema = RiskAnalysisRequestSchema()
        params = schema.load(request.json)
        
        logger.info(f"Calculating risk metrics for {len(params['returns_data'])} data points")
        
        # Initialize risk analytics
        risk_analytics = RiskAnalytics(
            confidence_levels=params['confidence_levels']
        )
        
        returns_data = np.array(params['returns_data'])
        
        # Calculate comprehensive risk measures
        risk_measures = risk_analytics.calculate_comprehensive_risk_measures(
            returns_data, risk_free_rate=params['risk_free_rate']
        )
        
        # Convert to JSON-serializable format
        results = {
            'analysis_id': str(uuid.uuid4()),
            'parameters': params,
            'risk_measures': {
                'var_95': float(risk_measures.var_95),
                'var_99': float(risk_measures.var_99),
                'expected_shortfall_95': float(risk_measures.expected_shortfall_95),
                'expected_shortfall_99': float(risk_measures.expected_shortfall_99),
                'maximum_drawdown': float(risk_measures.maximum_drawdown),
                'sharpe_ratio': float(risk_measures.sortino_ratio),  # Using sortino as example
                'calmar_ratio': float(risk_measures.calmar_ratio),
                'omega_ratio': float(risk_measures.omega_ratio),
                'tail_ratio': float(risk_measures.tail_ratio)
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Risk analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/analytics/portfolio-risk', methods=['POST'])
@limiter.limit("20 per minute")
@token_required
def portfolio_risk_analysis():
    """Calculate portfolio-level risk metrics"""
    try:
        request_data = request.json
        weights = np.array(request_data['weights'])
        returns_data = pd.DataFrame(request_data['returns_data'])
        
        logger.info(f"Calculating portfolio risk for {len(weights)} assets")
        
        # Initialize portfolio risk analyzer
        risk_analytics = RiskAnalytics()
        portfolio_analyzer = PortfolioRiskAnalyzer(risk_analytics)
        
        # Calculate portfolio risk
        portfolio_risk = portfolio_analyzer.calculate_portfolio_var(weights, returns_data)
        
        results = {
            'analysis_id': str(uuid.uuid4()),
            'portfolio_var': float(portfolio_risk.portfolio_var),
            'component_var': {k: float(v) for k, v in portfolio_risk.component_var.items()},
            'marginal_var': {k: float(v) for k, v in portfolio_risk.marginal_var.items()},
            'diversification_ratio': float(portfolio_risk.diversification_ratio),
            'concentration_measure': float(portfolio_risk.concentration_measure),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Portfolio risk analysis error: {e}")
        return jsonify({'error': str(e)}), 500

# Markov Models Endpoints
@app.route('/api/v1/models/hmm/fit', methods=['POST'])
@limiter.limit("10 per minute")
@token_required
def fit_hmm():
    """Fit Hidden Markov Model"""
    try:
        request_data = request.json
        observations = np.array(request_data['observations'])
        n_states = request_data.get('n_states', 2)
        max_iterations = request_data.get('max_iterations', 100)
        
        logger.info(f"Fitting HMM with {n_states} states on {len(observations)} observations")
        
        # Create and fit HMM
        hmm = HiddenMarkovModel(n_states=n_states, n_observations=2)
        hmm.fit(observations, max_iterations=max_iterations)
        
        # Get results
        state_sequence = hmm.viterbi_decode(observations)
        state_probabilities = hmm.predict_states(observations)
        
        results = {
            'model_id': str(uuid.uuid4()),
            'n_states': n_states,
            'transition_matrix': hmm.transition_matrix.tolist(),
            'emission_matrix': hmm.emission_matrix.tolist(),
            'initial_probabilities': hmm.initial_probabilities.tolist(),
            'state_sequence': state_sequence.tolist(),
            'state_probabilities': state_probabilities.tolist(),
            'log_likelihood': float(hmm.log_likelihood) if hasattr(hmm, 'log_likelihood') else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"HMM fitting error: {e}")
        return jsonify({'error': str(e)}), 500

# Backtesting Endpoints
@app.route('/api/v1/backtesting/run', methods=['POST'])
@limiter.limit("5 per minute")
@token_required
def run_backtest():
    """Run strategy backtesting"""
    try:
        schema = BacktestRequestSchema()
        params = schema.load(request.json)
        
        logger.info(f"Running backtest from {params['start_date']} to {params['end_date']}")
        
        # This is a simplified version - in production, you'd need to:
        # 1. Validate and sandbox the strategy code
        # 2. Load actual market data
        # 3. Execute the full backtesting pipeline
        
        # For now, return a mock result
        results = {
            'backtest_id': str(uuid.uuid4()),
            'parameters': params,
            'results': {
                'total_return': 0.15,
                'annualized_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 0.67,
                'max_drawdown': -0.08,
                'num_trades': 45,
                'win_rate': 0.62
            },
            'status': 'completed',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Backtesting error: {e}")
        return jsonify({'error': str(e)}), 500

# Report Generation Endpoints
@app.route('/api/v1/reports/generate', methods=['POST'])
@limiter.limit("5 per minute")
@token_required
def generate_report():
    """Generate PDF or HTML report"""
    try:
        request_data = request.json
        report_type = request_data.get('type', 'pdf')
        report_data = request_data.get('data', {})
        
        logger.info(f"Generating {report_type} report")
        
        if report_type == 'pdf':
            generator = PDFReportGenerator("API Generated Report")
        else:
            generator = HTMLReportGenerator("API Generated Report")
        
        # Add executive summary if provided
        if 'summary' in report_data:
            generator.add_executive_summary(report_data['summary'])
        
        # Add sections
        for section in report_data.get('sections', []):
            generator.add_section(
                section['title'], 
                section['content'], 
                section.get('type', 'text')
            )
        
        # Generate report
        filename = f"report_{uuid.uuid4().hex[:8]}.{report_type}"
        generator.generate_report(filename)
        
        results = {
            'report_id': str(uuid.uuid4()),
            'filename': filename,
            'type': report_type,
            'status': 'generated',
            'download_url': f'/api/v1/reports/download/{filename}',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'error': str(e)}), 500

# Async task status endpoint
@app.route('/api/v1/tasks/<task_id>/status', methods=['GET'])
@token_required
def get_task_status(task_id):
    """Get status of long-running task"""
    # This would integrate with a task queue like Celery
    # For now, return a mock response
    return jsonify({
        'task_id': task_id,
        'status': 'completed',
        'progress': 100,
        'result_url': f'/api/v1/tasks/{task_id}/result',
        'timestamp': datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
