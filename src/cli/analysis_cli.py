"""
CLI commands for data analysis and risk calculations
"""

import click
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

from analytics_engine.risk_analytics import RiskAnalytics
from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine

@click.group()
def analysis_cli():
    """Analysis and risk calculation commands"""
    pass

@analysis_cli.command()
@click.option("--input-file", "-i", required=True, type=click.Path(exists=True), 
              help="Input CSV file with returns data")
@click.option("--confidence-levels", "-c", default="0.95,0.99", 
              help="Confidence levels (comma-separated)")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "output_format", default="json", type=click.Choice(["json", "csv"]))
def calculate_var(input_file, confidence_levels, output, output_format):
    """Calculate Value at Risk from returns data"""
    
    try:
        # Load data
        df = pd.read_csv(input_file)
        if 'returns' not in df.columns:
            click.echo("Error: CSV file must contain a 'returns' column", err=True)
            return
            
        returns = df['returns'].values
        confidence_levels = [float(x.strip()) for x in confidence_levels.split(',')]
        
        # Calculate VaR
        risk_analytics = RiskAnalytics(confidence_levels=confidence_levels)
        
        results = {}
        for cl in confidence_levels:
            var_hist = risk_analytics.calculate_var(returns, cl, method='historical')
            var_param = risk_analytics.calculate_var(returns, cl, method='parametric')
            
            results[f'var_{int(cl*100)}_historical'] = var_hist
            results[f'var_{int(cl*100)}_parametric'] = var_param
            
        # Add metadata
        results['calculation_date'] = datetime.now().isoformat()
        results['data_points'] = len(returns)
        results['mean_return'] = float(np.mean(returns))
        results['volatility'] = float(np.std(returns))
        
        # Output results
        if output:
            if output_format == 'json':
                with open(output, 'w') as f:
                    json.dump(results, f, indent=2)
            elif output_format == 'csv':
                results_df = pd.DataFrame([results])
                results_df.to_csv(output, index=False)
            click.echo(f"Results saved to {output}")
        else:
            click.echo(json.dumps(results, indent=2))
            
    except Exception as e:
        click.echo(f"Error calculating VaR: {e}", err=True)

@analysis_cli.command()
@click.option("--input-file", "-i", required=True, type=click.Path(exists=True))
@click.option("--window", "-w", default=252, help="Rolling window size")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def rolling_metrics(input_file, window, output):
    """Calculate rolling risk metrics"""
    
    try:
        df = pd.read_csv(input_file)
        if 'returns' not in df.columns:
            click.echo("Error: CSV file must contain a 'returns' column", err=True)
            return
            
        returns = df['returns'].values
        
        if len(returns) < window:
            click.echo(f"Error: Not enough data points. Need at least {window}", err=True)
            return
            
        # Calculate rolling metrics
        rolling_results = []
        risk_analytics = RiskAnalytics()
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            
            # Calculate metrics for this window
            var_95 = risk_analytics.calculate_var(window_returns, 0.95)
            var_99 = risk_analytics.calculate_var(window_returns, 0.99)
            volatility = np.std(window_returns)
            
            rolling_results.append({
                'date_index': i,
                'var_95': var_95,
                'var_99': var_99,
                'volatility': volatility,
                'mean_return': np.mean(window_returns)
            })
            
        # Create output DataFrame
        results_df = pd.DataFrame(rolling_results)
        
        if output:
            results_df.to_csv(output, index=False)
            click.echo(f"Rolling metrics saved to {output}")
        else:
            click.echo(results_df.to_string(index=False))
            
    except Exception as e:
        click.echo(f"Error calculating rolling metrics: {e}", err=True)
