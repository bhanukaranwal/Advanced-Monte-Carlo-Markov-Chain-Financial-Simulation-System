"""
CLI commands for Monte Carlo simulations
"""

import click
import numpy as np
import json
from datetime import datetime

from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from monte_carlo_engine.multi_asset import MultiAssetEngine

@click.group()
def simulation_cli():
    """Monte Carlo simulation commands"""
    pass

@simulation_cli.command()
@click.option("--n-simulations", "-n", default=10000, help="Number of simulations")
@click.option("--n-steps", "-s", default=252, help="Number of time steps")
@click.option("--initial-price", "-p", default=100.0, help="Initial price")
@click.option("--drift", "-d", default=0.05, help="Drift parameter")
@click.option("--volatility", "-v", default=0.2, help="Volatility parameter")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--format", "output_format", default="json", type=click.Choice(["json", "csv"]))
def gbm(n_simulations, n_steps, initial_price, drift, volatility, seed, output, output_format):
    """Run Geometric Brownian Motion simulation"""
    
    click.echo(f"Running GBM simulation with {n_simulations} paths...")
    
    engine = GeometricBrownianMotionEngine(
        n_simulations=n_simulations,
        n_steps=n_steps,
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        random_seed=seed
    )
    
    paths = engine.simulate_paths()
    
    # Calculate statistics
    final_prices = paths[:, -1]
    results = {
        "parameters": {
            "n_simulations": n_simulations,
            "n_steps": n_steps,
            "initial_price": initial_price,
            "drift": drift,
            "volatility": volatility,
            "seed": seed
        },
        "statistics": {
            "mean_final_price": float(np.mean(final_prices)),
            "std_final_price": float(np.std(final_prices)),
            "min_final_price": float(np.min(final_prices)),
            "max_final_price": float(np.max(final_prices))
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if output:
        if output_format == "json":
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
        elif output_format == "csv":
            import pandas as pd
            df = pd.DataFrame(paths)
            df.to_csv(output, index=False)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(json.dumps(results, indent=2))

@simulation_cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="Multi-asset configuration file")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def multi_asset(config, output):
    """Run multi-asset simulation from configuration file"""
    
    with open(config, 'r') as f:
        config_data = json.load(f)
    
    click.echo("Running multi-asset simulation...")
    
    engine = MultiAssetEngine(**config_data["parameters"])
    paths = engine.simulate_correlated_paths()
    
    results = {
        "config": config_data,
        "shape": list(paths.shape),
        "timestamp": datetime.now().isoformat()
    }
    
    if output:
        # Save paths as HDF5 for large data
        import h5py
        with h5py.File(output, 'w') as f:
            f.create_dataset('paths', data=paths)
            f.attrs['config'] = json.dumps(results)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(json.dumps(results, indent=2))
