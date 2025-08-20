"""
Main CLI entry point
"""

import click
import logging
from pathlib import Path

from .simulation_cli import simulation_cli
from .analysis_cli import analysis_cli  
from .backtest_cli import backtest_cli
from .data_cli import data_cli
from .report_cli import report_cli

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="1.0.0")
@click.option("--config", "-c", type=click.Path(exists=True), help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", type=click.Path(), help="Log file path")
def main(config, verbose, log_file):
    """Monte Carlo-Markov Finance System CLI"""
    
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(level=level, format=format_str, filename=log_file)
    else:
        logging.basicConfig(level=level, format=format_str)
    
    if config:
        click.echo(f"Using configuration file: {config}")
        # Load configuration logic here
    
    logger.info("MCMF CLI initialized")

# Add command groups
main.add_command(simulation_cli)
main.add_command(analysis_cli)
main.add_command(backtest_cli)
main.add_command(data_cli)
main.add_command(report_cli)

if __name__ == "__main__":
    main()
