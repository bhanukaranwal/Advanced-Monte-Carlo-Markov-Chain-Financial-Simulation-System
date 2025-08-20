text
# Getting Started with Monte Carlo-Markov Finance System

This guide will help you get up and running with the Monte Carlo-Markov Finance System (MCMF) quickly.

## Prerequisites

- Python 3.8+ (Python 3.10+ recommended)
- 8GB+ RAM (16GB+ recommended for large simulations)
- Git
- Optional: NVIDIA GPU with CUDA 11.8+ for acceleration

## Installation

### Standard Installation

Clone the repository
git clone https://github.com/your-org/monte-carlo-markov-finance.git
cd monte-carlo-markov-finance

Create virtual environment
python -m venv mcmf-env
source mcmf-env/bin/activate # On Windows: mcmf-env\Scripts\activate

Install dependencies
pip install -r requirements.txt

Install the package
pip install -e .

text

### GPU-Accelerated Installation

After standard installation, add GPU support
pip install -r requirements-gpu.txt

Verify GPU installation
python -c "import cupy; print('GPU acceleration available')"

text

## Your First Simulation

### Basic Monte Carlo Simulation

from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
import numpy as np

Create simulation engine
engine = GeometricBrownianMotionEngine(
n_simulations=1000,
n_steps=252,
initial_price=100.0,
drift=0.05,
volatility=0.2
)

Generate paths
paths = engine.simulate_paths()
print(f"Generated {paths.shape} paths")

Analyze results
final_prices = paths[:, -1]
print(f"Mean final price: ${np.mean(final_prices):.2f}")
print(f"Price volatility: {np.std(final_prices):.2f}")

text

### Risk Analysis

from analytics_engine.risk_analytics import RiskAnalytics

Calculate returns
returns = np.diff(np.log(paths), axis=1).mean(axis=0)

Risk analysis
risk_analytics = RiskAnalytics()
risk_measures = risk_analytics.calculate_comprehensive_risk_measures(returns)

print(f"95% VaR: {risk_measures.var_95:.4f}")
print(f"Maximum Drawdown: {risk_measures.maximum_drawdown:.2%}")

text

### Interactive Dashboard

Launch Streamlit dashboard
streamlit run src/visualization/dashboard.py

text

Visit http://localhost:8501 to see your dashboard.

## Next Steps

- [User Guide](user-guide.md) - Comprehensive features overview
- [Examples](../examples/) - Detailed code examples
- [API Reference](api/index.html) - Complete API documentation

## Troubleshooting

### Common Issues

**ImportError: No module named 'cupy'**
pip install cupy-cuda11x # For CUDA 11.x

text

**Memory Error during large simulations**
Use chunked processing
from optimization.memory_optimization import ChunkedProcessor

text

**Slow performance**
Enable GPU acceleration
export USE_GPU=true
export GPU_BACKEND=cupy

text

## Support

- Documentation: [docs.mcmf-system.com](https://docs.mcmf-system.com)
- Issues: [GitHub Issues](https://github.com/your-org/monte-carlo-markov-finance/issues)
- Community: [Discord](https://discord.gg/mcmf)
