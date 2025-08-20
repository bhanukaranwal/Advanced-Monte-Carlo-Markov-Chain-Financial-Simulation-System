"""
Setup configuration for Monte Carlo-Markov Finance System
"""
from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Optional GPU dependencies
gpu_requirements = [
    "cupy-cuda11x>=11.0.0; sys_platform != 'darwin'",
    "pyopencl>=2022.1"
]

# Development dependencies
dev_requirements = [
    "pytest>=7.1.0",
    "pytest-cov>=3.0.0", 
    "pytest-benchmark>=3.4.0",
    "black>=22.6.0",
    "isort>=5.10.0",
    "flake8>=5.0.0",
    "mypy>=0.971",
    "pre-commit>=2.20.0"
]

setup(
    name="monte-carlo-markov-finance",
    version="1.0.0",
    author="MCMF Development Team",
    author_email="dev@mcmf-system.com",
    description="Comprehensive financial modeling framework with Monte Carlo and Markov models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/monte-carlo-markov-finance",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/monte-carlo-markov-finance/issues",
        "Documentation": "https://docs.mcmf-system.com",
        "Source Code": "https://github.com/your-org/monte-carlo-markov-finance",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": gpu_requirements,
        "dev": dev_requirements,
        "all": gpu_requirements + dev_requirements
    },
    entry_points={
        "console_scripts": [
            "mcmf-dashboard=visualization.dashboard:main",
            "mcmf-backtest=validation.backtesting:main",
            "mcmf-simulate=monte_carlo_engine.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords="finance, monte-carlo, markov, quantitative, risk-management, derivatives, portfolio-optimization",
)
