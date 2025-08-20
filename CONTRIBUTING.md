# Contributing to Monte Carlo-Markov Finance System

Thank you for your interest in contributing to MCMF! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct. By participating, you're expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before creating bug reports, check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**  
- **Environment details** (OS, Python version, etc.)
- **Code samples** if applicable

### Suggesting Features

Feature suggestions are welcome! Please:

- **Use a clear title** describing the feature
- **Provide detailed description** of the proposed functionality
- **Explain the use case** and benefits
- **Consider implementation complexity**

### Pull Requests

1. **Fork** the repository
2. **Create a feature branch** from `develop`
3. **Make your changes** following code standards
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Submit a pull request**

#### Pull Request Process

- Ensure CI checks pass
- Update documentation for any new features
- Add or update tests as appropriate
- Follow the existing code style
- Write clear commit messages

## Development Setup

Clone your fork
git clone https://github.com/your-username/monte-carlo-markov-finance.git
cd monte-carlo-markov-finance

Setup development environment
./scripts/setup_dev_environment.sh

Install pre-commit hooks
pre-commit install
## Code Standards

### Python Style Guide

- Follow **PEP 8** guidelines
- Use **Black** for code formatting
- Use **isort** for import sorting
- Add **type hints** to all functions
- Write **comprehensive docstrings**

### Testing

- Write tests for all new features
- Maintain >90% test coverage
- Use pytest for testing framework
- Include both unit and integration tests

### Documentation

- Update docstrings for any changed functions
- Add examples for new features
- Update README if needed
- Follow Google-style docstrings

## Project Structure

src/
├── monte_carlo_engine/ # Core simulation engines
├── markov_models/ # Markov chain models
├── analytics_engine/ # Risk analytics
├── api/ # REST and WebSocket APIs
├── visualization/ # Dashboards and reports
└── utils/ # Utility functions

tests/ # Test suite
docs/ # Documentation
examples/ # Usage examples
deployment/ # Docker and K8s configs
## Commit Guidelines

### Commit Message Format

<type>(<scope>): <description>

<body> <footer> ```
Types:

feat: New feature

fix: Bug fix

docs: Documentation changes

style: Code style changes

refactor: Code refactoring

perf: Performance improvements

test: Test additions/modifications

chore: Maintenance tasks

Example:
feat(monte_carlo): add Heston model simulation

Implement stochastic volatility simulation using Heston model
with calibration functionality and Greeks calculation.

Closes #123Release Process
Version bump in setup.py and __init__.py

Update CHANGELOG.md

Create release PR to main branch

Tag release after merge

Deploy to production environment

Getting Help
GitHub Issues: Bug reports and feature requests

Email: bhanu@karanwalcapital.com

Recognition
Contributors will be acknowledged in:

README.md contributors section

Release notes for significant contributions

Documentation for major features

Thank you for contributing to MCMF! 🚀


