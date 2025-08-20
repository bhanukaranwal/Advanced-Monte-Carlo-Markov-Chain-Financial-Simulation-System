# Monte Carlo-Markov Finance System Makefile

.PHONY: help install install-dev test test-cov lint format type-check clean build deploy docs

# Default target
help:
	@echo "Monte Carlo-Markov Finance System - Available commands:"
	@echo ""
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run test suite"
	@echo "  test-cov     Run tests with coverage"
	@echo "  benchmark    Run performance benchmarks"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run mypy type checking"
	@echo "  clean        Clean temporary files"
	@echo "  build        Build package"
	@echo "  deploy       Deploy with Docker Compose"
	@echo "  docs         Generate documentation"
	@echo ""

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

install-gpu:
	pip install -r requirements.txt
	pip install -r requirements-gpu.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

benchmark:
	pytest tests/test_performance.py --benchmark-only

test-integration:
	pytest tests/test_integration.py -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/

# Building
build: clean
	python setup.py sdist bdist_wheel

build-docker:
	docker build -t mcmf-system .

build-docker-gpu:
	docker build -t mcmf-system-gpu --target gpu .

# Deployment
deploy:
	docker-compose up -d

deploy-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

stop:
	docker-compose down

logs:
	docker-compose logs -f mcmf-app

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Development helpers
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

dashboard:
	streamlit run src/visualization/dashboard.py --server.port=8501

# Performance profiling
profile:
	python -m cProfile -o profile.stats examples/performance_test.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile:
	python -m memory_profiler examples/memory_test.py

# Data management
download-data:
	python scripts/download_market_data.py

preprocess-data:
	python scripts/preprocess_data.py

# CI/CD helpers
ci-install:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .

ci-test: test-cov lint type-check

# Security
security-check:
	bandit -r src/
	safety check

# Release
release-patch:
	bump2version patch
	git push origin main --tags

release-minor:
	bump2version minor
	git push origin main --tags

release-major:
	bump2version major
	git push origin main --tags

# Monitoring
start-monitoring:
	docker-compose up -d prometheus grafana

stop-monitoring:
	docker-compose stop prometheus grafana
