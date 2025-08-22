# Multi-stage Dockerfile for MCMF API
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
        libhdf5-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r mcmf && useradd -r -g mcmf mcmf

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Set ownership
RUN chown -R mcmf:mcmf /app

# Switch to non-root user
USER mcmf

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Development command
CMD ["python", "-m", "src.api.main"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY alembic.ini .
COPY alembic/ ./alembic/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/temp \
    && chown -R mcmf:mcmf /app

# Switch to non-root user
USER mcmf

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Production command with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "src.api.main:app"]

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini .
COPY .coveragerc .

# Run tests
RUN python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Security scanning stage  
FROM base as security

# Install security tools
RUN pip install bandit safety

# Copy source code
COPY src/ ./src/

# Run security scans
RUN bandit -r src/ -f json -o bandit-report.json || true
RUN safety check --json --output safety-report.json || true

# Worker Dockerfile (separate for GPU support)
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-worker-base

# Install Python and dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create app user
RUN groupadd -r mcmf && useradd -r -g mcmf mcmf

WORKDIR /app

# Install Python packages with CUDA support
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip install cupy-cuda11x

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

RUN chown -R mcmf:mcmf /app
USER mcmf

# Set CUDA environment
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "-m", "src.workers.monte_carlo_worker"]

# CPU worker stage
FROM base as cpu-worker

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

RUN chown -R mcmf:mcmf /app
USER mcmf

CMD ["python", "-m", "src.workers.monte_carlo_worker"]
