# Multi-stage Docker build for Monte Carlo-Markov Finance System
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-prod.txt

# Development stage
FROM base as development

COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

COPY . .
RUN pip install -e .

CMD ["python", "-m", "pytest", "tests/"]

# Production stage
FROM base as production

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install the package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcmf
USER mcmf

# Expose ports
EXPOSE 8501 8050 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import monte_carlo_engine; print('OK')" || exit 1

# Default command
CMD ["streamlit", "run", "--server.port=8501", "--server.address=0.0.0.0", "src/visualization/dashboard.py"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
COPY requirements-gpu.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-gpu.txt

# Copy application
COPY src/ ./src/
COPY setup.py .
COPY README.md .

RUN pip install .

EXPOSE 8501 8050 8000

CMD ["python", "-c", "import cupy; print('GPU acceleration available')"]
