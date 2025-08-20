#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up MCMF development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [[ $(echo "$python_version >= $required_version" | bc -l) -eq 0 ]]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data reports config

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📄 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

# Run tests to verify setup
echo "🧪 Running tests to verify setup..."
pytest tests/ -v --tb=short

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Start PostgreSQL and Redis services"
echo "3. Run 'make test' to verify everything works"
echo "4. Run 'streamlit run src/visualization/dashboard.py' to start the dashboard"
