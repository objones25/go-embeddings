#!/bin/bash

# Exit on any error
set -e

# Get the script's directory
SCRIPT_DIR=$(dirname "$0")
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
    echo "📝 Loading environment variables..."
    source "$ROOT_DIR/.env"
    
    # Verify HuggingFace API key exists
    if [ -z "$HUGGINGFACE_API_KEY" ]; then
        echo "❌ HUGGINGFACE_API_KEY not found in .env file"
        echo "Please add your HuggingFace API key to .env file"
        exit 1
    fi
else
    echo "⚠️  No .env file found at $ROOT_DIR/.env"
    echo "Please create one with your HuggingFace API key"
    exit 1
fi

echo "📦 Setting up environment..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create and activate virtual environment in the root directory
if [ ! -d "$ROOT_DIR/venv" ]; then
    python3 -m venv "$ROOT_DIR/venv"
fi

source "$ROOT_DIR/venv/bin/activate" || {
    echo "❌ Failed to activate virtual environment"
    exit 1
}

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip || {
    echo "❌ Failed to upgrade pip"
    exit 1
}

# Install requirements from the scripts directory
echo "📥 Installing requirements..."
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "❌ requirements.txt not found at $SCRIPT_DIR/requirements.txt"
    exit 1
fi

pip install -r "$SCRIPT_DIR/requirements.txt" || {
    echo "❌ Failed to install requirements"
    exit 1
}

# Create models directory in root
echo "📁 Creating models directory..."
mkdir -p "$ROOT_DIR/models"

# Run the conversion script for all available models
echo "🚀 Running model conversion..."
if [ ! -f "$SCRIPT_DIR/convert_model.py" ]; then
    echo "❌ convert_model.py not found at $SCRIPT_DIR/convert_model.py"
    exit 1
fi

# Convert all models with optimizations
python "$SCRIPT_DIR/convert_model.py" \
    "all-mpnet-base-v2" \
    "all-MiniLM-L6-v2" \
    "paraphrase-multilingual-MiniLM-L12-v2" \
    "all-roberta-large-v1" \
    "multi-qa-MiniLM-L6-cos-v1" \
    --output-dir "$ROOT_DIR/models" \
    --optimize \
    --quantize \
    --max-workers 2 || {
    echo "❌ Model conversion failed"
    exit 1
}

echo "✨ Setup complete!"

# Deactivate virtual environment
deactivate

# Final status check
echo "🔍 Verifying setup..."
if [ -d "$ROOT_DIR/models" ] && [ -d "$ROOT_DIR/venv" ]; then
    echo "✅ Setup completed successfully!"
    
    # List downloaded models
    echo "📦 Downloaded models:"
    ls -l "$ROOT_DIR/models"
else
    echo "⚠️  Setup may have incomplete components"
fi