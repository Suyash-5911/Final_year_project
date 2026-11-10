#!/bin/bash

# Quick start script for Brain Tumor Classifier Dashboard

echo "🧠 Brain Tumor Classifier Dashboard"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check if model exists
if [ ! -f "models/best_model_v2.pth" ]; then
    echo ""
    echo "⚠️  Model file not found: models/best_model_v2.pth"
    echo "Please ensure the trained model is in the models/ directory."
    exit 1
fi

echo "✓ Model file found"

# Run Streamlit
echo ""
echo "Starting Streamlit dashboard..."
echo "Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
