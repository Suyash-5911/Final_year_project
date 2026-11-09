#!/bin/bash

# GPU Setup Script for NVIDIA Quadro P1000
# Installs PyTorch with CUDA support using Conda

echo "=========================================="
echo "GPU Setup for Brain Tumor Classification"
echo "=========================================="
echo ""
echo "Detected GPU: NVIDIA Quadro P1000"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo ""
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "Or use the existing anaconda3 installation at:"
    echo "  /home/itd-2495/anaconda3/bin/conda"
    exit 1
fi

echo "Step 1: Creating conda environment..."
echo "--------------------------------------"

# Use existing conda
CONDA_EXE="/home/itd-2495/anaconda3/bin/conda"

# Create environment
$CONDA_EXE create -n tumor_classification python=3.11 -y

echo ""
echo "Step 2: Installing PyTorch with CUDA..."
echo "--------------------------------------"

# Activate and install PyTorch
source /home/itd-2495/anaconda3/bin/activate tumor_classification

# Install PyTorch with CUDA 11.8 (compatible with Quadro P1000)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo ""
echo "Step 3: Installing other dependencies..."
echo "--------------------------------------"

# Install other packages
conda install scikit-learn scikit-image matplotlib seaborn -y
pip install tqdm

echo ""
echo "Step 4: Verifying installation..."
echo "--------------------------------------"

python << EOF
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ CUDA not available")
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  conda activate tumor_classification"
echo ""
echo "To start training:"
echo "  cd /home/itd-2495/Documents/Final_year_project"
echo "  python train.py --backbone resnet50 --epochs 50 --batch-size 16"
echo ""
echo "Note: Using batch-size 16 for 4GB GPU"
echo "=========================================="
