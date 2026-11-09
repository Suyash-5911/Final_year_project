#!/bin/bash

# Brain Tumor Classification - Training Startup Script
# Created: November 9, 2026
# Purpose: Start training with proper environment checks

echo "=========================================="
echo "Brain Tumor Classification - Training"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Python
echo "Checking Python..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found!"
    exit 1
fi

# Check 2: GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}✓${NC} GPU found: $GPU_INFO"
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠${NC} No GPU found. Training will use CPU (very slow!)"
    DEVICE="cpu"
fi

# Check 3: PyTorch
echo ""
echo "Checking PyTorch..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} PyTorch is working"
else
    echo -e "${RED}✗${NC} PyTorch has issues (CUDA libraries may be missing)"
    echo ""
    echo "Solutions:"
    echo "  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  2. Or reinstall PyTorch CPU-only: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check 4: Dataset
echo ""
echo "Checking dataset..."
if [ -d "data/brisc/brisc2025" ]; then
    TRAIN_COUNT=$(find data/brisc/brisc2025/classification_task/train -name "*.jpg" 2>/dev/null | wc -l)
    TEST_COUNT=$(find data/brisc/brisc2025/classification_task/test -name "*.jpg" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓${NC} Dataset found: $TRAIN_COUNT train, $TEST_COUNT test images"
else
    echo -e "${RED}✗${NC} Dataset not found at data/brisc/brisc2025"
    exit 1
fi

# Check 5: Code
echo ""
echo "Checking code..."
if [ -f "train.py" ] && [ -d "src/models" ] && [ -d "src/data" ]; then
    echo -e "${GREEN}✓${NC} Training code found"
else
    echo -e "${RED}✗${NC} Training code missing!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Environment Check Complete!"
echo "=========================================="
echo ""

# Training configurations
echo "Available training configurations:"
echo ""
echo "1. FAST BASELINE (30 min, ResNet18)"
echo "   python train.py --backbone resnet18 --epochs 30 --batch-size 64"
echo ""
echo "2. RECOMMENDED (2-3 hrs, ResNet50)"
echo "   python train.py --backbone resnet50 --epochs 50 --batch-size 32 --lr 1e-4"
echo ""
echo "3. MAXIMUM ACCURACY (6-8 hrs, ResNet101)"
echo "   python train.py --backbone resnet101 --freeze-backbone --unfreeze-after 20 --epochs 100 --batch-size 16"
echo ""
echo "4. QUICK TEST (5 min)"
echo "   python train.py --epochs 2 --batch-size 8"
echo ""

# Ask user which config
read -p "Select configuration (1-4) or press Enter for RECOMMENDED: " CONFIG_CHOICE

case $CONFIG_CHOICE in
    1)
        echo ""
        echo "Starting FAST BASELINE training..."
        python train.py \
            --backbone resnet18 \
            --epochs 30 \
            --batch-size 64 \
            --lr 1e-3 \
            --device $DEVICE
        ;;
    3)
        echo ""
        echo "Starting MAXIMUM ACCURACY training..."
        python train.py \
            --backbone resnet101 \
            --freeze-backbone \
            --unfreeze-after 20 \
            --epochs 100 \
            --batch-size 16 \
            --lr 5e-5 \
            --optimizer adamw \
            --augmentation heavy \
            --early-stopping 15 \
            --device $DEVICE
        ;;
    4)
        echo ""
        echo "Starting QUICK TEST..."
        python train.py \
            --epochs 2 \
            --batch-size 8 \
            --device $DEVICE
        ;;
    *)
        echo ""
        echo "Starting RECOMMENDED training..."
        python train.py \
            --backbone resnet50 \
            --epochs 50 \
            --batch-size 32 \
            --lr 1e-4 \
            --optimizer adam \
            --scheduler cosine \
            --augmentation medium \
            --early-stopping 10 \
            --device $DEVICE
        ;;
esac

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  Best model: models/checkpoints/best_model.pth"
echo "  Training curves: models/checkpoints/training_curves.png"
echo "  Confusion matrix: models/checkpoints/confusion_matrix.png"
echo ""
