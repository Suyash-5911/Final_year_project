# ✅ Working Local GPU Setup - Step by Step

**Your GPU**: NVIDIA Quadro P1000 (Detected!)

---

## 🔧 The Issue We're Facing

Your system has complex Python environment interactions:
- System Python 3.8 (in `/usr/bin/`)
- Anaconda Python (multiple versions)
- User-local packages (in `~/.local/`)
- Conda environments

The pip commands keep defaulting to system/user paths instead of conda environment.

---

## ✅ WORKING SOLUTION: Use Conda with Proper Isolation

### Step 1: Clean Start

```bash
# Remove any conflicting environments
conda env remove -n tumor -y
conda env remove -n tumor_classification -y

# Clear pip cache
rm -rf ~/.cache/pip
```

### Step 2: Create Environment with NO User Packages

```bash
# Create environment
conda create -n tumor python=3.10 -y

# Activate it
conda activate tumor

# IMPORTANT: Disable user site packages
export PYTHONNOUSERSITE=1
```

### Step 3: Install PyTorch (Choose ONE method)

**Method A: Conda (Preferred if it works)**
```bash
conda activate tumor
export PYTHONNOUSERSITE=1
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
```

**Method B: Pip with proper isolation**
```bash
conda activate tumor
export PYTHONNOUSERSITE=1
# Update pip first
python -m pip install --upgrade pip
# Install PyTorch
python -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
```

### Step 4: Install Other Dependencies

```bash
conda activate tumor
export PYTHONNOUSERSITE=1
python -m pip install --no-cache-dir scikit-learn scikit-image matplotlib seaborn tqdm
```

### Step 5: Verify Installation

```bash
conda activate tumor
export PYTHONNOUSERSITE=1

python << 'EOF'
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

import torch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ GPU READY!")
else:
    print("\n⚠️ GPU not detected")
    print("Checking CUDA...")
    print(f"CUDA version in PyTorch: {torch.version.cuda}")
EOF
```

---

## 🎯 Alternative: Use System Python with Virtual Environment

If conda continues to have issues:

```bash
# Use system Python 3.10 (if available)
python3.10 -m venv ~/tumor_venv
source ~/tumor_venv/bin/activate

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install dependencies
pip install scikit-learn scikit-image matplotlib seaborn tqdm

# Test
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🚀 Once Working: Start Training

### Quick Test (5 minutes)

```bash
conda activate tumor
export PYTHONNOUSERSITE=1
cd /home/itd-2495/Documents/Final_year_project

python train.py --epochs 2 --batch-size 8
```

### Full Training (3-4 hours)

```bash
conda activate tumor
export PYTHONNOUSERSITE=1
cd /home/itd-2495/Documents/Final_year_project

python train.py \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler cosine \
  --augmentation medium \
  --early-stopping 10 \
  --use-amp
```

---

## 💡 Pro Tips

### Always Use These When Activating

```bash
conda activate tumor
export PYTHONNOUSERSITE=1  # Prevents user-site conflicts
```

Add to your `~/.bashrc`:
```bash
alias train_tumor='conda activate tumor && export PYTHONNOUSERSITE=1'
```

Then just run:
```bash
train_tumor
cd /home/itd-2495/Documents/Final_year_project
python train.py ...
```

---

## 🔍 Debugging Commands

### Check which Python is being used
```bash
conda activate tumor
which python
python -c "import sys; print(sys.executable)"
```

### Check where packages are installed
```bash
conda activate tumor
python -c "import torch; print(torch.__file__)"
python -c "import sys; print(sys.path)"
```

### Check GPU from command line
```bash
lspci | grep -i nvidia  # Should show Quadro P1000
```

---

## ⚠️ If Still Not Working

### Last Resort: Docker

```bash
# Install Docker with NVIDIA support
# Then run:
docker run --gpus all -it --rm \
  -v /home/itd-2495/Documents/Final_year_project:/workspace \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  bash

# Inside container:
cd /workspace
pip install scikit-learn scikit-image matplotlib seaborn tqdm
python train.py --backbone resnet50 --epochs 50 --batch-size 16
```

---

## 📊 Success Indicators

You'll know it's working when:

✅ `python -c "import torch; print(torch.cuda.is_available())"` returns `True`  
✅ `python train.py --epochs 2 --batch-size 8` runs without errors  
✅ GPU utilization shows in `nvidia-smi` (if drivers installed)  

---

## 🎯 My Recommendation

1. **Try Step 1-5 above** (15 minutes)
2. **If it works**: Start training immediately
3. **If it fails**: Note the error and we'll troubleshoot
4. **If frustrated**: Use Colab (works 100% of the time)

---

## 📞 Common Errors & Fixes

### Error: "CUDA not available"
```bash
# Check if pytorch-cuda package is installed
conda list | grep cuda

# Reinstall if missing
conda install pytorch-cuda=11.7 -c nvidia -y
```

### Error: "undefined symbol" or "library not found"
```bash
# Clean install
conda env remove -n tumor -y
# Start from Step 2 again
```

### Error: "Permission denied" when installing
```bash
# Don't use sudo with conda/pip
# Make sure you're in conda environment
conda activate tumor
```

---

## ✅ Next Steps

1. Run Steps 1-5 above
2. When `torch.cuda.is_available()` returns `True`
3. Run quick test: `python train.py --epochs 2 --batch-size 8`
4. If successful, start full training!

---

**Let me know what error you get (if any) and I'll help troubleshoot!** 🚀
