# 🚀 How to Start Training

**Current Status**: Code is ready, but CUDA libraries are required for PyTorch.

---

## ⚠️ Important: CUDA Requirement

The current environment has PyTorch installed with CUDA support, but the CUDA runtime libraries are not available. You have two options:

### Option 1: Use a GPU Machine (Recommended)

Transfer your code to a machine with:
- ✅ NVIDIA GPU (GTX 1060+ or better)
- ✅ CUDA 11.7+ installed
- ✅ Python 3.8+
- ✅ PyTorch with CUDA support

### Option 2: Use CPU (Not Recommended - Very Slow)

Reinstall PyTorch for CPU-only:
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Warning**: Training on CPU will take 50-100x longer (~100+ hours instead of 2-3 hours).

---

## 🖥️ Recommended: Use Google Colab (FREE GPU!)

### Step 1: Upload Your Code

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. Enable GPU: Runtime > Change runtime type > GPU (T4)

### Step 2: Setup Environment

```python
# In Colab notebook
!git clone https://github.com/your-repo/Final_year_project.git  # Or upload manually
%cd Final_year_project

# Install requirements
!pip install -q torch torchvision scikit-learn scikit-image matplotlib seaborn tqdm
```

### Step 3: Upload Dataset

```python
# Option A: Upload from local (if dataset is on your machine)
from google.colab import files
# Upload brisc2025.zip, then:
!unzip brisc2025.zip -d data/brisc/

# Option B: Download directly (faster)
!wget https://figshare.com/ndownloader/files/51677885 -O brisc2025.zip
!unzip brisc2025.zip -d data/brisc/
```

### Step 4: Start Training

```python
# Quick test (5 minutes)
!python train.py --epochs 2 --batch-size 8

# Full training (2-3 hours)
!python train.py \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler cosine \
  --augmentation medium \
  --early-stopping 10
```

### Step 5: Download Results

```python
# Download trained model
from google.colab import files
files.download('models/checkpoints/best_model.pth')
files.download('models/checkpoints/training_curves.png')
files.download('models/checkpoints/confusion_matrix.png')
```

---

## 🖥️ Alternative: Use Your Own GPU Machine

### On Your GPU Machine:

```bash
# 1. Check GPU
nvidia-smi

# 2. Navigate to project
cd /path/to/Final_year_project

# 3. Install requirements (if needed)
pip install torch torchvision scikit-learn scikit-image matplotlib seaborn tqdm

# 4. Run startup script (interactive)
./start_training.sh

# Or directly:
python train.py \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler cosine \
  --augmentation medium \
  --early-stopping 10
```

---

## 🖥️ Alternative: Use Cloud Services

### AWS EC2 (with GPU)

```bash
# 1. Launch g4dn.xlarge instance (NVIDIA T4 GPU, ~$0.50/hr)
# 2. SSH into instance
# 3. Install CUDA and PyTorch
# 4. Clone/upload your code
# 5. Run training
```

### Azure ML / GCP Compute

Similar process - launch GPU-enabled VM and run training.

---

## 📋 Current Environment Issue

**Error**: `libcublas.so.*[0-9] not found in the system path`

**Cause**: PyTorch is compiled with CUDA support, but CUDA runtime libraries are not installed on the system.

**Solutions**:

### Solution 1: Install CUDA Toolkit

```bash
# For Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Solution 2: Use Conda (Easier)

```bash
# Create conda environment with CUDA
conda create -n tumor python=3.11
conda activate tumor
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install scikit-learn scikit-image matplotlib seaborn tqdm

# Then run training
python train.py --epochs 50 --batch-size 32
```

### Solution 3: CPU-Only PyTorch (Slow!)

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then run with --device cpu
python train.py --epochs 50 --batch-size 32 --device cpu
```

---

## ⚡ Quick Start Commands (When CUDA Works)

### Test Run (5 minutes)
```bash
python train.py --epochs 2 --batch-size 8
```

### Fast Baseline (30 minutes)
```bash
python train.py --backbone resnet18 --epochs 30 --batch-size 64
```

### Recommended (2-3 hours) ⭐
```bash
python train.py \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler cosine \
  --augmentation medium \
  --early-stopping 10
```

### Maximum Accuracy (6-8 hours)
```bash
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
  --use-amp
```

---

## 📊 What to Expect During Training

### Terminal Output
```
======================================================================
BRAIN TUMOR CLASSIFICATION - TRAINING
======================================================================

Configuration:
  Dataset:        data/brisc/brisc2025
  Backbone:       resnet50
  Epochs:         50
  Batch Size:     32
  Learning Rate:  0.0001
  Device:         cuda
======================================================================

Loading dataset...

======================================================================
BRISC DATASET INFORMATION
======================================================================
📊 CLASSIFICATION TASK:
  TRAIN Split: 3933 images
    - glioma      : 1147 (29.14%)
    - meningioma  : 1329 (33.78%)
    - pituitary   : 1457 (37.04%)
======================================================================

Creating model...
======================================================================
MODEL INFORMATION
======================================================================
Architecture:         resnet50
Number of Classes:    3
Feature Dimension:    2048
----------------------------------------------------------------------
Total Parameters:     23,528,515
Trainable Parameters: 23,528,515
======================================================================

======================================================================
STARTING TRAINING
======================================================================

Epoch 1/50 [Train]: 100%|██████| 124/124 [01:23<00:00]
  loss: 0.8234, acc: 67.82%
Validating: 100%|██████████████| 27/27 [00:18<00:00]

Epoch 1/50 Summary:
  Train Loss: 0.8234 | Train Acc: 67.82%
  Val Loss:   0.5621 | Val Acc:   78.14%
  Learning Rate: 0.000100
  Time: 101.45s
  ✓ New best model! (Val Acc: 78.14%)
----------------------------------------------------------------------

[... training continues ...]

Epoch 50/50 Summary:
  Train Loss: 0.0521 | Train Acc: 98.23%
  Val Loss:   0.1834 | Val Acc:   95.67%
  Learning Rate: 0.000001
  Time: 98.12s
----------------------------------------------------------------------

======================================================================
TRAINING COMPLETE
======================================================================
Total Time: 82.34 minutes
Best Val Acc: 95.67% (Epoch 47)
======================================================================
```

### Generated Files
```
models/checkpoints/
├── best_model.pth              # Best model checkpoint
├── last_model.pth              # Last epoch checkpoint
├── training_history.json       # Loss/acc per epoch
├── training_curves.png         # Training visualization
└── confusion_matrix.png        # Final confusion matrix
```

---

## 🎯 Recommended Path Forward

Given the CUDA library issue in your current environment:

### Immediate Option: Google Colab (FREE!)

1. **Go to**: https://colab.research.google.com/
2. **Enable GPU**: Runtime > Change runtime type > GPU
3. **Upload code**: Use file upload or mount Google Drive
4. **Upload dataset**: Use Colab file upload or wget
5. **Run training**: Copy commands from above

**Advantages**:
- ✅ Free GPU (Tesla T4)
- ✅ No setup required
- ✅ Works immediately
- ✅ 2-3 hours training time

### Long-term Option: Fix Local CUDA

1. **Install CUDA Toolkit**: Follow Solution 1 or 2 above
2. **Test PyTorch**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Run training**: Use `./start_training.sh`

**Advantages**:
- ✅ Your own hardware
- ✅ Faster if you have good GPU
- ✅ No upload/download needed

---

## 💡 My Recommendation

**Use Google Colab for your first training run!**

Reasons:
1. ✅ Works immediately (no CUDA setup)
2. ✅ Free GPU access
3. ✅ Takes 2-3 hours total
4. ✅ You can verify everything works
5. ✅ Then fix local CUDA at your leisure

After successful Colab training, you'll have:
- ✅ Trained model weights
- ✅ Confidence the code works
- ✅ Baseline results to compare
- ✅ Can proceed to Increment 2 (Grad-CAM)

---

## 📞 Need Help?

If you encounter issues:

1. **CUDA errors**: Use Colab or reinstall PyTorch
2. **Out of memory**: Reduce batch size (`--batch-size 16` or `8`)
3. **Dataset errors**: Run `python test_data_basic.py`
4. **Slow training**: Enable AMP (`--use-amp`) or reduce epochs

---

**Next Steps**: Choose your path (Colab recommended) and start training! 🚀
