# 🔧 Manual GPU Setup Guide

**Your GPU**: NVIDIA Quadro P1000 (4GB GDDR5)

---

## ⚠️ Current Issue

The conda environment installation is having issues. Let's fix this with a manual approach.

---

## 📋 Option 1: Clean Conda Install (RECOMMENDED)

### Step 1: Remove old environment

```bash
conda env remove -n tumor_classification
```

### Step 2: Create fresh environment with PyTorch

```bash
# Create environment
conda create -n tumor_classification python=3.11 -y

# Activate it
conda activate tumor_classification

# Install PyTorch with CUDA (all in one command)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install scikit-learn scikit-image matplotlib seaborn tqdm
```

### Step 3: Verify

```bash
python << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
```

---

## 📋 Option 2: Use System Python (If Option 1 Fails)

Your system already has PyTorch 1.11.0 installed, but with missing CUDA libraries.

### Step 1: Install CUDA Toolkit

```bash
# Download CUDA 11.3 (matches your PyTorch version)
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run

# Install (no driver, just toolkit)
sudo sh cuda_11.3.0_465.19.01_linux.run --silent --toolkit

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Verify

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📋 Option 3: Use Colab Instead (EASIEST!)

Since local setup is proving difficult, **Google Colab is still the fastest option**:

1. Upload `Brain_Tumor_Training_Colab.ipynb` to Colab
2. Enable GPU
3. Start training immediately

No setup needed!

---

## 🎯 Option 4: Docker with NVIDIA Runtime (Advanced)

```bash
# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run PyTorch container
docker run --gpus all -it -v $(pwd):/workspace \
    pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime \
    bash
```

---

## ✅ Recommended Path Forward

Given the issues, here's what I recommend:

### **Immediate Solution: Google Colab**
- ✅ No setup issues
- ✅ Free GPU
- ✅ Works immediately
- ✅ 2-3 hours training

### **Long-term Solution: Fix Local Setup**
Try Option 1 (Clean Conda Install) when you have time.

---

## 🔍 Debugging Commands

### Check GPU
```bash
lspci | grep -i nvidia  # Should show Quadro P1000
```

### Check NVIDIA Driver
```bash
lsmod | grep nvidia     # Should show nvidia kernel modules
nvidia-smi             # Should show GPU info
```

### Check CUDA
```bash
nvcc --version         # Should show CUDA version
```

### Check PyTorch
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## 💡 What Went Wrong?

The conda environment creation worked, but PyTorch installation defaulted to using system Python 3.8 instead of the conda environment Python 3.11.

**Fix**: Use Option 1 above with explicit pip install in the activated conda environment.

---

## 🚀 Quick Decision Guide

**Need to train NOW?** → Use Google Colab

**Want local training?** → Try Option 1 (Clean Conda Install)

**Option 1 fails?** → Try Option 2 (CUDA Toolkit)

**Still issues?** → Use Colab, fix local setup later

---

## 📞 Next Steps

1. Try Option 1 (Clean Conda Install)
2. If it works: Start training!
3. If it fails: Use Colab for now
4. Fix local setup in parallel

---

**Remember**: Your code is 100% ready. It's just a PyTorch/CUDA setup issue that can be resolved!
