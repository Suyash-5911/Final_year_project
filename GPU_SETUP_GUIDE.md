# 🎮 Your GPU Setup Guide

**Detected**: NVIDIA Quadro P1000 (4GB GDDR5, 640 CUDA cores)

---

## ✅ Good News!

You have a capable GPU for training! The Quadro P1000 is perfect for this project.

---

## 📊 What Your GPU Can Handle

| Configuration | Feasible? | Batch Size | Training Time |
|---------------|-----------|------------|---------------|
| ResNet18      | ✅ Yes    | 32         | ~1 hour       |
| ResNet50      | ✅ Yes    | 16         | ~3-4 hours    |
| ResNet101     | ⚠️ Tight  | 8          | ~8-10 hours   |

**Recommended**: ResNet50 with batch-size 16

---

## 🔧 Setup Steps (Easy - 5 minutes)

### Step 1: Run Setup Script

I created `setup_gpu.sh` that does everything automatically:

```bash
cd /home/itd-2495/Documents/Final_year_project
./setup_gpu.sh
```

**What it does**:
1. Creates conda environment `tumor_classification`
2. Installs PyTorch with CUDA 11.8
3. Installs all dependencies
4. Verifies GPU access

### Step 2: Activate Environment

```bash
conda activate tumor_classification
```

### Step 3: Verify GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU: Quadro P1000
```

### Step 4: Start Training!

```bash
# Recommended configuration for Quadro P1000
python train.py \
  --backbone resnet50 \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --optimizer adam \
  --scheduler cosine \
  --augmentation medium \
  --early-stopping 10
```

---

## 🎯 Optimized Training Commands for Your GPU

### Quick Test (5 minutes)
```bash
python train.py --epochs 2 --batch-size 8
```

### Fast Baseline (1 hour) - ResNet18
```bash
python train.py \
  --backbone resnet18 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3
```

### Recommended (3-4 hours) - ResNet50 ⭐
```bash
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

### Maximum (Tight fit) - ResNet101
```bash
python train.py \
  --backbone resnet101 \
  --epochs 50 \
  --batch-size 8 \
  --lr 5e-5 \
  --use-amp
```

**Note**: `--use-amp` enables mixed precision training = 2x faster + less memory!

---

## ⚡ Performance Tips for 4GB GPU

### If you get "Out of Memory" error:

1. **Reduce batch size**:
   ```bash
   --batch-size 8  # or even 4
   ```

2. **Use mixed precision** (saves memory):
   ```bash
   --use-amp
   ```

3. **Use smaller backbone**:
   ```bash
   --backbone resnet18  # or resnet34
   ```

4. **Reduce image size** (edit in code if needed):
   ```python
   --image-size 192  # instead of 224
   ```

---

## 📋 Manual Setup (If script doesn't work)

```bash
# 1. Create conda environment
/home/itd-2495/anaconda3/bin/conda create -n tumor_classification python=3.11 -y

# 2. Activate environment
source /home/itd-2495/anaconda3/bin/activate tumor_classification

# 3. Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install other packages
conda install scikit-learn scikit-image matplotlib seaborn -y
pip install tqdm

# 5. Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🔍 Troubleshooting

### Issue 1: Conda command not found

```bash
# Use full path
/home/itd-2495/anaconda3/bin/conda --version

# Or add to PATH
export PATH="/home/itd-2495/anaconda3/bin:$PATH"
```

### Issue 2: GPU still not detected after setup

```bash
# Check if NVIDIA driver needs to be installed
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue 3: Out of Memory during training

```bash
# Start with smaller batch size
python train.py --batch-size 4 --use-amp
```

### Issue 4: Training very slow

```bash
# Verify GPU is being used
python -c "import torch; x = torch.randn(1000, 1000).cuda(); print('GPU working!' if x.is_cuda else 'Using CPU')"

# Check GPU usage during training
watch -n 1 nvidia-smi  # In another terminal
```

---

## 📊 Expected Performance (Quadro P1000)

### ResNet50, Batch Size 16, 50 epochs:

| Metric | Value |
|--------|-------|
| Training Time | 3-4 hours |
| Val Accuracy | 92-95% |
| Memory Usage | ~3.5 GB |
| GPU Utilization | ~90-95% |

---

## ✅ Quick Start Checklist

- [ ] Run `./setup_gpu.sh`
- [ ] Activate conda environment
- [ ] Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Quick test: `python train.py --epochs 2 --batch-size 8`
- [ ] Full training: Use recommended command above
- [ ] Monitor: `watch -n 1 nvidia-smi` (separate terminal)

---

## 🎯 Recommended Workflow

### First Time:
1. Run setup script (5 min)
2. Quick test run (5 min)
3. If test passes, start full training (3-4 hours)

### Every Time After:
```bash
# Activate environment
conda activate tumor_classification

# Navigate to project
cd /home/itd-2495/Documents/Final_year_project

# Start training
python train.py --backbone resnet50 --epochs 50 --batch-size 16 --use-amp
```

---

## 💡 Pro Tips for Quadro P1000

1. **Always use `--use-amp`** (2x faster, less memory)
2. **Monitor GPU**: `watch -n 1 nvidia-smi` in separate terminal
3. **Batch size 16 is optimal** for ResNet50
4. **Close other GPU apps** while training
5. **Check temperature**: Should stay under 85°C

---

## 🚀 Ready to Start!

Once you run `./setup_gpu.sh`, you'll be ready to train!

**Next commands**:
```bash
# Setup (5 min)
./setup_gpu.sh

# Activate
conda activate tumor_classification

# Train!
python train.py --backbone resnet50 --epochs 50 --batch-size 16 --use-amp
```

---

**Your GPU is perfect for this project!** 🎮
