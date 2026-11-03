# Quick Start Guide

## 🚀 Evidence-Aware Brain Tumour Classification System

### Installation

```bash
# Clone/navigate to project directory
cd Final_year_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

Place your brain tumour dataset in the following structure:

```
data/brain_tumours/
├── glioma/
│   ├── images/
│   │   ├── patient001_slice_01.png
│   │   └── ...
│   └── masks/
│       ├── patient001_slice_01.png
│       └── ...
├── meningioma/
│   ├── images/
│   └── masks/
└── pituitary/
    ├── images/
    └── masks/
```

### Usage

#### 1. Train + Evaluate (Full Pipeline)

```bash
python main.py --mode both --data_dir data/brain_tumours
```

This will:
- Train the CNN classifier
- Evaluate on test set
- Compute reliability metrics (AUROC, AUPRC)
- Generate risk-coverage analysis

#### 2. Single Image Prediction

```bash
python demo_single_prediction.py \
    --image data/brain_tumours/glioma/images/sample.png \
    --mask data/brain_tumours/glioma/masks/sample.png \
    --model models/best_classifier.pth
```

This will:
- Classify the tumour
- Generate Grad-CAM explanation
- Evaluate localization (IoU, Dice, Pointing Game)
- Assess stability (SSIM, Spearman correlation)
- Compute reliability score
- Generate visualization

#### 3. Only Training

```bash
python main.py --mode train --data_dir data/brain_tumours
```

#### 4. Only Evaluation

```bash
python main.py --mode eval --data_dir data/brain_tumours
```

### Output Structure

After running, you'll find:

```
models/
└── best_classifier.pth          # Trained model checkpoint

results/
└── prediction_*.png             # Visualization outputs

logs/
└── training.log                 # Training logs
```

### What Gets Evaluated

#### Classification Metrics
- Accuracy, Precision, Recall, F1-Score

#### Localization Metrics
- **IoU** (Intersection over Union)
- **Dice Coefficient**
- **Pointing Game** (max activation in mask?)
- **Energy Ratio** (heatmap energy inside mask)

#### Stability Metrics
- **SSIM** (Structural Similarity)
- **Spearman Rank Correlation**

#### Reliability Metrics
- **AUROC** (Area Under ROC Curve)
- **AUPRC** (Area Under Precision-Recall Curve)
- **Reliability Score** = 0.3×Confidence + 0.4×Localization + 0.3×Stability

### Sample Output

```
PREDICTION RESULTS
Class: GLIOMA
Confidence: 95.3%

RELIABILITY ASSESSMENT
Score: 0.847
Flag: RELIABLE

EVIDENCE BREAKDOWN
• Confidence: 0.953
• Localization: 0.781
• Stability: 0.824
```

### Troubleshooting

**Q: No module named 'src'**
```bash
# Make sure you're in the project root directory
export PYTHONPATH="${PYTHONPATH}:${PWD}"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

**Q: CUDA out of memory**
```python
# Edit src/config.py, reduce batch size:
TRAIN_CONFIG = {
    "batch_size": 16,  # Change from 32 to 16
    ...
}
```

**Q: Model not found**
```bash
# Train the model first
python main.py --mode train
```

### Next Steps

1. **Customize Configuration**: Edit `src/config.py`
2. **Adjust Reliability Weights**: Modify `RELIABILITY_CONFIG['weights']`
3. **Try Different Backbones**: Change `MODEL_CONFIG['backbone']` to 'resnet34', 'densenet121', etc.
4. **Tune Threshold**: Adjust `RELIABILITY_CONFIG['threshold']`

### Project Structure

```
Final_year_project/
├── src/                         # Source code
│   ├── data/                    # Data loading
│   ├── models/                  # CNN classifier
│   ├── explainability/          # Grad-CAM
│   ├── evaluation/              # Localization & stability
│   └── reliability/             # Reliability assessment
├── diagrams/                    # Generated diagrams
├── main.py                      # Main pipeline
├── demo_single_prediction.py    # Demo script
└── requirements.txt             # Dependencies
```

### Support

For issues or questions, check:
- README.md for detailed documentation
- src/config.py for all configuration options
- Individual module docstrings for technical details

---

**Academic Project** | Bharati Vidyapeeth / SPPU | 2026-2027
