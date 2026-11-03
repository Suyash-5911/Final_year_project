# Evidence-Aware Brain Tumour Classification System

An explainable AI system for brain tumour classification using CNN with Grad-CAM explanations, localization metrics, and reliability assessment.

## Features

- **CNN Classification**: ResNet-based classifier for 3 tumour types (Glioma, Meningioma, Pituitary)
- **Visual Explanations**: Grad-CAM heatmap generation
- **Localization Evaluation**: IoU, Dice, Pointing Game, Energy Ratio metrics
- **Stability Assessment**: Perturbation-based stability using SSIM and rank correlation
- **Reliability Scoring**: Composite score combining confidence, localization, and stability
- **Interactive Dashboard**: Streamlit web application for clinical use

## Project Structure

```
├── src/
│   ├── data/              # Dataset loading and preprocessing
│   ├── models/            # CNN classifier architecture
│   ├── explainability/    # Grad-CAM implementation
│   ├── evaluation/        # Localization and stability metrics
│   ├── reliability/       # Reliability assessment
│   └── dashboard/         # Streamlit web app
├── data/                  # Dataset directory
├── models/                # Saved model weights
├── results/               # Experiment results
├── diagrams/              # System diagrams (auto-generated)
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train Classifier

```python
from src.models import TumourClassifier, ModelTrainer
from src.data import DatasetLoader, Preprocessor, TrainTransform

# Load data
loader = DatasetLoader('data/brain_tumours')
preprocessor = Preprocessor()

train_loader = loader.get_data_loader('train', transform=TrainTransform(preprocessor))
val_loader = loader.get_data_loader('val', transform=ValTestTransform(preprocessor))

# Initialize model and trainer
model = TumourClassifier(backbone='resnet50', num_classes=3)
trainer = ModelTrainer(model)

# Train
for epoch in range(50):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate(val_loader)
    print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
```

### 2. Generate Explanations

```python
from src.explainability import ExplanationGenerator

# Generate Grad-CAM
explainer = ExplanationGenerator(model)
cam = explainer.generate_cam(image_tensor)

# Overlay on image
overlay = explainer.overlay_on_image(original_image, cam)
```

### 3. Evaluate Reliability

```python
from src.reliability import ReliabilityAnalyzer

analyzer = ReliabilityAnalyzer(model, explainer, evaluator)
reliability_score, flag = analyzer.assess_reliability(image, mask)
print(f"Reliability Score: {reliability_score:.3f} - {flag}")
```

### 4. Run Dashboard

```bash
streamlit run src/dashboard/app.py
```

## Model Development Timeline

- **Aug-Sep 2026**: Literature survey and problem finalization
- **Oct 2026**: System design
- **Nov-Dec 2026**: Dataset preparation and classifier training
- **Dec 2026-Jan 2027**: Explanation generation and localization eval
- **Jan-Feb 2027**: Perturbation and stability evaluation
- **Feb-Mar 2027**: Reliability assessment
- **Mar 2027**: Dashboard development
- **Apr 2027**: Testing and final report

## Methodology

### Incremental Development (5 Increments)

1. **Dataset & Classifier**: Patient-wise split, CNN training, baseline metrics
2. **Explanation & Localization**: Grad-CAM, IoU, Dice, Pointing Game
3. **Perturbation & Stability**: Noise/blur perturbations, SSIM, rank correlation
4. **Reliability Assessment**: Composite scoring, AUROC/AUPRC evaluation
5. **Dashboard & Release**: Streamlit app with visual evidence viewer

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Localization**: IoU, Dice Coefficient, Pointing Game, Energy Ratio
- **Stability**: SSIM, Spearman Rank Correlation
- **Reliability**: AUROC, AUPRC, Risk-Coverage Analysis

## License

Academic project for Bharati Vidyapeeth / SPPU

## Contributors

Final Year Project Team
