"""
Configuration file for Evidence-Aware Brain Tumour Classification System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet50",
    "num_classes": 3,
    "class_names": ["glioma", "meningioma", "pituitary"],
    "input_size": (224, 224),
    "pretrained": True,
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "optimizer": "adam",
    "loss_function": "cross_entropy",
    "early_stopping_patience": 10,
}

# Data preprocessing
PREPROCESS_CONFIG = {
    "target_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "intensity_range": (0, 1),
}

# Explanation configuration (Grad-CAM)
EXPLANATION_CONFIG = {
    "method": "grad-cam",
    "target_layer": "layer4",
    "colormap": "jet",
    "alpha": 0.5,
}

# Localization evaluation
LOCALIZATION_CONFIG = {
    "binarize_method": "otsu",
    "threshold": 0.5,
    "metrics": ["iou", "dice", "pointing_game", "energy_ratio"],
}

# Perturbation and stability
PERTURBATION_CONFIG = {
    "num_perturbations": 5,
    "noise_stds": [0.01, 0.03, 0.05],
    "blur_radii": [1.0, 1.5, 2.0],
    "max_shift_px": 3,
    "brightness_range": (0.8, 1.2),
}

STABILITY_CONFIG = {
    "ssim_window_size": 7,
    "metrics": ["ssim", "spearman_correlation"],
}

# Reliability assessment
RELIABILITY_CONFIG = {
    "threshold": 0.70,
    "weights": {
        "confidence": 0.3,
        "localization": 0.4,
        "stability": 0.3,
    },
    "baseline_methods": ["confidence_only", "random"],
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "app_title": "Evidence-Aware Brain Tumour Classifier",
    "port": 8501,
    "theme": "light",
}
