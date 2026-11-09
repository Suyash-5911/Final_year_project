"""
Evaluation metrics for brain tumor classification.

Provides comprehensive metrics including accuracy, precision, recall, F1-score,
confusion matrix analysis, and per-class performance.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculate and track classification metrics.
    
    Supports both binary and multi-class classification with comprehensive metrics.
    
    Args:
        num_classes: Number of classes
        class_names: Optional list of class names for readable output
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            if len(class_names) != num_classes:
                raise ValueError(f"Expected {num_classes} class names, got {len(class_names)}")
            self.class_names = class_names
        
        # Storage for predictions and labels
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics with new batch of predictions.
        
        Args:
            predictions: Predicted class indices (B,)
            labels: Ground truth labels (B,)
            probabilities: Optional class probabilities (B, num_classes)
        """
        # Convert to numpy
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        self.all_predictions.extend(predictions)
        self.all_labels.extend(labels)
        
        if probabilities is not None:
            probabilities = probabilities.detach().cpu().numpy()
            self.all_probabilities.extend(probabilities)
    
    def compute(self, average: str = 'macro') -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            average: Averaging strategy ('macro', 'micro', 'weighted')
        
        Returns:
            Dictionary of metric values
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average=average, zero_division=0),
            'recall': recall_score(labels, predictions, average=average, zero_division=0),
            'f1_score': f1_score(labels, predictions, average=average, zero_division=0),
        }
        
        # Add per-class metrics
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Add AUC metrics if probabilities are available
        if len(self.all_probabilities) > 0:
            probabilities = np.array(self.all_probabilities)
            
            # Convert labels to one-hot for multi-class AUC
            if self.num_classes > 2:
                from sklearn.preprocessing import label_binarize
                labels_binary = label_binarize(labels, classes=range(self.num_classes))
                
                try:
                    metrics['roc_auc_macro'] = roc_auc_score(
                        labels_binary, probabilities, average='macro', multi_class='ovr'
                    )
                    metrics['roc_auc_weighted'] = roc_auc_score(
                        labels_binary, probabilities, average='weighted', multi_class='ovr'
                    )
                except ValueError:
                    # Handle case where not all classes are present
                    pass
            else:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(labels, probabilities[:, 1])
        
        return metrics
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Formatted classification report string
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        report = classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            zero_division=0,
        )
        
        return report
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix of shape (num_classes, num_classes)
        """
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        cm = confusion_matrix(labels, predictions)
        
        return cm
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def print_metrics(self, average: str = 'macro'):
        """Print all metrics in a readable format."""
        metrics = self.compute(average=average)
        
        print("=" * 70)
        print("CLASSIFICATION METRICS")
        print("=" * 70)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f} ({average})")
        print(f"Recall:    {metrics['recall']:.4f} ({average})")
        print(f"F1-Score:  {metrics['f1_score']:.4f} ({average})")
        
        if 'roc_auc_macro' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc_macro']:.4f} (macro)")
        
        print("\n" + "-" * 70)
        print("PER-CLASS METRICS")
        print("-" * 70)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for class_name in self.class_names:
            precision = metrics[f'precision_{class_name}']
            recall = metrics[f'recall_{class_name}']
            f1 = metrics[f'f1_{class_name}']
            print(f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        print("=" * 70)


class ConfusionMatrixAnalyzer:
    """
    Analyze and visualize confusion matrices.
    
    Args:
        class_names: List of class names
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        normalize: bool = False,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            normalize: Whether to normalize values
            save_path: Path to save figure
            figsize: Figure size
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_confusion_matrix(self, cm: np.ndarray) -> Dict[str, any]:
        """
        Analyze confusion matrix and extract insights.
        
        Args:
            cm: Confusion matrix
        
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Total predictions
        total = cm.sum()
        analysis['total_samples'] = int(total)
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        analysis['per_class_accuracy'] = {
            self.class_names[i]: float(acc) for i, acc in enumerate(per_class_acc)
        }
        
        # Most confused pairs
        confused_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.class_names[i],
                        'predicted_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100),
                    })
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        analysis['most_confused_pairs'] = confused_pairs[:5]
        
        # Class-wise statistics
        class_stats = []
        for i, class_name in enumerate(self.class_names):
            true_positives = cm[i, i]
            false_positives = cm[:, i].sum() - true_positives
            false_negatives = cm[i, :].sum() - true_positives
            true_negatives = total - true_positives - false_positives - false_negatives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_stats.append({
                'class': class_name,
                'true_positives': int(true_positives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives),
                'true_negatives': int(true_negatives),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
            })
        
        analysis['class_statistics'] = class_stats
        
        return analysis
    
    def print_analysis(self, cm: np.ndarray):
        """Print confusion matrix analysis."""
        analysis = self.analyze_confusion_matrix(cm)
        
        print("=" * 70)
        print("CONFUSION MATRIX ANALYSIS")
        print("=" * 70)
        print(f"Total Samples: {analysis['total_samples']}")
        
        print("\n" + "-" * 70)
        print("PER-CLASS ACCURACY")
        print("-" * 70)
        for class_name, acc in analysis['per_class_accuracy'].items():
            print(f"{class_name:<15}: {acc:.2%}")
        
        print("\n" + "-" * 70)
        print("MOST CONFUSED PAIRS")
        print("-" * 70)
        for pair in analysis['most_confused_pairs']:
            print(f"{pair['true_class']:<12} → {pair['predicted_class']:<12}: "
                  f"{pair['count']:3d} samples ({pair['percentage']:5.2f}%)")
        
        print("\n" + "-" * 70)
        print("CLASS STATISTICS")
        print("-" * 70)
        print(f"{'Class':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 70)
        for stats in analysis['class_statistics']:
            print(f"{stats['class']:<12} "
                  f"{stats['true_positives']:<6} "
                  f"{stats['false_positives']:<6} "
                  f"{stats['false_negatives']:<6} "
                  f"{stats['precision']:<10.4f} "
                  f"{stats['recall']:<10.4f} "
                  f"{stats['f1_score']:<10.4f}")
        
        print("=" * 70)


def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate metrics for a single batch or dataset.
    
    Convenience function for one-time metric calculation.
    
    Args:
        predictions: Predicted class indices
        labels: Ground truth labels
        probabilities: Optional class probabilities
        class_names: Optional class names
    
    Returns:
        Dictionary of metrics
    """
    num_classes = int(predictions.max()) + 1
    
    calculator = MetricsCalculator(num_classes, class_names)
    calculator.update(predictions, labels, probabilities)
    
    return calculator.compute()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[str] = None,
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    else:
        plt.show()
    
    plt.close()
