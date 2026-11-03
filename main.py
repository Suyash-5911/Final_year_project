"""
Main pipeline for Evidence-Aware Brain Tumour Classification
Complete workflow from training to reliability assessment
"""
import torch
import numpy as np
from pathlib import Path
import argparse

from src.config import *
from src.data import DatasetLoader, Preprocessor, TrainTransform, ValTestTransform
from src.models import TumourClassifier, ModelTrainer
from src.explainability import ExplanationGenerator
from src.evaluation import LocalizationEvaluator, StabilityEvaluator, Perturbation
from src.reliability import ReliabilityAnalyzer


def train_classifier(args):
    """Train the tumour classifier"""
    print("=" * 60)
    print("PHASE 1: TRAINING CLASSIFIER")
    print("=" * 60)
    
    # Setup data
    loader = DatasetLoader(
        dataset_dir=args.data_dir,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15
    )
    
    # Load dataset
    print("\nLoading dataset...")
    loader.load_slices_and_masks()
    stats = loader.get_dataset_statistics()
    print(f"Dataset statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Train: {stats['train_samples']}")
    print(f"  Val: {stats['val_samples']}")
    print(f"  Test: {stats['test_samples']}")
    
    # Preprocessor
    preprocessor = Preprocessor(**PREPROCESS_CONFIG)
    
    # Data loaders
    train_loader = loader.get_data_loader(
        'train',
        batch_size=TRAIN_CONFIG['batch_size'],
        transform=TrainTransform(preprocessor),
        shuffle=True
    )
    
    val_loader = loader.get_data_loader(
        'val',
        batch_size=TRAIN_CONFIG['batch_size'],
        transform=ValTestTransform(preprocessor),
        shuffle=False
    )
    
    # Initialize model
    model = TumourClassifier(
        backbone=MODEL_CONFIG['backbone'],
        num_classes=MODEL_CONFIG['num_classes'],
        pretrained=MODEL_CONFIG['pretrained']
    )
    
    # Trainer
    trainer = ModelTrainer(
        model,
        learning_rate=TRAIN_CONFIG['learning_rate'],
        optimizer_type=TRAIN_CONFIG['optimizer']
    )
    
    # Training loop
    print("\nTraining model...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODELS_DIR / "best_classifier.pth"
            trainer.save_checkpoint(save_path, epoch, val_loss)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")
    return model, loader, preprocessor


def evaluate_reliability(args):
    """Evaluate reliability on test set"""
    print("\n" + "=" * 60)
    print("PHASE 2: RELIABILITY EVALUATION")
    print("=" * 60)
    
    # Load model
    model = TumourClassifier(**MODEL_CONFIG)
    checkpoint_path = MODELS_DIR / "best_classifier.pth"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from {checkpoint_path}")
    else:
        print("⚠ No trained model found. Please train first.")
        return
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Setup components
    explainer = ExplanationGenerator(model, colormap='jet', device=device)
    loc_evaluator = LocalizationEvaluator(**LOCALIZATION_CONFIG)
    stability_evaluator = StabilityEvaluator(**STABILITY_CONFIG)
    perturbation_engine = Perturbation(**PERTURBATION_CONFIG)
    reliability_analyzer = ReliabilityAnalyzer(**RELIABILITY_CONFIG)
    
    # Load test data
    loader = DatasetLoader(args.data_dir)
    loader.load_slices_and_masks()
    preprocessor = Preprocessor(**PREPROCESS_CONFIG)
    
    test_loader = loader.get_data_loader(
        'test',
        batch_size=1,
        transform=ValTestTransform(preprocessor),
        shuffle=False
    )
    
    print(f"\nEvaluating on {len(test_loader)} test samples...")
    
    all_reliability_scores = []
    all_correctness = []
    all_confidence_scores = []
    
    for idx, batch in enumerate(test_loader):
        image = batch['image'].to(device)
        mask = batch['mask'].cpu().numpy()[0, 0]
        true_label = batch['label'].item()
        
        # Predict
        with torch.no_grad():
            class_name, confidence, probs = model.predict(image)
            predicted_label = probs.argmax().item()
        
        # Correctness
        is_correct = (predicted_label == true_label)
        all_correctness.append(1 if is_correct else 0)
        all_confidence_scores.append(confidence)
        
        # Generate explanation
        cam = explainer.generate_cam(image, predicted_label)
        
        # Localization evaluation
        loc_metrics = loc_evaluator.evaluate_all(cam, mask)
        
        # Stability evaluation (perturbation)
        perturbed_images = perturbation_engine.generate_perturbation_suite(
            image[0],
            num_perturbations=PERTURBATION_CONFIG['num_perturbations']
        )
        
        perturbed_cams = []
        for p_img in perturbed_images:
            p_img_batch = p_img.unsqueeze(0).to(device)
            p_cam = explainer.generate_cam(p_img_batch, predicted_label)
            perturbed_cams.append(p_cam)
        
        stability_metrics = stability_evaluator.evaluate_stability(cam, perturbed_cams)
        
        # Reliability assessment
        rel_score, flag, breakdown = reliability_analyzer.assess_reliability(
            confidence, loc_metrics, stability_metrics
        )
        
        all_reliability_scores.append(rel_score)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(test_loader)} samples...")
    
    # Compute final metrics
    all_reliability_scores = np.array(all_reliability_scores)
    all_correctness = np.array(all_correctness)
    all_confidence_scores = np.array(all_confidence_scores)
    
    print("\n" + "=" * 60)
    print("RELIABILITY ASSESSMENT RESULTS")
    print("=" * 60)
    
    # AUROC/AUPRC
    auroc_auprc = reliability_analyzer.compute_auroc_auprc(
        all_correctness,
        all_reliability_scores
    )
    print(f"\nEvidence-Aware Metrics:")
    print(f"  AUROC: {auroc_auprc['auroc']:.4f}")
    print(f"  AUPRC: {auroc_auprc['auprc']:.4f}")
    
    # Baseline comparison
    baseline_metrics = reliability_analyzer.compute_auroc_auprc(
        all_correctness,
        all_confidence_scores
    )
    print(f"\nBaseline (Confidence-Only) Metrics:")
    print(f"  AUROC: {baseline_metrics['auroc']:.4f}")
    print(f"  AUPRC: {baseline_metrics['auprc']:.4f}")
    
    # Improvement
    auroc_improvement = auroc_auprc['auroc'] - baseline_metrics['auroc']
    auprc_improvement = auroc_auprc['auprc'] - baseline_metrics['auprc']
    print(f"\nImprovement:")
    print(f"  AUROC: {auroc_improvement:+.4f}")
    print(f"  AUPRC: {auprc_improvement:+.4f}")
    
    # Risk-coverage analysis
    risk_coverage = reliability_analyzer.risk_coverage_analysis(
        all_reliability_scores,
        all_correctness,
        coverage_levels=[0.5, 0.7, 0.8, 0.9]
    )
    
    print(f"\nRisk-Coverage Analysis:")
    for key, metrics in risk_coverage.items():
        print(f"  {key}: Acc={metrics['accuracy']:.2%}, Error={metrics['error_rate']:.2%}")
    
    # Summary
    summary = reliability_analyzer.get_audit_summary()
    print(f"\nAudit Summary:")
    print(f"  Total assessments: {summary['total_assessments']}")
    print(f"  Mean reliability: {summary['mean_reliability']:.3f}")
    print(f"  Reliable predictions: {summary['reliable_percentage']:.1f}%")
    
    print("\n✓ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Evidence-Aware Brain Tumour Classification System"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'both'],
        default='both',
        help='Mode: train, eval, or both'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/brain_tumours',
        help='Path to dataset directory'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("EVIDENCE-AWARE BRAIN TUMOUR CLASSIFICATION SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    if args.mode in ['train', 'both']:
        train_classifier(args)
    
    if args.mode in ['eval', 'both']:
        evaluate_reliability(args)
    
    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
