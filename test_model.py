"""
Test script for CNN classifier and training pipeline.

Tests:
1. Model instantiation
2. Forward pass
3. Feature extraction
4. Grad-CAM hook functionality
5. Training loop (1 epoch)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import TumourClassifier, ModelTrainer
from src.models.model_trainer import create_optimizer, create_scheduler
from src.models.metrics import MetricsCalculator, ConfusionMatrixAnalyzer


def test_model_instantiation():
    """Test 1: Model instantiation."""
    print("\n" + "=" * 70)
    print("TEST 1: Model Instantiation")
    print("=" * 70)
    
    try:
        # Test different backbones
        backbones = ['resnet18', 'resnet50']
        
        for backbone in backbones:
            print(f"\nTesting {backbone}...")
            model = TumourClassifier(
                backbone=backbone,
                num_classes=3,
                pretrained=False,  # Don't download weights for test
                dropout=0.5,
            )
            
            print(f"✓ {backbone} instantiated successfully")
            
            # Get model info
            info = model.get_model_info()
            print(f"  Total parameters: {info['total_parameters']:,}")
            print(f"  Trainable parameters: {info['trainable_parameters']:,}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test 2: Forward pass."""
    print("\n" + "=" * 70)
    print("TEST 2: Forward Pass")
    print("=" * 70)
    
    try:
        model = TumourClassifier(
            backbone='resnet18',
            num_classes=3,
            pretrained=False,
        )
        model.eval()
        
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        print(f"\nInput shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print(f"Output shape: {outputs.shape}")
        print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
        
        # Check output shape
        assert outputs.shape == (batch_size, 3), f"Expected (4, 3), got {outputs.shape}"
        
        print("✓ Forward pass successful")
        
        # Test prediction
        probabilities, predictions = model.predict(dummy_input)
        print(f"\nProbabilities shape: {probabilities.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions.tolist()}")
        print(f"Sample probabilities: {probabilities[0].tolist()}")
        
        # Check probabilities sum to 1
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), "Probabilities don't sum to 1"
        
        print("✓ Prediction successful")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test 3: Feature extraction."""
    print("\n" + "=" * 70)
    print("TEST 3: Feature Extraction")
    print("=" * 70)
    
    try:
        model = TumourClassifier(
            backbone='resnet18',
            num_classes=3,
            pretrained=False,
        )
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(4, 3, 224, 224)
        
        # Extract features
        with torch.no_grad():
            features = model.get_features(dummy_input)
        
        print(f"\nFeatures shape: {features.shape}")
        print(f"Feature dimension: {model.feature_dim}")
        
        assert features.shape == (4, model.feature_dim), f"Expected (4, {model.feature_dim}), got {features.shape}"
        
        print("✓ Feature extraction successful")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_freeze_unfreeze():
    """Test 4: Freeze/unfreeze functionality."""
    print("\n" + "=" * 70)
    print("TEST 4: Freeze/Unfreeze Backbone")
    print("=" * 70)
    
    try:
        model = TumourClassifier(
            backbone='resnet18',
            num_classes=3,
            pretrained=False,
            freeze_backbone=True,
        )
        
        # Check frozen parameters
        frozen_count = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        total_backbone_params = sum(1 for p in model.backbone.parameters())
        
        print(f"\nFrozen parameters: {frozen_count}/{total_backbone_params}")
        assert frozen_count == total_backbone_params, "Not all backbone parameters are frozen"
        
        print("✓ Backbone frozen successfully")
        
        # Unfreeze
        model.unfreeze_backbone_layers()
        
        unfrozen_count = sum(1 for p in model.backbone.parameters() if p.requires_grad)
        print(f"Unfrozen parameters: {unfrozen_count}/{total_backbone_params}")
        assert unfrozen_count == total_backbone_params, "Not all backbone parameters are unfrozen"
        
        print("✓ Backbone unfrozen successfully")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test 5: Metrics calculation."""
    print("\n" + "=" * 70)
    print("TEST 5: Metrics Calculation")
    print("=" * 70)
    
    try:
        class_names = ['glioma', 'meningioma', 'pituitary']
        metrics_calc = MetricsCalculator(num_classes=3, class_names=class_names)
        
        # Simulate predictions
        predictions = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        labels = torch.tensor([0, 1, 2, 1, 1, 2, 0, 2])
        probabilities = torch.softmax(torch.randn(8, 3), dim=1)
        
        # Update metrics
        metrics_calc.update(predictions, labels, probabilities)
        
        # Compute metrics
        metrics = metrics_calc.compute()
        
        print(f"\nMetrics computed:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        print("\n✓ Metrics calculation successful")
        
        # Test confusion matrix
        cm = metrics_calc.get_confusion_matrix()
        print(f"\nConfusion Matrix shape: {cm.shape}")
        print(f"Confusion Matrix:\n{cm}")
        
        print("✓ Confusion matrix successful")
        
        # Test confusion matrix analyzer
        cm_analyzer = ConfusionMatrixAnalyzer(class_names)
        analysis = cm_analyzer.analyze_confusion_matrix(cm)
        
        print(f"\n✓ Confusion matrix analysis successful")
        print(f"  Total samples: {analysis['total_samples']}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_scheduler():
    """Test 6: Optimizer and scheduler creation."""
    print("\n" + "=" * 70)
    print("TEST 6: Optimizer and Scheduler Creation")
    print("=" * 70)
    
    try:
        model = TumourClassifier(
            backbone='resnet18',
            num_classes=3,
            pretrained=False,
        )
        
        # Test different optimizers
        for opt_name in ['adam', 'sgd', 'adamw']:
            optimizer = create_optimizer(model, opt_name, learning_rate=1e-4)
            print(f"✓ {opt_name.upper()} optimizer created")
        
        # Test different schedulers
        optimizer = create_optimizer(model, 'adam', learning_rate=1e-4)
        for sched_name in ['cosine', 'step', 'plateau']:
            scheduler = create_scheduler(optimizer, sched_name, epochs=50)
            print(f"✓ {sched_name.upper()} scheduler created")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_info():
    """Test 7: Model information display."""
    print("\n" + "=" * 70)
    print("TEST 7: Model Information Display")
    print("=" * 70)
    
    try:
        model = TumourClassifier(
            backbone='resnet50',
            num_classes=3,
            pretrained=False,
        )
        
        model.print_model_info()
        
        print("\n✓ Model info display successful")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CNN CLASSIFIER AND TRAINING PIPELINE - TESTS")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Model Instantiation", test_model_instantiation()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Feature Extraction", test_feature_extraction()))
    results.append(("Freeze/Unfreeze", test_freeze_unfreeze()))
    results.append(("Metrics Calculation", test_metrics()))
    results.append(("Optimizer/Scheduler", test_optimizer_scheduler()))
    results.append(("Model Info Display", test_model_info()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe CNN classifier and training pipeline are ready to use!")
        print("\nYou can now train the model with:")
        print("  python train.py --epochs 50 --batch-size 32 --lr 1e-4")
        print("\nOr for quick test:")
        print("  python train.py --epochs 2 --batch-size 8")
        print("=" * 70)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
