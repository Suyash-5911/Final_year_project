"""
Main training script for brain tumor classification.

Usage:
    python train.py --epochs 50 --batch-size 32 --lr 1e-4
    python train.py --backbone resnet50 --freeze-backbone --unfreeze-after 10
    python train.py --augmentation heavy --use-amp
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import DatasetLoader, Preprocessor, TrainTransform, ValTestTransform
from src.models import TumourClassifier, ModelTrainer
from src.models.model_trainer import create_optimizer, create_scheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classifier')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/brisc/brisc2025',
                        help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--include-no-tumor', action='store_true',
                        help='Include no_tumor class (4-class problem)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone layers initially')
    parser.add_argument('--unfreeze-after', type=int, default=None,
                        help='Unfreeze backbone after N epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--early-stopping', type=int, default=None,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--augmentation', type=str, default='medium',
                        choices=['light', 'medium', 'heavy'],
                        help='Data augmentation level')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='models/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device setup
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("\n" + "=" * 70)
    print("BRAIN TUMOR CLASSIFICATION - TRAINING")
    print("=" * 70)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:        {args.data_dir}")
    print(f"  Backbone:       {args.backbone}")
    print(f"  Pretrained:     {args.pretrained}")
    print(f"  Freeze:         {args.freeze_backbone}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Learning Rate:  {args.lr}")
    print(f"  Optimizer:      {args.optimizer}")
    print(f"  Scheduler:      {args.scheduler}")
    print(f"  Augmentation:   {args.augmentation}")
    print(f"  Device:         {args.device}")
    print(f"  Mixed Precision:{args.use_amp}")
    print("=" * 70)
    
    # Determine number of classes
    num_classes = 4 if args.include_no_tumor else 3
    class_names = ['glioma', 'meningioma', 'pituitary']
    if args.include_no_tumor:
        class_names.append('no_tumor')
    
    # Create data loaders
    print("\nLoading dataset...")
    loader = DatasetLoader(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_no_tumor=args.include_no_tumor,
    )
    
    # Print dataset info
    loader.print_dataset_info()
    
    # Create transforms
    preprocessor = Preprocessor(image_size=args.image_size)
    
    # Configure augmentation based on level
    aug_config = {
        'none': {'hflip_prob': 0, 'rotation_degrees': 0, 'color_jitter': False},
        'light': {'hflip_prob': 0.3, 'rotation_degrees': 10, 'color_jitter': False},
        'medium': {'hflip_prob': 0.5, 'rotation_degrees': 15, 'color_jitter': True, 'brightness': 0.2, 'contrast': 0.2},
        'heavy': {'hflip_prob': 0.7, 'rotation_degrees': 20, 'color_jitter': True, 'brightness': 0.3, 'contrast': 0.3, 'blur_prob': 0.3},
    }
    
    train_transform = TrainTransform(preprocessor, **aug_config.get(args.augmentation, aug_config['medium']))
    val_transform = ValTestTransform(preprocessor)
    
    # Get data loaders
    train_loader = loader.get_classification_loader('train', train_transform)
    val_loader = loader.get_classification_loader('test', val_transform)
    
    print(f"\nCreated DataLoaders:")
    print(f"  Training batches:   {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = TumourClassifier(
        backbone=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        scheduler=scheduler,
        class_names=class_names,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=args.use_amp,
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    
    # Progressive unfreezing callback
    if args.freeze_backbone and args.unfreeze_after is not None:
        print(f"\nNote: Backbone will be unfrozen after epoch {args.unfreeze_after}")
    
    # Custom training loop with progressive unfreezing
    if args.freeze_backbone and args.unfreeze_after is not None:
        # Train with frozen backbone first
        print(f"\nPhase 1: Training with frozen backbone ({args.unfreeze_after} epochs)")
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.unfreeze_after,
            early_stopping_patience=args.early_stopping,
            save_best=True,
            save_last=False,
        )
        
        # Unfreeze and continue training
        print(f"\nPhase 2: Fine-tuning with unfrozen backbone")
        model.unfreeze_backbone_layers()
        
        # Optionally reduce learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1
        print(f"Reduced learning rate to {args.lr * 0.1:.6f} for fine-tuning")
        
        # Continue training
        remaining_epochs = args.epochs - args.unfreeze_after
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=remaining_epochs,
            early_stopping_patience=args.early_stopping,
            save_best=True,
            save_last=True,
        )
    else:
        # Standard training
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping,
            save_best=True,
            save_last=True,
        )
    
    # Plot training curves
    print("\nGenerating training curves...")
    trainer.plot_training_curves()
    
    # Final evaluation on validation set
    print("\nFinal Evaluation on Validation Set:")
    val_loss, val_acc, metrics_calc = trainer.validate(val_loader, verbose=False)
    
    print("\nClassification Report:")
    print(metrics_calc.get_classification_report())
    
    print("\nConfusion Matrix:")
    from src.models.metrics import ConfusionMatrixAnalyzer
    cm_analyzer = ConfusionMatrixAnalyzer(class_names)
    cm = metrics_calc.get_confusion_matrix()
    cm_analyzer.print_analysis(cm)
    
    # Save confusion matrix plot
    cm_path = Path(args.checkpoint_dir) / 'confusion_matrix.png'
    cm_analyzer.plot_confusion_matrix(cm, normalize=True, save_path=str(cm_path))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Model: {Path(args.checkpoint_dir) / 'best_model.pth'}")
    print(f"Best Val Acc: {trainer.best_val_acc:.2f}% (Epoch {trainer.best_epoch})")
    print(f"Training Curves: {Path(args.checkpoint_dir) / 'training_curves.png'}")
    print(f"Confusion Matrix: {cm_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
