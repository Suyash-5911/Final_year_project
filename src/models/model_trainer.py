"""
Training pipeline for brain tumor classification models.

Provides comprehensive training loop with validation, checkpointing, early stopping,
learning rate scheduling, and progress tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path
import time
from tqdm import tqdm
import json

from .metrics import MetricsCalculator


class ModelTrainer:
    """
    Comprehensive model trainer with advanced features.
    
    Features:
    - Training and validation loops
    - Automatic checkpointing
    - Early stopping
    - Learning rate scheduling
    - Progress tracking
    - Metrics logging
    - Mixed precision training support
    
    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        class_names: Optional class names for metrics
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        scheduler: Optional[_LRScheduler] = None,
        class_names: Optional[List[str]] = None,
        checkpoint_dir: str = 'models/checkpoints',
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.class_names = class_names or ['Class 0', 'Class 1', 'Class 2']
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.patience_counter = 0
        
        print(f"Trainer initialized on device: {device}")
        if use_amp:
            print("Using Automatic Mixed Precision (AMP)")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            verbose: Whether to show progress bar
        
        Returns:
            Average loss and accuracy for the epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Progress bar
        if verbose:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        else:
            pbar = train_loader
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            batch_size = images.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward and backward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size
            
            # Update progress bar
            if verbose:
                current_loss = running_loss / total_samples
                current_acc = 100.0 * running_corrects / total_samples
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * running_corrects / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(
        self,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Tuple[float, float, MetricsCalculator]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            verbose: Whether to show progress bar
        
        Returns:
            Average loss, accuracy, and metrics calculator
        """
        self.model.eval()
        
        running_loss = 0.0
        total_samples = 0
        
        # Metrics calculator
        num_classes = len(self.class_names)
        metrics_calc = MetricsCalculator(num_classes, self.class_names)
        
        # Progress bar
        if verbose:
            pbar = tqdm(val_loader, desc="Validating")
        else:
            pbar = val_loader
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                batch_size = images.size(0)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Update metrics
                metrics_calc.update(preds, labels, probabilities)
                
                # Statistics
                running_loss += loss.item() * batch_size
                total_samples += batch_size
        
        epoch_loss = running_loss / total_samples
        
        # Compute metrics
        metrics = metrics_calc.compute()
        epoch_acc = metrics['accuracy'] * 100.0
        
        return epoch_loss, epoch_acc, metrics_calc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best: bool = True,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs (None to disable)
            save_best: Whether to save the best model
            save_last: Whether to save the last model
            verbose: Whether to print progress
        """
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Epochs: {epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch, verbose)
            
            # Validate
            val_loss, val_acc, metrics_calc = self.validate(val_loader, verbose)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            if verbose:
                print(f"\nEpoch {epoch}/{epochs} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
                print(f"  Learning Rate: {current_lr:.6f}")
                print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(
                        epoch=epoch,
                        val_acc=val_acc,
                        filename='best_model.pth'
                    )
                
                if verbose:
                    print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if early_stopping_patience is not None:
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
                    break
            
            print("-" * 70)
        
        # Save last model
        if save_last:
            self.save_checkpoint(
                epoch=epochs,
                val_acc=val_acc,
                filename='last_model.pth'
            )
        
        # Training complete
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total Time: {total_time / 60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print("=" * 70)
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        filename: str = 'checkpoint.pth',
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Validation accuracy
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_acc': val_acc,
            'history': self.history,
            'config': {
                'backbone': getattr(self.model, 'backbone_name', 'unknown'),
                'num_classes': getattr(self.model, 'num_classes', 3),
                'dropout': 0.5,  # Default value
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if filename == 'best_model.pth':
            print(f"  ✓ Saved best model to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Epoch: {epoch}, Best Val Acc: {self.best_val_acc:.2f}%")
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / filename
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Saved training history to: {history_path}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        from .metrics import plot_training_curves
        
        if save_path is None:
            save_path = str(self.checkpoint_dir / 'training_curves.png')
        
        plot_training_curves(
            self.history['train_loss'],
            self.history['val_loss'],
            self.history['train_acc'],
            self.history['val_acc'],
            save_path=save_path,
        )


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Factory function to create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: Optimizer type ('adam', 'sgd', 'adamw')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer arguments
    
    Returns:
        PyTorch optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=kwargs.get('nesterov', True),
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"Created {optimizer_name.upper()} optimizer (lr={learning_rate}, wd={weight_decay})")
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    epochs: int = 50,
    **kwargs
) -> Optional[_LRScheduler]:
    """
    Factory function to create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Scheduler type ('cosine', 'step', 'plateau', 'none')
        epochs: Total number of epochs
        **kwargs: Additional scheduler arguments
    
    Returns:
        PyTorch scheduler or None
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=kwargs.get('eta_min', 1e-6),
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1),
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5),
            verbose=True,
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    if scheduler is not None:
        print(f"Created {scheduler_name.upper()} learning rate scheduler")
    
    return scheduler
