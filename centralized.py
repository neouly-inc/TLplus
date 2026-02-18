"""Centralized training baseline for TL++.

This script provides a fair comparison baseline by training the same CNN
architecture as TL++ in standard centralized mode (no distribution, no MPC).
Uses identical hyperparameters and training procedures.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm
from typing import Tuple


# ==============================================================================
# CONSTANTS
# ==============================================================================

# CIFAR-10 normalization statistics (same as TL++)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# Default hyperparameters (matching TL++ orchestrator defaults)
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 256
DEFAULT_EPOCHS = 1000
DEFAULT_PATIENCE = 50
DEFAULT_LR = 0.1
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_GRAD_CLIP_NORM = 1.0


# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

# Korean Standard Time (UTC+9)
KST = timezone(timedelta(hours=9))


class KSTFormatter(logging.Formatter):
    """Custom log formatter using Korean Standard Time."""
    
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=KST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()


def setup_logging(log_dir: str = 'logs') -> Path:
    """Configure logging to file and console.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Path to created log file
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'centralized_{timestamp}.log'
    
    formatter = KSTFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S KST'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("=" * 80)
    logging.info("TL++ Centralized Baseline")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")
    
    return log_file


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class CNN(nn.Module):
    """CNN for CIFAR-10 (identical to TL++ orchestrator model).
    
    Architecture matches CNNOrchestrator from models.py:
    - Block 1: Conv(3→64)→ReLU→Conv(64→64)→ReLU→MaxPool
    - Block 2: Conv(64→128)→ReLU→Conv(128→128)→ReLU→MaxPool
    - Block 3: Conv(128→256)→ReLU→Conv(256→256)→ReLU→MaxPool
    - Flatten → FC(256×4×4→512)→ReLU→Dropout(0.5)
    - FC(512→num_classes)
    """
    
    def __init__(self, num_classes: int = 10):
        """Initialize CNN.
        
        Args:
            num_classes: Number of output classes
        """
        super(CNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Block 1: 3 -> 64 -> 64 (spatial: 32x32 -> 16x16)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 64 -> 128 -> 128 (spatial: 16x16 -> 8x8)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 128 -> 256 -> 256 (spatial: 8x8 -> 4x4)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights (same as TL++)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, 3, 32, 32]
        
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Block 1
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # Block 2
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        # Block 3
        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = self.pool3(x)
        
        # Classifier
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data(num_classes: int = 10,
              train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
              test_batch_size: int = DEFAULT_TEST_BATCH_SIZE,
              data_dir: str = './data') -> Tuple:
    """Load CIFAR-10 training and test data.
    
    Uses same transforms and normalization as TL++.
    
    Args:
        num_classes: Number of classes to include (1-10)
        train_batch_size: Training batch size
        test_batch_size: Test batch size
        data_dir: Directory to store/load data
    
    Returns:
        (train_loader, test_loader)
    """
    logging.info("-" * 80)
    logging.info("DATA LOADING")
    logging.info("-" * 80)
    
    # Training transform (with augmentation, same as NodeDataLoader)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Test transform (no augmentation, same as OrchestratorDataLoader)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Load full datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Filter by number of classes if needed (matching TL++ behavior)
    if num_classes < 10:
        logging.info(f"Filtering to {num_classes} classes (0-{num_classes-1})")
        train_indices = [i for i, (_, label) in enumerate(train_dataset)
                        if label < num_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset)
                       if label < num_classes]
        
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logging.info(f"Training samples:   {len(train_dataset):,}")
    logging.info(f"Test samples:       {len(test_dataset):,}")
    logging.info(f"Training batches:   {len(train_loader)}")
    logging.info(f"Test batches:       {len(test_loader)}")
    logging.info("")
    
    return train_loader, test_loader


# ==============================================================================
# TRAINING
# ==============================================================================

def train_epoch(model: nn.Module,
                train_loader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                grad_clip_norm: float,
                epoch: int,
                total_epochs: int) -> float:
    """Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Computation device
        grad_clip_norm: Gradient clipping threshold (0 to disable)
        epoch: Current epoch number
        total_epochs: Total number of epochs
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}',
               ncols=100, leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (same as TL++)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss


def evaluate(model: nn.Module,
            test_loader,
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, float, dict]:
    """Evaluate model on test set.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Computation device
    
    Returns:
        (average_loss, accuracy_percentage, info_dict)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().tolist())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    unique_preds = len(set(all_predictions))
    
    info = {
        'unique_predictions': unique_preds,
        'total_samples': total,
        'correct': correct
    }
    
    return avg_loss, accuracy, info


def train(model: nn.Module,
         train_loader,
         test_loader,
         criterion: nn.Module,
         optimizer: optim.Optimizer,
         scheduler,
         device: torch.device,
         epochs: int,
         patience: int,
         grad_clip_norm: float) -> None:
    """Main training loop with early stopping.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Computation device
        epochs: Maximum number of epochs
        patience: Early stopping patience
        grad_clip_norm: Gradient clipping threshold
    """
    logging.info("-" * 80)
    logging.info("TRAINING")
    logging.info("-" * 80)
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info("")
        logging.info(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f}")
        
        # Train one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, grad_clip_norm, epoch, epochs
        )
        
        # Evaluate
        test_loss, test_acc, info = evaluate(model, test_loader, criterion, device)
        
        logging.info(
            f"  Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}%"
        )
        
        # Check for model collapse
        if info['unique_predictions'] == 1:
            logging.warning("  ⚠️ Model predicting only 1 class - possible collapse")
        
        # Update learning rate
        scheduler.step()
        
        # Track best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            logging.info(f"  🏆 New best accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
            logging.info(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            logging.info("")
            logging.info(f"Early stopping after {patience} epochs without improvement")
            break
    
    # Training complete
    logging.info("")
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Best accuracy:  {best_acc:.2f}% (Epoch {best_epoch})")
    logging.info(f"Final accuracy: {test_acc:.2f}%")
    logging.info("=" * 80)


# ==============================================================================
# CONFIGURATION AND SETUP
# ==============================================================================

def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
    
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('scheduler', 'cosine')
    epochs = config.get('epochs', DEFAULT_EPOCHS)
    
    if scheduler_type == 'cosine':
        t_max = config.get('t_max') or epochs
        eta_min = config.get('eta_min', 1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 50)
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'multistep':
        milestones = config.get('milestones', [60, 120, 160])
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def get_device(no_accel: bool) -> torch.device:
    """Determine computation device.
    
    Args:
        no_accel: If True, force CPU usage
    
    Returns:
        PyTorch device
    """
    if no_accel:
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TL++ Centralized Training Baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes to train on')
    
    # Training configuration
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE,
                       help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE,
                       help='Test batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                       help='Early stopping patience')
    
    # Optimizer
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=DEFAULT_MOMENTUM,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                       help='Use Nesterov momentum')
    parser.add_argument('--no_nesterov', action='store_false', dest='nesterov',
                       help='Disable Nesterov momentum')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'multistep'],
                       help='Learning rate scheduler')
    parser.add_argument('--t_max', type=int, default=None,
                       help='Cosine scheduler T_max')
    parser.add_argument('--eta_min', type=float, default=1e-6,
                       help='Cosine scheduler minimum LR')
    parser.add_argument('--step_size', type=int, default=50,
                       help='Step scheduler step size')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Scheduler decay factor')
    parser.add_argument('--milestones', type=int, nargs='+', default=[60, 120, 160],
                       help='MultiStep scheduler milestones')
    
    # Regularization
    parser.add_argument('--grad_clip_norm', type=float, default=DEFAULT_GRAD_CLIP_NORM,
                       help='Gradient clipping norm (0 to disable)')
    
    # Hardware and paths
    parser.add_argument('--no_accel', action='store_true',
                       help='Use CPU only')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for CIFAR-10 data')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for log files')
    
    args = parser.parse_args()
    
    if not 1 <= args.num_classes <= 10:
        parser.error('num_classes must be between 1 and 10')
    
    return vars(args)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point."""
    config = parse_args()
    
    # Setup logging
    log_file = setup_logging(config['log_dir'])
    logging.info("")
    
    # Determine device
    device = get_device(config['no_accel'])
    logging.info(f"Device: {device}")
    logging.info("")
    
    # Log configuration
    logging.info("-" * 80)
    logging.info("CONFIGURATION")
    logging.info("-" * 80)
    logging.info(f"Classes:          {config['num_classes']}")
    logging.info(f"Batch size:       {config['train_batch_size']}")
    logging.info(f"Learning rate:    {config['lr']}")
    logging.info(f"Scheduler:        {config['scheduler']}")
    logging.info("")
    
    # Load data
    train_loader, test_loader = load_data(
        num_classes=config['num_classes'],
        train_batch_size=config['train_batch_size'],
        test_batch_size=config['test_batch_size'],
        data_dir=config['data_dir']
    )
    
    # Create model
    logging.info("-" * 80)
    logging.info("MODEL INITIALIZATION")
    logging.info("-" * 80)
    model = CNN(num_classes=config['num_classes']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters:     {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info("")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        nesterov=config['nesterov']
    )
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    logging.info("-" * 80)
    logging.info("OPTIMIZER & SCHEDULER")
    logging.info("-" * 80)
    logging.info(f"Optimizer:        SGD (lr={config['lr']}, momentum={config['momentum']})")
    logging.info(f"Scheduler:        {config['scheduler']}")
    logging.info(f"Gradient clip:    {config['grad_clip_norm']}")
    logging.info("")
    
    # Train
    try:
        train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=config['epochs'],
            patience=config['patience'],
            grad_clip_norm=config['grad_clip_norm']
        )
    except KeyboardInterrupt:
        logging.info("")
        logging.info("=" * 80)
        logging.info("Training interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error(f"Training failed: {e}")
        logging.error("=" * 80)
        import traceback
        logging.error(traceback.format_exc())
    finally:
        logging.info("")
        logging.info(f"Full log: {log_file}")
        logging.info("=" * 80)


if __name__ == '__main__':
    main()