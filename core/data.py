"""Data loaders for TL++ on CIFAR-10."""

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Tuple


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Standard normalization statistics for CIFAR-10 dataset
# These values are computed from the entire CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


# ==============================================================================
# NODE DATA LOADER
# ==============================================================================


class NodeDataLoader:
    """Data loader for node with class-specific CIFAR-10 training data.
    
    Each node is assigned a specific CIFAR-10 class and loads only
    training samples from that class. This simulates a traversal
    learning scenario where each node has data from a specific distribution.
    """
    
    def __init__(self, class_id: int, augment: bool = True, data_dir: str = './data'):
        """Initialize node data loader.
        
        Args:
            class_id: CIFAR-10 class to load (1-10, will be converted to 0-9 internally)
            augment: Whether to apply data augmentation (random crop and flip)
            data_dir: Directory to store/load CIFAR-10 data
        
        Raises:
            TypeError: If class_id is not an integer
            ValueError: If class_id is not in range 1-10
        """
        # Validate input types
        if not isinstance(class_id, int):
            raise TypeError(f"class_id must be int, got {type(class_id).__name__}")
        
        # Validate class_id is in valid range
        if not 1 <= class_id <= 10:
            raise ValueError(f"class_id must be 1-10, got {class_id}")
        
        # Store configuration
        self.class_id = class_id
        self.cifar_label = class_id - 1  # Convert from 1-indexed to 0-indexed
        self.augment = augment
        self.data_dir = data_dir
        
        # Build data transformation pipeline
        transform_ops = []
        
        # Add augmentation transforms if enabled (for training robustness)
        if augment:
            transform_ops.extend([
                transforms.RandomCrop(32, padding=4),  # Random crop with padding
                transforms.RandomHorizontalFlip(),      # Random horizontal flip
            ])
        
        # Add standard preprocessing transforms (always applied)
        transform_ops.extend([
            transforms.ToTensor(),                      # Convert PIL Image to tensor
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)  # Normalize using dataset statistics
        ])
        transform = transforms.Compose(transform_ops)
        
        # Load the full CIFAR-10 training dataset
        full_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,           # Use training split
            transform=transform,
            download=True         # Download if not present
        )
        
        # Filter dataset to only include samples from our assigned class
        # This creates a list of indices where the label matches our class
        indices = [i for i, (_, label) in enumerate(full_dataset)
                  if label == self.cifar_label]
        
        # Create a subset containing only our class samples
        self.dataset = torch.utils.data.Subset(full_dataset, indices)
        self.n_samples = len(self.dataset)
    
    def __len__(self) -> int:
        """Return number of samples in this node's dataset.
        
        Returns:
            Number of samples
        """
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index (0 to len-1)
            
        Returns:
            Tuple of (image tensor [C, H, W], label)
        """
        return self.dataset[idx]
    
    def __repr__(self) -> str:
        """Return string representation for debugging.
        
        Returns:
            String representation of the data loader
        """
        return (f"NodeDataLoader(class_id={self.class_id}, "
                f"n_samples={self.n_samples}, augment={self.augment})")


# ==============================================================================
# ORCHESTRATOR DATA LOADER
# ==============================================================================


class OrchestratorDataLoader:
    """Data loader for orchestrator with CIFAR-10 test set.
    
    The orchestrator uses the test set for evaluation. It can optionally
    filter to only include classes that are represented in the training
    nodes (for fair evaluation when using fewer than 10 classes).
    """
    
    def __init__(self, 
                 batch_size: int = 100,
                 num_classes: int = 10,
                 data_dir: str = './data'):
        """Initialize orchestrator data loader.
        
        Args:
            batch_size: Batch size for evaluation
            num_classes: Number of classes to include (1-10). If less than 10,
                        only the first num_classes will be included in the test set.
            data_dir: Directory to store/load CIFAR-10 data
        
        Raises:
            TypeError: If batch_size or num_classes is not an integer
            ValueError: If num_classes is not in range 1-10 or batch_size < 1
        """
        # Validate input types
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be int, got {type(batch_size).__name__}")
        if not isinstance(num_classes, int):
            raise TypeError(f"num_classes must be int, got {type(num_classes).__name__}")
        
        # Validate values
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 1 <= num_classes <= 10:
            raise ValueError(f"num_classes must be 1-10, got {num_classes}")
        
        # Store configuration
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.data_dir = data_dir
        
        # Create test transform (no augmentation for consistent evaluation)
        transform = transforms.Compose([
            transforms.ToTensor(),                      # Convert PIL Image to tensor
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)  # Normalize using dataset statistics
        ])
        
        # Load the full CIFAR-10 test dataset
        full_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,          # Use test split
            transform=transform,
            download=True         # Download if not present
        )
        
        # Filter by number of classes if needed
        # This ensures fair evaluation when training with fewer than 10 classes
        if num_classes < 10:
            # Keep only samples from classes 0 to num_classes-1
            indices = [i for i, (_, label) in enumerate(full_dataset)
                      if label < num_classes]
            dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            # Use full test set
            dataset = full_dataset
        
        # Store dataset and sample count
        self.dataset = dataset
        self.n_samples = len(dataset)
        
        # Create PyTorch DataLoader for batched iteration
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,        # Don't shuffle for consistent evaluation
            num_workers=2,        # Use 2 worker processes for data loading
            pin_memory=True       # Pin memory for faster GPU transfer
        )
    
    def __len__(self) -> int:
        """Return number of samples in test set.
        
        Returns:
            Number of samples
        """
        return self.n_samples
    
    def __iter__(self):
        """Return dataloader iterator for batched iteration.
        
        Returns:
            Iterator yielding (batch_images, batch_labels) tuples
        """
        return iter(self.loader)
    
    def __repr__(self) -> str:
        """Return string representation for debugging.
        
        Returns:
            String representation of the data loader
        """
        return (f"OrchestratorDataLoader(batch_size={self.batch_size}, "
                f"n_samples={self.n_samples}, num_classes={self.num_classes})")
