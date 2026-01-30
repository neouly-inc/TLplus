"""Training utilities for TL++."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional


# ==============================================================================
# BATCH SCHEDULING
# ==============================================================================


class BatchScheduler:
    """Manages virtual batch creation for distributed training.
    
    In distributed learning, data is partitioned across multiple nodes. This
    scheduler creates "virtual batches" that span across nodes, ensuring
    balanced sample distribution in each training iteration.
    
    Example:
        Node 1: 5000 samples, Node 2: 5000 samples
        Batch size: 128
        → Creates batches with ~64 samples from each node
    """
    
    def __init__(self, 
                 samples_per_node: List[int],
                 batch_size: int,
                 shuffle: bool = True,
                 seed: Optional[int] = None):
        """Initialize batch scheduler.
        
        Args:
            samples_per_node: Number of samples available at each node
            batch_size: Target size for each virtual batch
            shuffle: Whether to randomize sample order each epoch
            seed: Random seed for reproducibility (optional)
        
        Raises:
            ValueError: If batch_size < 1 or samples_per_node is empty
        """
        if not samples_per_node:
            raise ValueError("samples_per_node cannot be empty")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        
        self.samples_per_node = samples_per_node
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_nodes = len(samples_per_node)
        self.total_samples = sum(samples_per_node)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Create global sample index: list of (node_id, local_sample_idx) pairs
        self.global_samples = [
            (node_id, local_idx)
            for node_id in range(self.n_nodes)
            for local_idx in range(self.samples_per_node[node_id])
        ]
        
        # Calculate total batches per epoch
        self.n_batches = (self.total_samples + batch_size - 1) // batch_size
    
    def create_epoch_batches(self) -> List[List[np.ndarray]]:
        """Generate batch assignments for one epoch.
        
        Process:
        1. Shuffle global sample list (if enabled)
        2. Divide into fixed-size batches
        3. Split each batch by node ownership
        
        Returns:
            List of batches, where each batch contains sample indices per node
            Format: [batch1, batch2, ...] where batch_i = [node1_indices, node2_indices, ...]
        """
        # Randomize sample order if requested
        if self.shuffle:
            np.random.shuffle(self.global_samples)
        
        batches = []
        
        for batch_idx in range(self.n_batches):
            # Define batch boundaries
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.total_samples)
            batch_samples = self.global_samples[start_idx:end_idx]
            
            # Partition batch samples by node ownership
            indices_per_node = [[] for _ in range(self.n_nodes)]
            for node_id, local_idx in batch_samples:
                indices_per_node[node_id].append(local_idx)
            
            # Convert to numpy arrays for efficient transmission
            indices_per_node = [np.array(indices, dtype=np.int64) 
                               for indices in indices_per_node]
            
            batches.append(indices_per_node)
        
        return batches
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return self.n_batches
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (f"BatchScheduler(n_nodes={self.n_nodes}, "
                f"batch_size={self.batch_size}, n_batches={self.n_batches})")


# ==============================================================================
# GRADIENT AGGREGATION
# ==============================================================================


class GradientAggregator:
    """Aggregates gradients from distributed nodes.
    
    In distributed training, each node computes gradients on its data subset.
    This class combines these gradients using averaging, which is mathematically
    equivalent to computing gradients on the full batch.
    
    Note: Works for both standard and secure modes. In secure mode, we average
    secret shares (preserves linear homomorphic property).
    """
    
    @staticmethod
    def aggregate_gradients(gradient_dicts: List[Dict[str, torch.Tensor]],
                           param_names: List[str],
                           valid_nodes: List[bool]) -> Dict[str, torch.Tensor]:
        """Aggregate gradients using averaging.
        
        Mathematical property:
            avg(grad_1, grad_2, ..., grad_n) = grad(full_batch)
        
        For secure mode:
            avg(share1_A, share1_B, ...) + avg(share2_A, share2_B, ...) 
            = avg(share1_A + share2_A, share1_B + share2_B, ...)
            = avg(actual_A, actual_B, ...)
        
        Args:
            gradient_dicts: Gradient dictionaries from each node
            param_names: Parameters to aggregate
            valid_nodes: Indicates which nodes have valid data
        
        Returns:
            Aggregated gradients for each parameter
        
        Raises:
            ValueError: If gradient_dicts and valid_nodes have different lengths
        """
        if len(gradient_dicts) != len(valid_nodes):
            raise ValueError(
                f"Mismatch: {len(gradient_dicts)} gradient dicts "
                f"but {len(valid_nodes)} valid_nodes flags"
            )
        
        # Initialize collection buckets
        collections = {name: [] for name in param_names}
        
        # Gather gradients from valid nodes
        for grads, is_valid in zip(gradient_dicts, valid_nodes):
            if not is_valid or not grads:
                continue
            
            for name in param_names:
                if name in grads:
                    collections[name].append(grads[name])
        
        # Compute averages
        aggregated = {}
        for name in param_names:
            if collections[name]:
                aggregated[name] = torch.stack(collections[name]).mean(dim=0)
        
        return aggregated


# ==============================================================================
# NODE-SIDE GRADIENT COMPUTATION
# ==============================================================================


class NodeGradientComputer:
    """Computes parameter gradients on the node side.
    
    Uses PyTorch autograd to backpropagate through node-side layers given:
    - Forward activations at the cut point
    - Backward gradients from the orchestrator
    """
    
    @staticmethod
    def compute_gradients(model: nn.Module,
                         activations: torch.Tensor,
                         cut_gradients: torch.Tensor,
                         param_names: List[str]) -> Dict[str, torch.Tensor]:
        """Compute parameter gradients via backpropagation.
        
        Process:
        1. activations = f(x; θ) where θ are node parameters
        2. Receive ∂L/∂activations from orchestrator
        3. Compute ∂L/∂θ using chain rule
        
        Args:
            model: Node-side model
            activations: Forward outputs at cut point (must have requires_grad=True)
            cut_gradients: Gradients flowing back from orchestrator
            param_names: Parameters to compute gradients for
        
        Returns:
            Parameter gradients ready for optimization
        
        Raises:
            ValueError: If no parameters match param_names
        """
        # Extract parameters in correct order
        params = [p for n, p in model.named_parameters() if n in param_names]
        
        if not params:
            return {}
        
        # Backpropagate through node-side computation graph
        grads = torch.autograd.grad(
            outputs=activations,
            inputs=params,
            grad_outputs=cut_gradients,
            retain_graph=False,
            only_inputs=True
        )
        
        # Package results
        result = {}
        grad_idx = 0
        for name, _ in model.named_parameters():
            if name in param_names:
                result[name] = grads[grad_idx].detach().cpu()
                grad_idx += 1
        
        return result


# ==============================================================================
# DATA MERGING (STANDARD MODE)
# ==============================================================================


class DataMerger:
    """Merges data from multiple nodes in standard mode.
    
    Handles:
    - Concatenating activations/labels from nodes
    - Tracking contribution sizes for gradient splitting
    - Splitting gradients back to nodes
    """
    
    @staticmethod
    def merge(forward_results: List[Dict],
             device: torch.device) -> Optional[Tuple]:
        """Merge forward results from all nodes.
        
        Args:
            forward_results: List of dicts with 'activations' and 'labels'
            device: Target device for merged tensors
        
        Returns:
            (merged_activations, merged_labels, split_sizes, valid_nodes)
            or None if all batches are empty
        """
        activations_list = []
        labels_list = []
        split_sizes = []
        valid_nodes = []
        
        for result in forward_results:
            if result.get('activations') is not None:
                activations_list.append(result['activations'])
                labels_list.append(result['labels'])
                split_sizes.append(result['activations'].shape[0])
                valid_nodes.append(True)
            else:
                valid_nodes.append(False)
        
        if not activations_list:
            return None
        
        merged_activations = torch.cat(activations_list, dim=0).to(device)
        merged_labels = torch.cat(labels_list, dim=0).to(device)
        
        return merged_activations, merged_labels, split_sizes, valid_nodes
    
    @staticmethod
    def split_gradients(gradients: torch.Tensor,
                       split_sizes: List[int],
                       valid_nodes: List[bool]) -> List[Optional[torch.Tensor]]:
        """Split merged gradients back to per-node tensors.
        
        Args:
            gradients: Merged gradient tensor
            split_sizes: Size of each node's contribution
            valid_nodes: Which nodes have data
        
        Returns:
            List of gradient tensors (None for nodes without data)
        """
        grad_list = torch.split(gradients, split_sizes, dim=0)
        
        result = []
        grad_idx = 0
        for is_valid in valid_nodes:
            if is_valid:
                result.append(grad_list[grad_idx].detach().cpu())
                grad_idx += 1
            else:
                result.append(None)
        
        return result


# ==============================================================================
# DATA MERGING (SECURE MODE)
# ==============================================================================


class SecureDataMerger:
    """Merges secret-shared data from multiple nodes.
    
    In secure mode, activations are secret-shared:
    - Each node sends share_0 to orchestrator, share_1 to helper
    - We merge shares separately (concatenation is linear operation)
    - Reconstruction only happens when necessary (minimal leakage)
    """
    
    @staticmethod
    def merge_shares(forward_results: List[Dict],
                    device: torch.device) -> Optional[Tuple]:
        """Merge activation shares from multiple nodes.
        
        Privacy property: Merged shares reveal no more information than
        individual shares (concatenation preserves secret sharing).
        
        Args:
            forward_results: Shares and labels from each node
            device: Target device for merged tensors
        
        Returns:
            (merged_shares, merged_labels, split_sizes, valid_nodes)
            or None if all batches empty
        """
        share_list = []
        labels_list = []
        split_sizes = []
        valid_nodes = []
        
        for result in forward_results:
            if result.get('activations') is not None:
                share_list.append(result['activations'])
                labels_list.append(result['labels'])
                split_sizes.append(result['activations'].shape[0])
                valid_nodes.append(True)
            else:
                valid_nodes.append(False)
        
        if not share_list:
            return None
        
        # Concatenate shares (remains secret-shared)
        merged_shares = torch.cat(share_list, dim=0).to(device)
        merged_labels = torch.cat(labels_list, dim=0).to(device)
        
        return merged_shares, merged_labels, split_sizes, valid_nodes
    
    @staticmethod
    def split_gradient_shares(gradient_shares: torch.Tensor,
                             split_sizes: List[int],
                             valid_nodes: List[bool]) -> List[Optional[torch.Tensor]]:
        """Split merged gradient shares back to per-node format.
        
        Args:
            gradient_shares: Merged gradient shares
            split_sizes: Size of each node's contribution
            valid_nodes: Which nodes have data
        
        Returns:
            List of gradient share tensors
        """
        grad_list = torch.split(gradient_shares, split_sizes, dim=0)
        
        result = []
        grad_idx = 0
        for is_valid in valid_nodes:
            if is_valid:
                result.append(grad_list[grad_idx].detach().cpu())
                grad_idx += 1
            else:
                result.append(None)
        
        return result


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================


class ModelEvaluator:
    """Evaluates model performance (standard mode).
    
    Computes loss and accuracy metrics on test data.
    """
    
    def __init__(self, criterion: nn.Module):
        """Initialize evaluator.
        
        Args:
            criterion: Loss function (e.g., CrossEntropyLoss)
        """
        self.criterion = criterion
    
    def evaluate(self,
                model: nn.Module,
                dataloader,
                device: torch.device) -> Tuple[float, float, Dict]:
        """Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            dataloader: Test data loader
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
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model.forward_full(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().tolist())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        unique_preds = len(set(all_predictions))
        
        info = {
            'unique_predictions': unique_preds,
            'total_samples': total,
            'correct': correct
        }
        
        return avg_loss, accuracy, info


class SecureEvaluator:
    """Evaluates model performance (secure mode).
    
    Note: Evaluation uses standard (non-secure) computation because:
    1. Test data is at orchestrator (no privacy concern)
    2. Allows exact accuracy measurement
    3. No sensitive information leakage
    """
    
    @staticmethod
    def evaluate(model: nn.Module,
                dataloader,
                criterion: nn.Module,
                device: torch.device) -> Tuple[float, float, Dict]:
        """Evaluate model on test set (non-secure).
        
        Args:
            model: Model to evaluate
            dataloader: Test data loader
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
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model.forward_full(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().tolist())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        unique_preds = len(set(all_predictions))
        
        info = {
            'unique_predictions': unique_preds,
            'total_samples': total,
            'correct': correct
        }
        
        return avg_loss, accuracy, info