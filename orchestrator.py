"""Orchestrator for TL++."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict

from runtime.protocol import (
    OrchestratorCommunicator, SecureOrchestratorCommunicator,
    SecretSharing, SecureSocketCommunicator, SecureMessageType
)
from core.data import OrchestratorDataLoader
from core.models import create_orchestrator_model, create_node_model
from runtime.utils import (
    BatchScheduler, GradientAggregator, ModelEvaluator, DataMerger,
    SecureDataMerger, SecureEvaluator
)


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8080
DEFAULT_HELPER_PORT = 8082
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_TEST_BATCH_SIZE = 256
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 30
DEFAULT_LR = 0.1
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_GRAD_CLIP_NORM = 1.0
NOTIFICATION_SLEEP_TIME = 1  # seconds


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


def setup_logging(log_dir: str = 'logs', prefix: str = 'TL++_base') -> Path:
    """Configure logging to file and console.
    
    Args:
        log_dir: Directory for log files
        prefix: Prefix for log filename
        
    Returns:
        Path to created log file
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'{prefix}_{timestamp}.log'
    
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
    logging.info("TL++ Orchestrator")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")
    
    return log_file


# ==============================================================================
# SECURE TRAINING PHASES
# ==============================================================================


class SecureTrainingPhases:
    """Encapsulates the phases of secure MPC training to improve code organization."""
    
    def __init__(self, orchestrator: 'SecureOrchestrator'):
        """Initialize with reference to parent orchestrator.
        
        Args:
            orchestrator: Parent SecureOrchestrator instance
        """
        self.orch = orchestrator
        self.comm_helper = SecureSocketCommunicator()
    
    def phase1_collect_shares(self, sample_indices: List) -> Optional[Tuple]:
        """Phase 1: Collect activation shares from nodes.
        
        Args:
            sample_indices: Sample indices for each node
            
        Returns:
            Tuple of (merged_share_0, merged_labels, split_sizes, valid_nodes)
            or None if batch is empty
        """
        # Send batch assignments to nodes
        node_state = self.orch.model.get_node_state()
        self.orch.comm.broadcast_batch_assignment(node_state, sample_indices)
        
        # Coordinate with helper - send model and configuration
        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {
                'phase': 'forward',
                'model_state': self.orch.model.state_dict(),
                'cut_layer': self.orch.config.get('cut_layer', 1),
                'num_classes': self.orch.config.get('n_nodes', 1),
                'n_nodes': len(self.orch.comm.node_handlers)
            }
        )
        
        # Collect share_0 from nodes
        forward_results_orch = self.orch.comm.collect_forward_results()
        
        # Wait for helper to collect share_1
        msg_type, helper_forward = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type == SecureMessageType.SHUTDOWN:
            self._send_empty_signals(len(forward_results_orch))
            return None
        
        # Merge orchestrator's shares
        merged_data = SecureDataMerger.merge_shares(forward_results_orch, self.orch.device)
        if merged_data is None:
            self.comm_helper._send_message(
                self.orch.comm.helper_socket,
                SecureMessageType.SHUTDOWN,
                {}
            )
            self._send_empty_signals(len(forward_results_orch))
            return None
        
        return merged_data
    
    def phase2_forward_computation(self, merged_share_0: torch.Tensor) -> Tuple:
        """Phase 2: Compute forward pass on shares.
        
        Args:
            merged_share_0: Orchestrator's share of activations
            
        Returns:
            Tuple of (outputs_share_0, merged_share_0_input) or (None, None) on error
        """
        # Orchestrator computes forward on share_0
        self.orch.model.train()
        merged_share_0_input = merged_share_0.requires_grad_(True)
        outputs_share_0 = self.orch.model.forward_from_cut(merged_share_0_input)
        
        # Signal helper to compute forward on share_1
        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {'phase': 'compute', 'action': 'forward'}
        )
        
        # Receive helper's output share
        msg_type, helper_output = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type != SecureMessageType.HELPER_READY:
            return None, None
        
        outputs_share_1 = helper_output['output_share'].to(self.orch.device)
        
        return outputs_share_0, outputs_share_1, merged_share_0_input
    
    def phase3_loss_and_backward(
        self,
        outputs_share_0: torch.Tensor,
        outputs_share_1: torch.Tensor,
        merged_share_0_input: torch.Tensor,
        merged_labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Phase 3: Compute loss and backward pass on shares.
        
        Args:
            outputs_share_0: Orchestrator's output share
            outputs_share_1: Helper's output share
            merged_share_0_input: Input to orchestrator's forward (with grad)
            merged_labels: Ground truth labels
            
        Returns:
            Reconstructed cut gradient or None on error
        """
        # Reconstruct outputs for loss computation
        # CRITICAL: Detach both shares to create clean computational graph
        outputs_for_loss = outputs_share_0.detach() + outputs_share_1.detach()
        outputs_for_loss.requires_grad_(True)
        
        # Compute loss on reconstructed outputs
        loss = self.orch.criterion(outputs_for_loss, merged_labels)
        
        # Get gradient at output level
        loss.backward()
        output_gradient = outputs_for_loss.grad.clone()
        
        # Manually backprop through orchestrator's share_0
        self.orch.optimizer.zero_grad()
        outputs_share_0.backward(output_gradient)
        
        # Get gradient at cut point for share_0
        cut_grad_share_0 = merged_share_0_input.grad
        
        if cut_grad_share_0 is None or output_gradient is None:
            self.comm_helper._send_message(
                self.orch.comm.helper_socket,
                SecureMessageType.SHUTDOWN,
                {}
            )
            return None, None
        
        # Signal helper to compute backward on share_1
        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {
                'phase': 'compute',
                'action': 'backward',
                'output_gradient': output_gradient.detach().cpu()
            }
        )
        
        # Receive helper's cut gradient
        msg_type, helper_grad = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type != SecureMessageType.HELPER_READY:
            return None, loss.item()
        
        cut_grad_share_1 = helper_grad['cut_gradient'].to(self.orch.device)
        
        # Reconstruct true gradient at cut point
        cut_gradient_reconstructed = cut_grad_share_0 + cut_grad_share_1
        
        return cut_gradient_reconstructed, loss.item()
    
    def phase4_distribute_gradients(
        self,
        cut_gradient: torch.Tensor,
        split_sizes: List[int],
        valid_nodes: List[bool]
    ) -> Optional[float]:
        """Phase 4: Distribute gradient shares and collect parameter updates.
        
        Args:
            cut_gradient: Reconstructed gradient at cut point
            split_sizes: Size of each node's contribution
            valid_nodes: Which nodes have valid data
            
        Returns:
            Loss value or None on error
        """
        # Re-share the gradient for distribution to nodes
        cut_gradient_share_0_new, cut_gradient_share_1_new = SecretSharing.share_tensor(
            cut_gradient,
            SecretSharing._gradient_noise_scale
        )
        
        # Split shares by node
        grad_splits_0 = torch.split(cut_gradient_share_0_new, split_sizes, dim=0)
        grad_splits_1 = torch.split(cut_gradient_share_1_new, split_sizes, dim=0)
        
        gradient_shares_orch = []
        gradient_shares_helper = []
        
        grad_idx = 0
        for is_valid in valid_nodes:
            if is_valid:
                gradient_shares_orch.append(grad_splits_0[grad_idx].detach().cpu())
                gradient_shares_helper.append(grad_splits_1[grad_idx].detach().cpu())
                grad_idx += 1
            else:
                gradient_shares_orch.append(None)
                gradient_shares_helper.append(None)
        
        # Send share_0 to nodes
        self.orch.comm.broadcast_backward_signal(gradient_shares_orch)
        
        # Send share_1 to helper for distribution
        self.comm_helper._send_message(
            self.orch.comm.helper_socket,
            SecureMessageType.HELPER_INIT,
            {'phase': 'backward', 'gradient_shares': gradient_shares_helper}
        )
        
        # Collect parameter gradient shares from nodes
        gradient_share_results_orch = self.orch.comm.collect_gradient_results()
        
        # Receive parameter gradient shares from helper
        msg_type, helper_param_grads = self.comm_helper._receive_message(
            self.orch.comm.helper_socket
        )
        if msg_type == SecureMessageType.SHUTDOWN:
            return None
        
        gradient_share_results_helper = helper_param_grads.get('gradient_shares', [])
        
        # Reconstruct and apply parameter gradients
        self._apply_reconstructed_gradients(
            gradient_share_results_orch,
            gradient_share_results_helper,
            valid_nodes
        )
        
        return True
    
    def _apply_reconstructed_gradients(
        self,
        grad_shares_orch: List[Dict],
        grad_shares_helper: List[Dict],
        valid_nodes: List[bool]
    ) -> None:
        """Reconstruct and apply parameter gradients to model.
        
        Args:
            grad_shares_orch: Gradient shares from orchestrator path
            grad_shares_helper: Gradient shares from helper path
            valid_nodes: Which nodes have valid data
        """
        node_param_names = self.orch.model.get_node_param_names()
        reconstructed_gradients = []
        
        for grad_orch, grad_helper, is_valid in zip(
            grad_shares_orch,
            grad_shares_helper,
            valid_nodes
        ):
            if not is_valid or not grad_orch or not grad_helper:
                reconstructed_gradients.append({})
                continue
            
            recon_grads = {}
            for name in node_param_names:
                if name in grad_orch and name in grad_helper:
                    recon_grads[name] = SecretSharing.reconstruct_tensor(
                        grad_orch[name], grad_helper[name]
                    )
            
            reconstructed_gradients.append(recon_grads)
        
        # Aggregate gradients
        aggregated_grads = GradientAggregator.aggregate_gradients(
            reconstructed_gradients, node_param_names, valid_nodes
        )
        
        # Apply gradients to model
        for name, param in self.orch.model.named_parameters():
            if name in aggregated_grads:
                if param.grad is None:
                    param.grad = aggregated_grads[name].to(self.orch.device)
                else:
                    param.grad += aggregated_grads[name].to(self.orch.device)
        
        # Gradient clipping
        if self.orch.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.orch.model.parameters(),
                max_norm=self.orch.grad_clip_norm
            )
        
        # Update parameters
        self.orch.optimizer.step()
    
    def _send_empty_signals(self, n_nodes: int) -> None:
        """Send empty signals to nodes when batch is empty."""
        self.orch.comm.broadcast_backward_signal([None] * n_nodes)
        self.orch.comm.collect_gradient_results()


# ==============================================================================
# MAIN ORCHESTRATOR CLASS
# ==============================================================================


class SecureOrchestrator:
    """Orchestrator for distributed training with optional secure mode.
    
    Responsibilities:
    - Network setup: Accept connections from nodes (and helper in secure mode)
    - Model management: Create and distribute model architecture
    - Training coordination: Schedule batches, aggregate gradients, update weights
    - Evaluation: Test model performance
    
    Modes:
    - Standard: Direct communication with nodes
    - Secure: Three-party MPC with orchestrator, nodes, and helper
    """
    
    def __init__(self, config: dict):
        """Initialize orchestrator.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.secure_mode = config.get('secure', False)
        
        # Determine computation device
        self._setup_device()
        
        logging.info(f"Computation device: {self.device}")
        logging.info(f"Secure mode: {'✓ ENABLED' if self.secure_mode else 'DISABLED'}")
        logging.info("")
        
        # Initialize communication layer
        self._setup_communication()
        
        # Initialize training components (set up later)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.test_loader = None
        self.batch_scheduler = None
        self.evaluator = None
        self.grad_clip_norm = None
        
        # Initialize secure training helper if needed
        if self.secure_mode:
            self.secure_phases = SecureTrainingPhases(self)
    
    def _setup_device(self) -> None:
        """Select computation device based on availability."""
        no_accel = self.config.get('no_accel', False)
        
        if no_accel:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
    
    def _setup_communication(self) -> None:
        """Initialize communication protocols based on mode."""
        if self.secure_mode:
            self.comm = SecureOrchestratorCommunicator(
                host=self.config.get('host', DEFAULT_HOST),
                port=self.config.get('port', DEFAULT_PORT),
                helper_host=self.config.get('helper_host', DEFAULT_HOST),
                helper_port=self.config.get('helper_port', DEFAULT_HELPER_PORT)
            )
        else:
            self.comm = OrchestratorCommunicator(
                host=self.config.get('host', DEFAULT_HOST),
                port=self.config.get('port', DEFAULT_PORT)
            )
    
    # ==========================================================================
    # SETUP PHASE
    # ==========================================================================
    
    def setup(self):
        """Complete orchestrator setup: network, model, data, optimization.
        
        Steps:
        1. Start network servers
        2. Wait for helper (secure mode only)
        3. Accept node connections
        4. Create and broadcast model
        5. Collect dataset information
        6. Initialize training components
        """
        logging.info("-" * 80)
        logging.info("NETWORK SETUP")
        logging.info("-" * 80)
        
        # Start listening for connections
        self.comm.start()
        logging.info(f"✓ Orchestrator listening on {self.config.get('host')}:{self.config.get('port')}")
        
        # Wait for helper in secure mode
        if self.secure_mode:
            self._setup_helper()
        
        # Accept node connections
        self._accept_nodes()
        
        # Setup model and data
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        
        logging.info("=" * 80)
        logging.info("Setup complete - Ready for training")
        logging.info("=" * 80)
        logging.info("")
    
    def _setup_helper(self) -> None:
        """Setup helper node connection (secure mode only)."""
        logging.info(f"✓ Helper node on {self.config.get('helper_host')}:{self.config.get('helper_port')}")
        logging.info("")
        logging.info("Waiting for helper node...")
        self.comm.wait_for_helper()
        logging.info(f"✓ Helper node connected")
        
        # Send noise configuration to helper
        act_noise, grad_noise = SecretSharing.get_noise_scaling()
        self.comm.send_noise_config_to_helper(act_noise, grad_noise)
        logging.info(f"✓ Noise config sent to helper (activation: {act_noise:.1%}, gradient: {grad_noise:.1%})")
        logging.info("")
    
    def _accept_nodes(self) -> None:
        """Accept all training node connections."""
        n_nodes = self.config.get('n_nodes', 1)
        logging.info(f"Waiting for {n_nodes} node(s)...")
        for i in range(n_nodes):
            self.comm.accept_node()
            logging.info(f"  ✓ Node {i+1}/{n_nodes} connected")
        logging.info(f"✓ All {n_nodes} node(s) connected")
        logging.info("")
    
    def _setup_model(self) -> None:
        """Create model and broadcast to nodes."""
        logging.info("-" * 80)
        logging.info("MODEL INITIALIZATION")
        logging.info("-" * 80)
        
        cut_layer = self.config.get('cut_layer', 1)
        n_nodes = self.config.get('n_nodes', 1)
        
        # Create node-side model template
        node_model_template = create_node_model(cut_layer=cut_layer)
        node_params = sum(p.numel() for p in node_model_template.parameters())
        logging.info(f"Node-side parameters: {node_params:,}")
        
        # Broadcast configuration to nodes
        logging.info("")
        logging.info("-" * 80)
        logging.info("BROADCASTING CONFIGURATION")
        logging.info("-" * 80)
        
        if self.secure_mode:
            act_noise, grad_noise = SecretSharing.get_noise_scaling()
            self.comm.broadcast_init(
                node_model=node_model_template,
                activation_noise=act_noise,
                gradient_noise=grad_noise
            )
        else:
            self.comm.broadcast_init(node_model=node_model_template)
        
        logging.info(f"✓ Configuration broadcast to all nodes")
        
        # Create full orchestrator model
        self.model = create_orchestrator_model(
            num_classes=n_nodes,
            cut_layer=cut_layer
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logging.info("")
        logging.info(f"Total parameters:     {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_data(self) -> None:
        """Load test data and create batch scheduler."""
        logging.info("")
        logging.info("-" * 80)
        logging.info("DATA LOADING")
        logging.info("-" * 80)
        
        # Collect dataset sizes from nodes
        samples_per_node = self.comm.collect_dataset_sizes()
        total_samples = sum(samples_per_node)
        
        logging.info(f"✓ Dataset information collected:")
        for i, samples in enumerate(samples_per_node):
            logging.info(f"    Node {i+1}: {samples:,} samples")
        logging.info(f"  Total training samples: {total_samples:,}")
        logging.info("")
        
        # Load test data
        self.test_loader = OrchestratorDataLoader(
            batch_size=self.config.get('test_batch_size', DEFAULT_TEST_BATCH_SIZE),
            num_classes=self.config.get('n_nodes', 1)
        )
        logging.info(f"✓ Test dataset loaded: {len(self.test_loader):,} samples")
        logging.info("")
        
        # Create batch scheduler
        self.batch_scheduler = BatchScheduler(
            samples_per_node=samples_per_node,
            batch_size=self.config.get('train_batch_size', DEFAULT_TRAIN_BATCH_SIZE),
            shuffle=True
        )
        logging.info(f"✓ Batch scheduler: {len(self.batch_scheduler)} batches/epoch")
        logging.info("")
    
    def _setup_optimization(self) -> None:
        """Initialize optimizer, scheduler, and other training components."""
        logging.info("-" * 80)
        logging.info("OPTIMIZATION SETUP")
        logging.info("-" * 80)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.get('lr', DEFAULT_LR),
            momentum=self.config.get('momentum', DEFAULT_MOMENTUM),
            weight_decay=self.config.get('weight_decay', DEFAULT_WEIGHT_DECAY),
            nesterov=self.config.get('nesterov', True)
        )
        
        # Learning rate scheduler
        self._setup_scheduler()
        
        # Gradient clipping
        self.grad_clip_norm = self.config.get('grad_clip_norm', DEFAULT_GRAD_CLIP_NORM)
        
        # Evaluator
        if self.secure_mode:
            self.evaluator = SecureEvaluator()
        else:
            self.evaluator = ModelEvaluator(self.criterion)
        
        logging.info(f"✓ Optimizer: SGD (lr={self.config.get('lr', DEFAULT_LR)}, momentum={self.config.get('momentum', DEFAULT_MOMENTUM)})")
        logging.info(f"✓ Scheduler: {self.config.get('scheduler', 'cosine')}")
        logging.info(f"✓ Gradient clipping: {self.grad_clip_norm}")
        logging.info("")
    
    def _setup_scheduler(self):
        """Configure learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', DEFAULT_EPOCHS)
        
        if scheduler_type == 'cosine':
            t_max = self.config.get('t_max') or epochs
            eta_min = self.config.get('eta_min', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=t_max, eta_min=eta_min
            )
        elif scheduler_type == 'step':
            step_size = self.config.get('step_size', 50)
            gamma = self.config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'multistep':
            milestones = self.config.get('milestones', [60, 120, 160])
            gamma = self.config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    # ==========================================================================
    # TRAINING PHASE
    # ==========================================================================
    
    def train(self):
        """Main training loop with early stopping."""
        logging.info("-" * 80)
        logging.info("TRAINING")
        logging.info("-" * 80)
        
        epochs = self.config.get('epochs', DEFAULT_EPOCHS)
        patience = self.config.get('patience', DEFAULT_PATIENCE)
        
        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logging.info("")
            logging.info(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6f}")
            
            # Train one epoch
            train_loss = self.train_epoch(epoch, epochs)
            
            # Evaluate
            test_loss, test_acc, info = self.evaluate()
            
            logging.info(
                f"  Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}%"
            )
            
            # Check for model collapse
            if info['unique_predictions'] == 1:
                logging.warning("  ⚠️ Model predicting only 1 class - possible collapse")
            
            # Update learning rate
            self.scheduler.step()
            
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
        
        logging.info("")
        logging.info("Notifying nodes of completion...")
        self._notify_training_complete()
    
    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Average training loss
        """
        self.model.train()
        batches = self.batch_scheduler.create_epoch_batches()
        
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(batches, desc=f'Epoch {epoch+1}/{total_epochs}',
                   ncols=100, leave=False)
        
        for batch_idx, sample_indices in enumerate(pbar):
            # Choose training method based on mode
            if self.secure_mode:
                loss = self._train_batch_secure(sample_indices)
            else:
                loss = self._train_batch_standard(sample_indices)
            
            if loss is not None:
                epoch_loss += loss
                n_batches += 1
                pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
    def _train_batch_standard(self, sample_indices: list) -> Optional[float]:
        """Train one batch in standard mode.
        
        Flow:
        1. Broadcast model state and batch assignments to nodes
        2. Collect forward results (activations + labels)
        3. Merge data and compute loss
        4. Backward pass to get cut-point gradients
        5. Send gradients to nodes
        6. Collect and aggregate parameter gradients
        7. Update model
        
        Args:
            sample_indices: Sample indices for each node
            
        Returns:
            Batch loss (or None if empty batch)
        """
        # Step 1: Send batch assignments
        node_state = self.model.get_node_state()
        self.comm.broadcast_batch_assignment(node_state, sample_indices)
        
        # Step 2: Collect forward results
        forward_results = self.comm.collect_forward_results()
        merged_data = DataMerger.merge(forward_results, self.device)
        
        if merged_data is None:
            # Empty batch - send empty signals
            self.comm.broadcast_backward_signal([None] * len(forward_results))
            self.comm.collect_gradient_results()
            return None
        
        merged_activations, merged_labels, split_sizes, valid_nodes = merged_data
        
        # Step 3: Forward pass from cut point
        self.model.train()
        merged_activations_input = merged_activations.to(self.device).requires_grad_(True)
        outputs = self.model.forward_from_cut(merged_activations_input)
        loss = self.criterion(outputs, merged_labels)
        
        # Step 4: Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        cut_gradients = merged_activations_input.grad
        if cut_gradients is None:
            self.comm.broadcast_backward_signal([None] * len(valid_nodes))
            self.comm.collect_gradient_results()
            return None
        
        # Step 5: Split and send gradients to nodes
        gradient_splits = DataMerger.split_gradients(
            cut_gradients, split_sizes, valid_nodes
        )
        self.comm.broadcast_backward_signal(gradient_splits)
        
        # Step 6: Collect and aggregate node gradients
        gradient_results = self.comm.collect_gradient_results()
        node_param_names = self.model.get_node_param_names()
        aggregated_grads = GradientAggregator.aggregate_gradients(
            gradient_results, node_param_names, valid_nodes
        )
        
        # Step 7: Apply aggregated gradients
        for name, param in self.model.named_parameters():
            if name in aggregated_grads:
                if param.grad is None:
                    param.grad = aggregated_grads[name].to(self.device)
                else:
                    param.grad += aggregated_grads[name].to(self.device)
        
        # Gradient clipping
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def _train_batch_secure(self, sample_indices: list) -> Optional[float]:
        """Train one batch in secure mode with minimal reconstruction.
        
        Uses SecureTrainingPhases to organize the complex multi-phase protocol.
        
        Args:
            sample_indices: Sample indices for each node
            
        Returns:
            Batch loss (or None if empty batch)
        """
        # Phase 1: Collect activation shares
        merged_data = self.secure_phases.phase1_collect_shares(sample_indices)
        if merged_data is None:
            return None
        
        merged_share_0, merged_labels, split_sizes, valid_nodes = merged_data
        
        # Phase 2: Forward computation on shares
        result = self.secure_phases.phase2_forward_computation(merged_share_0)
        if result[0] is None:
            self.secure_phases._send_empty_signals(len(valid_nodes))
            return None
        
        outputs_share_0, outputs_share_1, merged_share_0_input = result
        
        # Phase 3: Loss computation and backward pass
        result = self.secure_phases.phase3_loss_and_backward(
            outputs_share_0, outputs_share_1, merged_share_0_input, merged_labels
        )
        if result[0] is None:
            self.secure_phases._send_empty_signals(len(valid_nodes))
            return None
        
        cut_gradient_reconstructed, loss_value = result
        
        # Phase 4: Distribute gradients and update parameters
        success = self.secure_phases.phase4_distribute_gradients(
            cut_gradient_reconstructed, split_sizes, valid_nodes
        )
        if not success:
            return None
        
        return loss_value
    
    # ==========================================================================
    # EVALUATION PHASE
    # ==========================================================================
    
    def evaluate(self) -> Tuple[float, float, Dict]:
        """Evaluate model on test set.
        
        Returns:
            (average_loss, accuracy_percentage, info_dict)
        """
        if self.secure_mode:
            return self.evaluator.evaluate(
                self.model, self.test_loader, self.criterion, self.device
            )
        else:
            return self.evaluator.evaluate(
                self.model, self.test_loader, self.device
            )
    
    # ==========================================================================
    # CLEANUP
    # ==========================================================================
    
    def _notify_training_complete(self):
        """Notify all nodes that training is complete."""
        try:
            # Send empty batch assignment as shutdown signal
            empty_assignment = [[] for _ in self.comm.node_handlers]
            self.comm.broadcast_batch_assignment(None, empty_assignment)
            
            import time
            time.sleep(NOTIFICATION_SLEEP_TIME)
            
            logging.info("✓ Nodes notified")
        except Exception as e:
            logging.debug(f"Could not notify nodes: {e}")
    
    def cleanup(self):
        """Close all connections and cleanup resources."""
        logging.info("")
        logging.info("Closing connections...")
        self.comm.close()
        logging.info("✓ Cleanup complete")


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TL++ Orchestrator with Secure Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Network configuration
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                       help='Host address to bind')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help='Port for node connections')
    
    # Secure mode
    parser.add_argument('--secure', action='store_true',
                       help='Enable secure mode with MPC')
    parser.add_argument('--helper_host', type=str, default=DEFAULT_HOST,
                       help='Helper coordination host (secure mode)')
    parser.add_argument('--helper_port', type=int, default=DEFAULT_HELPER_PORT,
                       help='Helper coordination port (secure mode)')
    
    # Secure mode noise configuration
    parser.add_argument('--activation_noise', type=float, default=0.02,
                       help='Noise scale for activations in secure mode (0.0-1.0, default: 0.02 = 2%%)')
    parser.add_argument('--gradient_noise', type=float, default=0.10,
                       help='Noise scale for gradients in secure mode (0.0-1.0, default: 0.10 = 10%%)')

    # Model configuration
    parser.add_argument('--cut_layer', type=int, default=1, choices=[1, 2, 3],
                       help='Model split point')
    parser.add_argument('--n_nodes', type=int, default=1,
                       help='Number of training nodes')
    
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
    
    # Hardware
    parser.add_argument('--no_accel', action='store_true',
                       help='Use CPU only')
    
    args = parser.parse_args()
    
    if not 1 <= args.n_nodes <= 10:
        parser.error('n_nodes must be between 1 and 10')
    
    return vars(args)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def main():
    """Main entry point."""
    config = parse_args()
    
    # Setup logging
    prefix = 'TL++_secure' if config.get('secure') else 'TL++_base'
    log_file = setup_logging(prefix=prefix)
    logging.info("")

    # Configure noise scaling for secure mode
    if config.get('secure'):
        SecretSharing.configure_noise_scaling(
            activation_noise=config.get('activation_noise'),
            gradient_noise=config.get('gradient_noise')
        )
    
    # Log configuration
    logging.info("-" * 80)
    logging.info("CONFIGURATION")
    logging.info("-" * 80)
    logging.info(f"Secure mode:      {config.get('secure')}")
    if config.get('secure'):
        logging.info(f"Helper endpoint:  {config.get('helper_host')}:{config.get('helper_port')}")
        logging.info(f"Activation noise: {config.get('activation_noise'):.1%}")
        logging.info(f"Gradient noise:   {config.get('gradient_noise'):.1%}")
    logging.info(f"Nodes:            {config.get('n_nodes')}")
    logging.info(f"Cut layer:        {config.get('cut_layer')}")
    logging.info(f"Batch size:       {config.get('train_batch_size')}")
    logging.info(f"Learning rate:    {config.get('lr')}")
    logging.info("")
    
    # Create and run orchestrator
    orchestrator = SecureOrchestrator(config)
    
    try:
        orchestrator.setup()
        orchestrator.train()
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
        orchestrator.cleanup()
        logging.info("")
        logging.info(f"Full log: {log_file}")
        logging.info("=" * 80)


if __name__ == '__main__':
    main()