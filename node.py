"""Node for TL++."""

import argparse
import torch
import logging
from datetime import datetime, timezone, timedelta

from runtime.protocol import NodeCommunicator, SecureNodeCommunicator, SecretSharing
from core.data import NodeDataLoader
from runtime.utils import NodeGradientComputer


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8080
DEFAULT_HELPER_PORT = 8081


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


def setup_logging():
    """Configure logging with KST timestamps."""
    formatter = KSTFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S KST'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(console_handler)


# ==============================================================================
# MAIN NODE CLASS
# ==============================================================================


class SecureNode:
    """Training node with optional secure mode.
    
    Responsibilities:
    - Load and manage class-specific training data
    - Execute forward pass on node-side model layers
    - Compute gradients for node-side parameters
    - Communicate with orchestrator (and helper in secure mode)
    
    Modes:
    - Standard: Direct communication with orchestrator
    - Secure: Dual communication with orchestrator + helper using secret sharing
    """
    
    def __init__(self, config: dict):
        """Initialize node.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.secure_mode = config.get('secure', False)
        
        # Setup computation device
        self._setup_device()
        
        logging.info(f"Device: {self.device}")
        logging.info(f"Secure mode: {'✓ ENABLED' if self.secure_mode else 'DISABLED'}")
        
        # Setup communication
        self._setup_communication()
        
        # Initialize state (set during setup)
        self.node_id = None
        self.model = None
        self.dataset = None
        self.param_names = None
    
    def _setup_device(self) -> None:
        """Select computation device based on availability and config."""
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
        """Initialize communication layer based on mode."""
        if self.secure_mode:
            self.comm = SecureNodeCommunicator(
                orch_host=self.config.get('orch_host', '127.0.0.1'),
                orch_port=self.config.get('orch_port', 8080),
                helper_host=self.config.get('helper_host', '127.0.0.1'),
                helper_port=self.config.get('helper_port', 8081)
            )
        else:
            self.comm = NodeCommunicator(
                host=self.config.get('host', '127.0.0.1'),
                port=self.config.get('port', 8080)
            )
    
    # ==========================================================================
    # SETUP PHASE
    # ==========================================================================
    
    def setup(self):
        """Setup node: connect, receive config, load data, initialize model.
        
        Steps:
        1. Connect to orchestrator (and helper in secure mode)
        2. Receive configuration (node_id, model architecture)
        3. Load class-specific training data
        4. Initialize model
        5. Report dataset size to orchestrator
        """
        # Step 1: Connect
        self.comm.connect()
        if self.secure_mode:
            logging.info(f"✓ Connected to orchestrator and helper")
        else:
            logging.info(f"✓ Connected to orchestrator at {self.config.get('host')}:{self.config.get('port')}")
        
        # Step 2: Receive configuration
        init_config = self.comm.receive_init()
        self.node_id = init_config['node_id']
        logging.info(f"✓ Assigned node ID: {self.node_id}")
        
        # Log noise configuration if in secure mode
        if self.secure_mode:
            act_noise, grad_noise = SecretSharing.get_noise_scaling()
            logging.info(f"✓ Noise config (activation: {act_noise:.1%}, gradient: {grad_noise:.1%})")
        
        # Step 3: Initialize model
        node_model = init_config.get('node_model')
        if node_model is None:
            raise ValueError("Model not provided in initialization")
        
        self.model = node_model.to(self.device)
        logging.info("✓ Received model from orchestrator")
        
        # Get parameter names for gradient computation
        param_names, _ = self.model.get_state_names()
        self.param_names = param_names
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"✓ Model parameters: {total_params:,}")
        
        # Step 4: Load dataset
        self.dataset = NodeDataLoader(
            class_id=self.node_id,
            augment=True
        )
        logging.info(f"✓ Loaded {len(self.dataset)} training samples")
        
        # Step 5: Report dataset size
        self.comm.send_dataset_size(len(self.dataset))
        logging.info("✓ Ready for training")
        logging.info("")
    
    # ==========================================================================
    # TRAINING PHASE
    # ==========================================================================
    
    def run(self):
        """Main training loop.
        
        Continuously processes batches until training completes.
        """
        batch_count = 0
        
        while True:
            if not self.process_batch():
                break
            
            batch_count += 1
            if batch_count % 100 == 0:
                logging.info(f"Processed {batch_count} batches")
        
        logging.info(f"✓ Training complete. Total batches: {batch_count}")
    
    def process_batch(self) -> bool:
        """Process one training batch.
        
        Flow:
        1. Receive batch assignment (model state + sample indices)
        2. Load batch data from local dataset
        3. Forward pass through node-side layers
        4. Send results (or shares in secure mode)
        5. Receive gradients (or reconstruct from shares)
        6. Compute parameter gradients
        7. Send gradients (or shares in secure mode)
        
        Returns:
            True if batch processed successfully, False if training complete
        """
        try:
            # Step 1: Receive batch assignment
            assignment = self.comm.receive_batch_assignment()
            model_state = assignment['model_state']
            sample_indices = assignment['sample_indices']
            
            # Check for shutdown signal
            if model_state is None:
                logging.info("Training complete - received shutdown signal")
                return False
            
            # Step 2: Update model with latest weights
            model_state_device = {
                name: tensor.to(self.device)
                for name, tensor in model_state.items()
            }
            self.model.load_state_dict(model_state_device, strict=False)
            self.model.train()
            
            # Step 3: Handle empty batch
            if len(sample_indices) == 0:
                self._process_empty_batch()
                return True
            
            # Step 4: Load batch data
            images, labels = self._load_batch(sample_indices)
            
            # Step 5: Forward pass
            images.requires_grad_(True)
            activations = self.model(images)
            
            # Step 6: Send forward results (mode-dependent)
            if self.secure_mode:
                self.comm.send_forward_result_secure(activations, labels)
            else:
                self.comm.send_forward_result(
                    activations=activations.detach().cpu(),
                    labels=labels.cpu()
                )
            
            # Step 7: Receive gradients (mode-dependent)
            if self.secure_mode:
                cut_gradients = self.comm.receive_backward_signal_secure()
            else:
                cut_gradients = self.comm.receive_backward_signal()
            
            if cut_gradients is None:
                self._send_empty_gradients()
                return True
            
            # Step 8: Compute parameter gradients
            cut_gradients = cut_gradients.to(self.device)
            node_gradients = NodeGradientComputer.compute_gradients(
                model=self.model,
                activations=activations,
                cut_gradients=cut_gradients,
                param_names=self.param_names
            )
            
            # Step 9: Send parameter gradients (mode-dependent)
            if self.secure_mode:
                self.comm.send_gradient_result_secure(node_gradients)
            else:
                self.comm.send_gradient_result(node_gradients)
            
            return True
            
        except ConnectionError as e:
            if "closed by peer" in str(e).lower():
                logging.info("Training complete - connection closed")
            else:
                logging.error(f"Connection error: {e}")
            return False
        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            return False
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_empty_batch(self):
        """Handle empty batch (no samples assigned to this node)."""
        if self.secure_mode:
            self.comm.send_forward_result_secure(None, None)
            self.comm.receive_backward_signal_secure()
            self.comm.send_gradient_result_secure({})
        else:
            self.comm.send_forward_result(None, None)
            self.comm.receive_backward_signal()
            self.comm.send_gradient_result({})
    
    def _send_empty_gradients(self):
        """Send empty gradients when no backward pass needed."""
        if self.secure_mode:
            self.comm.send_gradient_result_secure({})
        else:
            self.comm.send_gradient_result({})
    
    def _load_batch(self, indices):
        """Load batch data from local dataset.
        
        Args:
            indices: Sample indices to load
            
        Returns:
            (images, labels) tensors
        """
        images = []
        labels = []
        
        for idx in indices:
            img, label = self.dataset[int(idx)]
            images.append(img)
            labels.append(label)
        
        images = torch.stack(images).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return images, labels
    
    # ==========================================================================
    # CLEANUP
    # ==========================================================================
    
    def cleanup(self):
        """Close connections and cleanup resources."""
        if self.comm:
            self.comm.close()
        logging.info("✓ Cleanup complete")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TL++ Training Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Standard mode arguments
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                       help='Orchestrator host (standard mode)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help='Orchestrator port (standard mode)')
    
    # Secure mode arguments
    parser.add_argument('--secure', action='store_true',
                       help='Enable secure mode with secret sharing')
    parser.add_argument('--orch_host', type=str, default=DEFAULT_HOST,
                       help='Orchestrator host (secure mode)')
    parser.add_argument('--orch_port', type=int, default=DEFAULT_PORT,
                       help='Orchestrator port (secure mode)')
    parser.add_argument('--helper_host', type=str, default=DEFAULT_HOST,
                       help='Helper host (secure mode)')
    parser.add_argument('--helper_port', type=int, default=DEFAULT_HELPER_PORT,
                       help='Helper port (secure mode)')

    # Hardware
    parser.add_argument('--no_accel', action='store_true',
                       help='Use CPU only')
    
    args = parser.parse_args()
    return vars(args)


def main():
    """Main entry point for node process."""
    # Parse configuration
    config = parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("TL++ Training Node")
    logging.info("=" * 80)
    logging.info("")

    # Create and run node
    node = SecureNode(config)
    
    try:
        node.setup()
        node.run()
    except KeyboardInterrupt:
        logging.info("")
        logging.info("=" * 80)
        logging.info("Node interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error(f"Node failed: {e}")
        logging.error("=" * 80)
        import traceback
        traceback.print_exc()
    finally:
        node.cleanup()


if __name__ == '__main__':
    main()