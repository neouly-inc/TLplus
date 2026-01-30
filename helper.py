"""Helper node for TL++ secure mode."""

import argparse
import socket
import logging
import torch
from datetime import datetime, timezone, timedelta

from runtime.protocol import SecureSocketCommunicator, SecureMessageType, SecretSharing
from core.models import create_orchestrator_model


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_HOST = '127.0.0.1'
DEFAULT_NODE_PORT = 8081
DEFAULT_ORCH_PORT = 8082


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
# HELPER NODE HANDLER
# ==============================================================================


class HelperNodeHandler(SecureSocketCommunicator):
    """Handler for individual node connection on helper side.
    
    Manages communication with a single training node.
    """
    
    def __init__(self, sock: socket.socket, address: tuple, node_id: int):
        """Initialize helper node handler.
        
        Args:
            sock: Connected socket
            address: Node address (host, port)
            node_id: Node identifier
        """
        self.socket = sock
        self.address = address
        self.node_id = node_id
        self.activation_share = None
    
    def receive_forward_share(self) -> dict:
        """Receive activation share from node.
        
        Returns:
            Dict with activation share
        """
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.SHARE_FORWARD
        
        # Store share for later use
        self.activation_share = payload.get('activations')
        
        return payload
    
    def send_backward_share(self, gradient_share) -> None:
        """Send gradient share to node.
        
        Args:
            gradient_share: Gradient tensor share
        """
        self._send_message(self.socket, SecureMessageType.SHARE_BACKWARD, gradient_share)
    
    def receive_gradient_share(self) -> dict:
        """Receive parameter gradient share from node.
        
        Returns:
            Dict with parameter gradient shares
        """
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.SHARE_GRADIENT
        return payload
    
    def close(self) -> None:
        """Close connection."""
        self.socket.close()


# ==============================================================================
# MAIN HELPER NODE CLASS
# ==============================================================================


class HelperNode:
    """Helper server for secure two-server MPC with minimal reconstruction.
    
    Responsibilities:
    - Accept connections from training nodes
    - Connect to orchestrator for coordination
    - Receive and process secret shares
    - Compute forward/backward passes on shares
    - Never reconstruct intermediate activations
    
    Privacy Properties:
    ✓ Intermediate activations: Never reconstructed
    ✓ Cut-layer gradients: Computed on shares separately
    ✓ Output shares: Sent to orchestrator (reconstructed there only)
    """
    
    def __init__(self, config: dict):
        """Initialize helper node.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Network configuration
        self.node_host = config.get('node_host', '127.0.0.1')
        self.node_port = config.get('node_port', 8081)
        self.orch_host = config.get('orch_host', '127.0.0.1')
        self.orch_port = config.get('orch_port', 8082)
        
        # Computation (helper uses CPU to save GPU resources for nodes)
        self.model = None
        self.device = torch.device('cpu')
        
        # Connection management
        self.node_server_socket = None
        self.orch_socket = None
        self.node_handlers = []
        
        # Computation state (updated each batch)
        self.merged_share_1 = None
        self.merged_share_1_input = None
        self.outputs_share_1 = None
        self.batch_count = 0
        
        logging.info(f"Helper node initialized")
        logging.info(f"  Node server:    {self.node_host}:{self.node_port}")
        logging.info(f"  Orchestrator:   {self.orch_host}:{self.orch_port}")
        logging.info(f"  Device:         {self.device}")
        logging.info("")
    
    # ==========================================================================
    # SETUP PHASE
    # ==========================================================================
    
    def start_node_server(self) -> None:
        """Start listening for node connections."""
        self.node_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.node_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.node_server_socket.bind((self.node_host, self.node_port))
        self.node_server_socket.listen(10)
        
        logging.info(f"✓ Helper server started on {self.node_host}:{self.node_port}")
    
    def connect_to_orchestrator(self) -> None:
        """Connect to orchestrator for coordination."""
        self.orch_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.orch_socket.connect((self.orch_host, self.orch_port))
        
        # Send ready signal
        comm = SecureSocketCommunicator()
        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'status': 'connected'
        })
        
        # Receive noise configuration from orchestrator
        msg_type, noise_config = comm._receive_message(self.orch_socket)
        if msg_type == SecureMessageType.HELPER_INIT and noise_config.get('phase') == 'noise_config':
            SecretSharing.configure_noise_scaling(
                activation_noise=noise_config['activation_noise'],
                gradient_noise=noise_config['gradient_noise']
            )
            act_noise, grad_noise = SecretSharing.get_noise_scaling()
            logging.info(f"✓ Received noise config (activation: {act_noise:.1%}, gradient: {grad_noise:.1%})")
    
        logging.info(f"✓ Connected to orchestrator at {self.orch_host}:{self.orch_port}")
    
    def accept_node(self) -> HelperNodeHandler:
        """Accept a node connection.
        
        Returns:
            Handler for the connected node
        """
        node_socket, node_address = self.node_server_socket.accept()
        handler = HelperNodeHandler(node_socket, node_address, len(self.node_handlers))
        self.node_handlers.append(handler)
        return handler
    
    def wait_for_nodes(self, n_nodes: int) -> None:
        """Wait for all training nodes to connect.
        
        Args:
            n_nodes: Expected number of nodes
        """
        logging.info(f"Waiting for {n_nodes} node(s)...")
        
        for i in range(n_nodes):
            self.accept_node()
            logging.info(f"  ✓ Node {i+1}/{n_nodes} connected")
        
        logging.info(f"✓ All {n_nodes} node(s) connected")
        logging.info("")
    
    # ==========================================================================
    # TRAINING PHASE
    # ==========================================================================
    
    def run(self) -> None:
        """Main processing loop.
        
        Continuously processes batches until training completes.
        """
        logging.info("=" * 80)
        logging.info("Helper node ready for secure MPC")
        logging.info("=" * 80)
        logging.info("")
        
        while True:
            if not self.process_batch():
                break
        
        logging.info("")
        logging.info(f"✓ Processing complete. Total batches: {self.batch_count}")
    
    def process_batch(self) -> bool:
        """Process one batch in minimal reconstruction protocol.
        
        The batch is processed in multiple phases coordinated with orchestrator.
        
        Returns:
            True to continue, False to stop
        """
        try:
            comm = SecureSocketCommunicator()
            
            # Receive coordination message from orchestrator
            msg_type, coord_msg = comm._receive_message(self.orch_socket)
            
            if msg_type == SecureMessageType.SHUTDOWN:
                return False
            
            phase = coord_msg.get('phase')
            
            # ==================================================================
            # PHASE 1: FORWARD PASS - Receive model and collect shares
            # ==================================================================
            
            if phase == 'forward':
                return self._phase_forward(comm, coord_msg)
            
            # ==================================================================
            # PHASE 2: COMPUTE FORWARD ON SHARE_1
            # ==================================================================
            
            elif phase == 'compute' and coord_msg.get('action') == 'forward':
                return self._phase_compute_forward(comm)
            
            # ==================================================================
            # PHASE 3: COMPUTE BACKWARD ON SHARE_1
            # ==================================================================
            
            elif phase == 'compute' and coord_msg.get('action') == 'backward':
                return self._phase_compute_backward(comm, coord_msg)
            
            # ==================================================================
            # PHASE 4: DISTRIBUTE GRADIENT SHARES TO NODES
            # ==================================================================
            
            elif phase == 'backward':
                return self._phase_backward(comm, coord_msg)
            
            else:
                logging.warning(f"Unknown phase: {phase}")
                return True
            
        except ConnectionError:
            logging.info("Connection closed - training complete")
            return False
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _phase_forward(self, comm, coord_msg) -> bool:
        """Phase 1: Receive model and collect activation shares.
        
        Steps:
        1. Receive model state from orchestrator
        2. Update local model
        3. Collect share_1 from all nodes
        4. Merge shares across nodes
        5. Send confirmation to orchestrator
        
        Args:
            comm: Communication helper
            coord_msg: Coordination message from orchestrator
            
        Returns:
            True to continue
        """
        # Receive model configuration
        model_state = coord_msg.get('model_state')
        cut_layer = coord_msg.get('cut_layer', 1)
        num_classes = coord_msg.get('num_classes', 1)
        
        # Create or update model
        if self.model is None:
            self.model = create_orchestrator_model(
                num_classes=num_classes,
                cut_layer=cut_layer
            ).to(self.device)
        
        # Load model weights
        if model_state is not None:
            self.model.load_state_dict(model_state)
            self.model.eval()  # Helper doesn't train, just computes
        
        # Collect share_1 from all nodes
        forward_shares = []
        for handler in self.node_handlers:
            share_data = handler.receive_forward_share()
            forward_shares.append(share_data)
        
        # Merge shares
        share_list = []
        for result in forward_shares:
            if result.get('activations') is not None:
                share_list.append(result['activations'])
        
        if not share_list:
            # Empty batch
            comm._send_message(self.orch_socket, SecureMessageType.SHUTDOWN, {})
            return True
        
        # Merge helper's shares (share_1)
        self.merged_share_1 = torch.cat(share_list, dim=0).to(self.device)
        
        # Send confirmation to orchestrator
        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'forward_shares': forward_shares,
            'status': 'forward_collected'
        })
        
        return True
    
    def _phase_compute_forward(self, comm) -> bool:
        """Phase 2: Compute forward pass on share_1.
        
        Steps:
        1. Enable gradient tracking on share_1
        2. Forward pass through model
        3. Send output share to orchestrator
        
        Args:
            comm: Communication helper
            
        Returns:
            True to continue
        """
        # Compute forward pass on share_1 (with gradient tracking)
        self.merged_share_1_input = self.merged_share_1.requires_grad_(True)
        self.outputs_share_1 = self.model.forward_from_cut(self.merged_share_1_input)
        
        # Send output share to orchestrator
        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'output_share': self.outputs_share_1.detach().cpu(),
            'status': 'forward_computed'
        })
        
        return True
    
    def _phase_compute_backward(self, comm, coord_msg) -> bool:
        """Phase 3: Compute backward pass on share_1.
        
        Steps:
        1. Receive loss gradient from orchestrator
        2. Backpropagate through share_1 computation
        3. Extract gradient at cut point
        4. Send gradient share to orchestrator
        
        Args:
            comm: Communication helper
            coord_msg: Coordination message with output gradient
            
        Returns:
            True to continue
        """
        # Receive the actual gradient from the loss
        output_gradient = coord_msg.get('output_gradient')
        
        if output_gradient is None or self.outputs_share_1 is None or self.merged_share_1_input is None:
            cut_grad_share_1 = None
        else:
            # Move gradient to device
            output_gradient = output_gradient.to(self.device)
            
            # Backpropagate using the actual loss gradient
            # This is dL/d(outputs) where outputs = outputs_share_0 + outputs_share_1
            self.outputs_share_1.backward(output_gradient)
            
            # Extract gradient at cut point
            cut_grad_share_1 = self.merged_share_1_input.grad
        
        # Send gradient share to orchestrator
        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'cut_gradient': cut_grad_share_1.detach().cpu() if cut_grad_share_1 is not None else None,
            'status': 'backward_computed'
        })
        
        return True
    
    def _phase_backward(self, comm, coord_msg) -> bool:
        """Phase 4: Distribute gradient shares to nodes.
        
        Steps:
        1. Receive gradient shares from orchestrator
        2. Send shares to nodes
        3. Collect parameter gradient shares from nodes
        4. Forward to orchestrator
        
        Args:
            comm: Communication helper
            coord_msg: Coordination message with gradient shares
            
        Returns:
            True to continue
        """
        gradient_shares = coord_msg.get('gradient_shares', [])
        
        # Send gradient shares (share_1) to nodes
        for handler, grad_share in zip(self.node_handlers, gradient_shares):
            handler.send_backward_share(grad_share)
        
        # Collect parameter gradient shares from nodes
        node_grad_shares = []
        for handler in self.node_handlers:
            grad_share = handler.receive_gradient_share()
            node_grad_shares.append(grad_share)
        
        # Send parameter gradient shares to orchestrator
        comm._send_message(self.orch_socket, SecureMessageType.HELPER_READY, {
            'gradient_shares': node_grad_shares,
            'status': 'backward_complete'
        })
        
        self.batch_count += 1
        
        if self.batch_count % 100 == 0:
            logging.info(f"Processed {self.batch_count} batches")
        
        return True
    
    # ==========================================================================
    # CLEANUP
    # ==========================================================================
    
    def cleanup(self) -> None:
        """Close all connections and cleanup resources."""
        logging.info("")
        logging.info("Closing connections...")
        
        for handler in self.node_handlers:
            handler.close()
        
        if self.orch_socket:
            self.orch_socket.close()
        
        if self.node_server_socket:
            self.node_server_socket.close()
        
        logging.info("✓ Cleanup complete")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Helper Node for TL++ Secure Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                       help='Host for node connections')
    parser.add_argument('--port', type=int, default=DEFAULT_NODE_PORT,
                       help='Port for node connections')
    parser.add_argument('--orch_host', type=str, default=DEFAULT_HOST,
                       help='Orchestrator host for coordination')
    parser.add_argument('--orch_port', type=int, default=DEFAULT_ORCH_PORT,
                       help='Orchestrator coordination port')
    parser.add_argument('--n_nodes', type=int, required=True,
                       help='Expected number of training nodes')
    
    args = parser.parse_args()
    return vars(args)


def main():
    """Main entry point for helper node."""
    config = parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("=" * 80)
    logging.info("TL++ Helper Node (Secure Mode)")
    logging.info("=" * 80)
    logging.info("")

    # Create helper
    helper = HelperNode(config)
    
    try:
        # Setup connections
        helper.start_node_server()
        helper.connect_to_orchestrator()
        helper.wait_for_nodes(config['n_nodes'])
        
        # Run training loop
        helper.run()
        
    except KeyboardInterrupt:
        logging.info("")
        logging.info("=" * 80)
        logging.info("Helper interrupted by user")
        logging.info("=" * 80)
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error(f"Helper failed: {e}")
        logging.error("=" * 80)
        import traceback
        traceback.print_exc()
    finally:
        helper.cleanup()


if __name__ == '__main__':
    main()