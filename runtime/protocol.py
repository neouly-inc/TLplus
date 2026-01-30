"""Communication protocols for TL++."""

import socket
import pickle
import struct
import torch
from typing import Any, Tuple, List, Dict, Optional
from enum import IntEnum


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Protocol constants
HEADER_SIZE = 8  # bytes: 4 for message type + 4 for payload length
HEADER_FORMAT = '!II'  # Network byte order, two unsigned ints

# Default timeout for socket operations (seconds)
DEFAULT_SOCKET_TIMEOUT = 30.0


# ==============================================================================
# MESSAGE TYPES
# ==============================================================================


class MessageType(IntEnum):
    """Message identifiers for standard communication protocol.
    
    Each type corresponds to a phase in the training workflow.
    """
    # Initialization
    INIT = 1                # Orchestrator → Node: Configuration
    DATASET_SIZE = 2        # Node → Orchestrator: Dataset size
    
    # Training loop
    BATCH_ASSIGNMENT = 3    # Orchestrator → Node: Batch and model state
    FORWARD_RESULT = 4      # Node → Orchestrator: Forward activations
    BACKWARD_SIGNAL = 5     # Orchestrator → Node: Cut-point gradients
    GRADIENT_RESULT = 6     # Node → Orchestrator: Parameter gradients
    
    # Control
    SHUTDOWN = 99           # Graceful shutdown


class SecureMessageType(IntEnum):
    """Extended message types for secure communication.
    
    Includes all standard types plus secure-mode coordination messages.
    """
    # Standard messages (same as MessageType)
    INIT = 1
    DATASET_SIZE = 2
    BATCH_ASSIGNMENT = 3
    FORWARD_RESULT = 4
    BACKWARD_SIGNAL = 5
    GRADIENT_RESULT = 6
    SHUTDOWN = 99
    
    # Secure mode coordination
    HELPER_INIT = 10        # Orchestrator → Helper: Coordination signal
    HELPER_READY = 11       # Helper → Orchestrator: Ready/result signal
    SHARE_FORWARD = 12      # Node → Helper: Activation share
    SHARE_BACKWARD = 13     # Helper → Node: Gradient share
    SHARE_GRADIENT = 14     # Node → Helper: Parameter gradient share


# ==============================================================================
# SECRET SHARING PRIMITIVES
# ==============================================================================


class SecretSharing:
    """Additive secret sharing for tensor privacy with configurable noise.
    
    Principle: Split tensor T into random shares S₀, S₁ where S₀ + S₁ = T
    
    Security: Neither share reveals information about T alone
    Computation: Linear operations work on shares (homomorphic property)
    
    Example:
        T = [1, 2, 3]
        S₀ = [0.5, 1.8, 2.1]  (random with controlled noise)
        S₁ = [0.5, 0.2, 0.9]  (= T - S₀)
        
        Operations:
        avg(S₀_A, S₀_B) + avg(S₁_A, S₁_B) = avg(T_A, T_B)
    
    Noise Configuration:
        Use configure_noise_scaling() to set global noise levels:
        - activation_noise: For large-magnitude tensors (default: 0.02 = 2%)
        - gradient_noise: For small-magnitude tensors (default: 0.10 = 10%)
    """
    
    # Class-level configurable noise scaling factors
    _activation_noise_scale = 0.02  # 2% default for activations
    _gradient_noise_scale = 0.10    # 10% default for gradients
    
    @classmethod
    def configure_noise_scaling(cls,
                               activation_noise: Optional[float] = None,
                               gradient_noise: Optional[float] = None) -> None:
        """Configure global noise scaling factors for secret sharing.
        
        This method should be called once at program startup to set the desired
        noise levels for the entire training session. All components (orchestrator,
        helper, and nodes) must use identical noise settings.
        
        Args:
            activation_noise: Noise scale for activations (0.0-1.0)
                            E.g., 0.02 = 2% of standard deviation
                            Higher = more privacy, potentially lower accuracy
            gradient_noise: Noise scale for gradients (0.0-1.0)
                          E.g., 0.10 = 10% of standard deviation
                          Higher = more privacy, potentially slower convergence
        
        Raises:
            ValueError: If noise values are outside valid range [0.0, 1.0]
        
        Example:
            # Set 5% noise for activations, 15% for gradients
            SecretSharing.configure_noise_scaling(
                activation_noise=0.05,
                gradient_noise=0.15
            )
            
            # Disable noise (for testing only)
            SecretSharing.configure_noise_scaling(
                activation_noise=0.0,
                gradient_noise=0.0
            )
        """
        if activation_noise is not None:
            if not 0.0 <= activation_noise <= 1.0:
                raise ValueError(
                    f"activation_noise must be in [0.0, 1.0], got {activation_noise}"
                )
            cls._activation_noise_scale = activation_noise
        
        if gradient_noise is not None:
            if not 0.0 <= gradient_noise <= 1.0:
                raise ValueError(
                    f"gradient_noise must be in [0.0, 1.0], got {gradient_noise}"
                )
            cls._gradient_noise_scale = gradient_noise
    
    @classmethod
    def get_noise_scaling(cls) -> Tuple[float, float]:
        """Get current noise scaling factors.
        
        Returns:
            Tuple of (activation_noise_scale, gradient_noise_scale)
        
        Example:
            act_noise, grad_noise = SecretSharing.get_noise_scaling()
            print(f"Current noise: activation={act_noise:.1%}, gradient={grad_noise:.1%}")
        """
        return cls._activation_noise_scale, cls._gradient_noise_scale
    
    @staticmethod
    def share_tensor(tensor: torch.Tensor,
                    noise_scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split tensor into two additive shares with configurable noise.
        
        Strategy: Adaptive noise scaling based on tensor statistics
        - Uses configured noise scale
        - Can be overridden with explicit noise_scale parameter
        
        Args:
            tensor: Original tensor to share
            noise_scale: Optional explicit noise scale (0.0-1.0)
                       If provided, overrides automatic selection
        
        Returns:
            Tuple of (share_0, share_1) where share_0 + share_1 = tensor
            Returns (None, None) if input tensor is None
        
        Example:
            # Automatic noise selection based on tensor magnitude
            share_0, share_1 = SecretSharing.share_tensor(activations)
            
            # Explicit noise override
            share_0, share_1 = SecretSharing.share_tensor(tensor, noise_scale=0.05)
        """
        if tensor is None:
            return None, None
        
        # Compute tensor statistics for adaptive noise
        tensor_std = tensor.std().item() if tensor.numel() > 1 else 1.0
        
        # Determine noise scale if not explicitly provided
        if noise_scale is None:
            noise_scale = 0.0
        
        # Calculate actual noise standard deviation
        actual_noise_std = tensor_std * noise_scale
        
        # Generate first share with controlled randomness
        # share_0 = T/2 + N(0, noise_std)
        # share_1 = T - share_0
        # This ensures: share_0 + share_1 = T exactly
        share_0 = torch.randn_like(tensor) * actual_noise_std + tensor * 0.5
        share_1 = tensor - share_0

        return share_0, share_1
    
    @staticmethod
    def reconstruct_tensor(share_0: torch.Tensor, 
                          share_1: torch.Tensor) -> torch.Tensor:
        """Reconstruct original tensor from shares.
        
        Args:
            share_0: First share
            share_1: Second share
        
        Returns:
            Reconstructed tensor (share_0 + share_1)
            Returns None if either share is None
        """
        if share_0 is None or share_1 is None:
            return None
        
        return share_0 + share_1
    
    @staticmethod
    def share_dict(tensor_dict: Dict[str, torch.Tensor],
                  noise_scale: Optional[float] = None) -> Tuple[Dict, Dict]:
        """Share all tensors in a dictionary.
        
        Useful for sharing multiple gradients simultaneously while maintaining
        their association with parameter names.
        
        Args:
            tensor_dict: Dictionary mapping names to tensors
            noise_scale: Optional explicit noise scale for all tensors
                       If None, each tensor uses adaptive noise selection
        
        Returns:
            Tuple of (share_0_dict, share_1_dict)
            Returns ({}, {}) if input dictionary is empty
        
        Example:
            gradients = {'conv1.weight': grad1, 'conv1.bias': grad2}
            share_0_dict, share_1_dict = SecretSharing.share_dict(gradients)
        """
        if not tensor_dict:
            return {}, {}
        
        share_0_dict = {}
        share_1_dict = {}
        
        for name, tensor in tensor_dict.items():
            share_0, share_1 = SecretSharing.share_tensor(tensor, noise_scale)
            share_0_dict[name] = share_0
            share_1_dict[name] = share_1
        
        return share_0_dict, share_1_dict
    
    @staticmethod
    def reconstruct_dict(share_0_dict: Dict[str, torch.Tensor],
                        share_1_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reconstruct all tensors in a dictionary from their shares.
        
        Args:
            share_0_dict: First share dictionary
            share_1_dict: Second share dictionary
        
        Returns:
            Dictionary mapping names to reconstructed tensors
            Returns {} if either input dictionary is empty
        
        Example:
            reconstructed = SecretSharing.reconstruct_dict(share_0_dict, share_1_dict)
        """
        if not share_0_dict or not share_1_dict:
            return {}
        
        result = {}
        for name in share_0_dict.keys():
            result[name] = SecretSharing.reconstruct_tensor(
                share_0_dict[name], 
                share_1_dict[name]
            )
        
        return result


# ==============================================================================
# BASE SOCKET COMMUNICATION
# ==============================================================================


class SocketCommunicator:
    """Base class for socket-based typed messaging.
    
    Message Format:
    ┌─────────────┬──────────────┬────────────────┐
    │ Type (4B)   │ Length (4B)  │ Payload (N B)  │
    └─────────────┴──────────────┴────────────────┘
    
    - Type: Message identifier (unsigned int)
    - Length: Payload size in bytes (unsigned int)
    - Payload: Pickle-serialized Python object
    """
    
    def _send_message(self, sock: socket.socket, 
                     msg_type: MessageType, payload: Any) -> None:
        """Send typed message with serialization.
        
        Args:
            sock: Socket to send through
            msg_type: Message type identifier
            payload: Data to send (any pickleable object)
        
        Raises:
            ConnectionError: If socket connection fails
        """
        try:
            # Serialize payload
            serialized = pickle.dumps(payload)
            
            # Pack header: network byte order, unsigned ints
            header = struct.pack(HEADER_FORMAT, int(msg_type), len(serialized))
            
            # Send header + payload atomically
            sock.sendall(header + serialized)
            
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ConnectionError(f"Failed to send message type {msg_type}: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error sending message: {e}")
    
    def _receive_message(self, sock: socket.socket) -> Tuple[MessageType, Any]:
        """Receive typed message with deserialization.
        
        Args:
            sock: Socket to receive from
        
        Returns:
            (message_type, payload)
        
        Raises:
            ConnectionError: If socket connection fails
        """
        try:
            # Receive fixed-size header (8 bytes)
            header = self._receive_exact(sock, HEADER_SIZE)
            msg_type_int, payload_length = struct.unpack(HEADER_FORMAT, header)
            msg_type = MessageType(msg_type_int)
            
            # Receive variable-size payload
            serialized = self._receive_exact(sock, payload_length)
            payload = pickle.loads(serialized)
            
            return msg_type, payload
            
        except (ConnectionResetError, struct.error) as e:
            raise ConnectionError(f"Failed to receive message: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error receiving message: {e}")
    
    def _receive_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket.
        
        Note: sock.recv() may return fewer bytes than requested,
        so we loop until all bytes are received.
        
        Args:
            sock: Socket to receive from
            num_bytes: Exact number of bytes to receive
        
        Returns:
            Received bytes
        
        Raises:
            ConnectionError: If connection closed prematurely
        """
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by peer")
            data += chunk
        return data


class SecureSocketCommunicator:
    """Base class for secure socket communication.
    
    Identical to SocketCommunicator but uses SecureMessageType enum.
    Separated for type safety and clarity.
    """
    
    def _send_message(self, sock: socket.socket, 
                     msg_type: SecureMessageType, payload: Any) -> None:
        """Send typed message with serialization."""
        try:
            serialized = pickle.dumps(payload)
            header = struct.pack(HEADER_FORMAT, int(msg_type), len(serialized))
            sock.sendall(header + serialized)
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ConnectionError(f"Failed to send secure message type {msg_type}: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error sending secure message: {e}")
    
    def _receive_message(self, sock: socket.socket) -> Tuple[SecureMessageType, Any]:
        """Receive typed message with deserialization."""
        try:
            header = self._receive_exact(sock, HEADER_SIZE)
            msg_type_int, payload_length = struct.unpack(HEADER_FORMAT, header)
            msg_type = SecureMessageType(msg_type_int)
            
            serialized = self._receive_exact(sock, payload_length)
            payload = pickle.loads(serialized)
            
            return msg_type, payload
        except (ConnectionResetError, struct.error) as e:
            raise ConnectionError(f"Failed to receive secure message: {e}")
        except Exception as e:
            raise ConnectionError(f"Unexpected error receiving secure message: {e}")
    
    def _receive_exact(self, sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from socket."""
        data = b''
        while len(data) < num_bytes:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by peer")
            data += chunk
        return data


# ==============================================================================
# NODE COMMUNICATORS
# ==============================================================================


class NodeCommunicator(SocketCommunicator):
    """Node-side communication handler (standard mode).
    
    Manages single connection to orchestrator.
    
    Workflow:
    1. Connect to orchestrator
    2. Receive configuration (node_id, model architecture)
    3. Training loop:
       - Receive batch assignment
       - Send forward results
       - Receive backward gradients
       - Send parameter gradients
    """
    
    def __init__(self, host: str, port: int):
        """Initialize node communicator.
        
        Args:
            host: Orchestrator hostname/IP
            port: Orchestrator port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.node_id = None
    
    def connect(self) -> None:
        """Establish TCP connection to orchestrator."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
    
    def receive_init(self) -> dict:
        """Receive initialization configuration.
        
        Returns:
            Configuration dict with node_id, model, etc.
        """
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.INIT
        self.node_id = payload['node_id']
        return payload
    
    def send_dataset_size(self, n_samples: int) -> None:
        """Report dataset size to orchestrator.
        
        Args:
            n_samples: Number of training samples available
        """
        self._send_message(self.socket, MessageType.DATASET_SIZE, n_samples)
    
    def receive_batch_assignment(self) -> dict:
        """Receive batch assignment from orchestrator.
        
        Returns:
            Dict with 'model_state' and 'sample_indices'
        """
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.BATCH_ASSIGNMENT
        return payload
    
    def send_forward_result(self, activations: Any, labels: Any) -> None:
        """Send forward pass results to orchestrator.
        
        Args:
            activations: Output at cut layer (None if empty batch)
            labels: Ground truth labels (None if empty batch)
        """
        payload = {
            'activations': activations,
            'labels': labels,
        }
        self._send_message(self.socket, MessageType.FORWARD_RESULT, payload)
    
    def receive_backward_signal(self) -> Any:
        """Receive gradients at cut point.
        
        Returns:
            Gradient tensor (None if no backward needed)
        """
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.BACKWARD_SIGNAL
        return payload
    
    def send_gradient_result(self, gradients: dict) -> None:
        """Send computed parameter gradients.
        
        Args:
            gradients: Dict mapping parameter names to gradients
        """
        self._send_message(self.socket, MessageType.GRADIENT_RESULT, gradients)
    
    def close(self) -> None:
        """Close connection gracefully."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


class SecureNodeCommunicator(SecureSocketCommunicator):
    """Node-side communication handler (secure mode).
    
    Manages dual connections:
    - Orchestrator: Receives tasks, sends share_0
    - Helper: Sends share_1, receives helper's share
    
    Secret Sharing Protocol:
    1. Forward: Split activations → share_0 to orch, share_1 to helper
    2. Backward: Receive share_0 from orch, share_1 from helper → reconstruct
    3. Gradients: Split gradients → share_0 to orch, share_1 to helper
    """
    
    def __init__(self, orch_host: str, orch_port: int,
                 helper_host: str, helper_port: int):
        """Initialize secure node communicator.
        
        Args:
            orch_host: Orchestrator hostname
            orch_port: Orchestrator port
            helper_host: Helper hostname
            helper_port: Helper port
        """
        self.orch_host = orch_host
        self.orch_port = orch_port
        self.helper_host = helper_host
        self.helper_port = helper_port
        
        self.orch_socket = None
        self.helper_socket = None
        self.node_id = None
    
    def connect(self) -> None:
        """Connect to both orchestrator and helper."""
        # Connect to orchestrator
        self.orch_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.orch_socket.connect((self.orch_host, self.orch_port))
        
        # Connect to helper
        self.helper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.helper_socket.connect((self.helper_host, self.helper_port))
    
    def receive_init(self) -> dict:
        """Receive initialization from orchestrator."""
        msg_type, payload = self._receive_message(self.orch_socket)
        assert msg_type == SecureMessageType.INIT
        self.node_id = payload['node_id']
        
        # Configure noise if provided by orchestrator
        if 'activation_noise' in payload and 'gradient_noise' in payload:
            SecretSharing.configure_noise_scaling(
                activation_noise=payload['activation_noise'],
                gradient_noise=payload['gradient_noise']
            )
        
        return payload
    
    def send_dataset_size(self, n_samples: int) -> None:
        """Send dataset size to orchestrator."""
        self._send_message(self.orch_socket, SecureMessageType.DATASET_SIZE, n_samples)
    
    def receive_batch_assignment(self) -> dict:
        """Receive batch assignment from orchestrator."""
        msg_type, payload = self._receive_message(self.orch_socket)
        assert msg_type == SecureMessageType.BATCH_ASSIGNMENT
        return payload
    
    def send_forward_result_secure(self, activations: torch.Tensor, 
                                   labels: torch.Tensor) -> None:
        """Send forward results using secret sharing.
        
        Privacy protocol:
        - Activations: Split into shares, send separately
        - Labels: Send plaintext to orchestrator (needed for loss)
        
        Args:
            activations: Forward outputs at cut point
            labels: Ground truth labels
        """
        if activations is None:
            # Empty batch
            self._send_message(self.orch_socket, SecureMessageType.FORWARD_RESULT, {
                'activations': None,
                'labels': None
            })
            self._send_message(self.helper_socket, SecureMessageType.SHARE_FORWARD, {
                'activations': None
            })
            return
        
        # Split activations into shares (uses configured activation noise scale)
        share_0, share_1 = SecretSharing.share_tensor(
            activations,
            noise_scale=SecretSharing._activation_noise_scale
        )
        
        # Send share_0 + labels to orchestrator
        self._send_message(self.orch_socket, SecureMessageType.FORWARD_RESULT, {
            'activations': share_0.detach().cpu(),
            'labels': labels.cpu()
        })
        
        # Send share_1 to helper
        self._send_message(self.helper_socket, SecureMessageType.SHARE_FORWARD, {
            'activations': share_1.detach().cpu(),
            'node_id': self.node_id
        })
    
    def receive_backward_signal_secure(self) -> torch.Tensor:
        """Receive and reconstruct backward gradients.
        
        Privacy protocol:
        - Receive share_0 from orchestrator
        - Receive share_1 from helper
        - Reconstruct locally
        
        Returns:
            Reconstructed gradients
        """
        # Receive share_0 from orchestrator
        msg_type, payload_orch = self._receive_message(self.orch_socket)
        assert msg_type == SecureMessageType.BACKWARD_SIGNAL
        share_0 = payload_orch
        
        # Receive share_1 from helper
        msg_type, payload_helper = self._receive_message(self.helper_socket)
        assert msg_type == SecureMessageType.SHARE_BACKWARD
        share_1 = payload_helper
        
        # Reconstruct
        if share_0 is None or share_1 is None:
            return None
        
        return SecretSharing.reconstruct_tensor(share_0, share_1)
    
    def send_gradient_result_secure(self, gradients: Dict[str, torch.Tensor]) -> None:
        """Send parameter gradients using secret sharing.
        
        Privacy protocol:
        - Split each gradient into shares
        - Send share_0 to orchestrator, share_1 to helper
        
        Args:
            gradients: Parameter gradients to share
        """
        if not gradients:
            # Empty gradients
            self._send_message(self.orch_socket, SecureMessageType.GRADIENT_RESULT, {})
            self._send_message(self.helper_socket, SecureMessageType.SHARE_GRADIENT, {})
            return
        
        # Share all gradients (uses configured gradient noise scale)
        share_0_dict, share_1_dict = SecretSharing.share_dict(
            gradients,
            noise_scale=SecretSharing._gradient_noise_scale
        )
        
        # Send share_0 to orchestrator
        self._send_message(self.orch_socket, SecureMessageType.GRADIENT_RESULT, share_0_dict)
        
        # Send share_1 to helper
        self._send_message(self.helper_socket, SecureMessageType.SHARE_GRADIENT, share_1_dict)
    
    def close(self) -> None:
        """Close both connections."""
        if self.orch_socket:
            try:
                self.orch_socket.close()
            except:
                pass
        if self.helper_socket:
            try:
                self.helper_socket.close()
            except:
                pass


# ==============================================================================
# ORCHESTRATOR COMMUNICATORS
# ==============================================================================


class OrchestratorCommunicator(SocketCommunicator):
    """Orchestrator-side communication manager (standard mode).
    
    Manages connections to multiple nodes for coordinated training.
    
    Responsibilities:
    - Accept node connections
    - Broadcast configuration and batch assignments
    - Collect forward/gradient results from all nodes
    """
    
    def __init__(self, host: str, port: int):
        """Initialize orchestrator communicator.
        
        Args:
            host: Host address to bind ('0.0.0.0' for all interfaces)
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.node_handlers = []
    
    def start(self) -> None:
        """Start listening for node connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
    
    def accept_node(self) -> 'NodeHandler':
        """Accept a node connection.
        
        Returns:
            NodeHandler for the connected node
        """
        node_socket, node_address = self.server_socket.accept()
        handler = NodeHandler(node_socket, node_address, len(self.node_handlers))
        self.node_handlers.append(handler)
        return handler
    
    def broadcast_init(self, **kwargs) -> None:
        """Broadcast initialization to all nodes.
        
        Args:
            **kwargs: Configuration parameters (node_model, etc.)
        """
        for handler in self.node_handlers:
            config = {
                'node_id': handler.node_id,
                **kwargs
            }
            handler.send_init(config)
    
    def collect_dataset_sizes(self) -> List[int]:
        """Collect dataset sizes from all nodes.
        
        Returns:
            List of dataset sizes (one per node)
        """
        return [handler.receive_dataset_size() for handler in self.node_handlers]
    
    def broadcast_batch_assignment(self, model_state: dict, 
                                   sample_indices_per_node: list) -> None:
        """Broadcast batch assignments to all nodes.
        
        Args:
            model_state: Current model parameters
            sample_indices_per_node: Sample indices for each node
        """
        for handler, sample_indices in zip(self.node_handlers, sample_indices_per_node):
            handler.send_batch_assignment(model_state, sample_indices)
    
    def collect_forward_results(self) -> List[dict]:
        """Collect forward results from all nodes.
        
        Returns:
            List of forward result dicts
        """
        return [handler.receive_forward_result() for handler in self.node_handlers]
    
    def broadcast_backward_signal(self, gradients_per_node: list) -> None:
        """Broadcast backward gradients to all nodes.
        
        Args:
            gradients_per_node: Gradient tensor for each node
        """
        for handler, gradients in zip(self.node_handlers, gradients_per_node):
            handler.send_backward_signal(gradients)
    
    def collect_gradient_results(self) -> List[dict]:
        """Collect parameter gradients from all nodes.
        
        Returns:
            List of gradient dicts
        """
        return [handler.receive_gradient_result() for handler in self.node_handlers]
    
    def close(self) -> None:
        """Close all connections."""
        for handler in self.node_handlers:
            handler.close()
        if self.server_socket:
            self.server_socket.close()


class SecureOrchestratorCommunicator(SecureSocketCommunicator):
    """Orchestrator-side communication manager (secure mode).
    
    Manages:
    - Connections to multiple nodes
    - Coordination with helper server
    
    Secure Protocol Flow:
    1. Helper connects to orchestrator's coordination port
    2. Nodes connect to orchestrator's main port
    3. Training: Orchestrator and helper coordinate via separate channel
    """
    
    def __init__(self, host: str, port: int, 
                 helper_host: str, helper_port: int):
        """Initialize secure orchestrator communicator.
        
        Args:
            host: Host for node connections
            port: Port for node connections
            helper_host: Host for helper coordination server
            helper_port: Port for helper coordination server
        """
        self.host = host
        self.port = port
        self.helper_host = helper_host
        self.helper_port = helper_port
        
        self.server_socket = None
        self.helper_server_socket = None
        self.helper_socket = None
        self.node_handlers = []
    
    def start(self) -> None:
        """Start listening for nodes and helper."""
        # Main server for nodes
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(10)
        
        # Coordination server for helper
        self.helper_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.helper_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.helper_server_socket.bind((self.helper_host, self.helper_port))
        self.helper_server_socket.listen(1)
    
    def wait_for_helper(self) -> None:
        """Wait for helper to connect."""
        self.helper_socket, helper_address = self.helper_server_socket.accept()
        
        # Confirm helper is ready
        msg_type, payload = self._receive_message(self.helper_socket)
        assert msg_type == SecureMessageType.HELPER_READY
    
    def send_noise_config_to_helper(self, activation_noise: float, gradient_noise: float) -> None:
        """Send noise configuration to helper.
        
        Args:
            activation_noise: Noise scale for activations
            gradient_noise: Noise scale for gradients
        """
        self._send_message(self.helper_socket, SecureMessageType.HELPER_INIT, {
            'phase': 'noise_config',
            'activation_noise': activation_noise,
            'gradient_noise': gradient_noise
        })
    
    def accept_node(self) -> 'SecureNodeHandler':
        """Accept a node connection."""
        node_socket, node_address = self.server_socket.accept()
        handler = SecureNodeHandler(node_socket, node_address, len(self.node_handlers))
        self.node_handlers.append(handler)
        return handler
    
    def broadcast_init(self, activation_noise: float = None, gradient_noise: float = None, **kwargs) -> None:
        """Broadcast initialization to all nodes.
        
        Args:
            activation_noise: Noise scale for activations (secure mode)
            gradient_noise: Noise scale for gradients (secure mode)
            **kwargs: Configuration parameters (node_model, etc.)
        """
        for handler in self.node_handlers:
            config = {
                'node_id': handler.node_id,
                **kwargs
            }
            # Add noise config for secure mode
            if activation_noise is not None and gradient_noise is not None:
                config['activation_noise'] = activation_noise
                config['gradient_noise'] = gradient_noise
            handler.send_init(config)
    
    def collect_dataset_sizes(self) -> List[int]:
        """Collect dataset sizes from all nodes."""
        return [handler.receive_dataset_size() for handler in self.node_handlers]
    
    def broadcast_batch_assignment(self, model_state: dict, 
                                   sample_indices_per_node: list) -> None:
        """Broadcast batch assignments to all nodes."""
        for handler, sample_indices in zip(self.node_handlers, sample_indices_per_node):
            handler.send_batch_assignment(model_state, sample_indices)
    
    def collect_forward_results(self) -> List[dict]:
        """Collect forward results (shares) from all nodes."""
        return [handler.receive_forward_result() for handler in self.node_handlers]
    
    def broadcast_backward_signal(self, gradients_per_node: list) -> None:
        """Broadcast backward signals (shares) to all nodes."""
        for handler, gradients in zip(self.node_handlers, gradients_per_node):
            handler.send_backward_signal(gradients)
    
    def collect_gradient_results(self) -> List[dict]:
        """Collect gradient results (shares) from all nodes."""
        return [handler.receive_gradient_result() for handler in self.node_handlers]
    
    def close(self) -> None:
        """Close all connections."""
        for handler in self.node_handlers:
            handler.close()
        if self.helper_socket:
            self.helper_socket.close()
        if self.helper_server_socket:
            self.helper_server_socket.close()
        if self.server_socket:
            self.server_socket.close()


# ==============================================================================
# NODE HANDLERS
# ==============================================================================


class NodeHandler(SocketCommunicator):
    """Handler for individual node connection (standard mode).
    
    Wraps socket communication for a single node with typed methods.
    """
    
    def __init__(self, sock: socket.socket, address: Tuple[str, int], node_id: int):
        """Initialize node handler.
        
        Args:
            sock: Connected socket
            address: Node address (host, port)
            node_id: Unique identifier (0-indexed, converted to 1-indexed)
        """
        self.socket = sock
        self.address = address
        self.node_id = node_id + 1  # Convert to 1-indexed for user-facing ID
    
    def send_init(self, config: dict) -> None:
        """Send initialization config."""
        self._send_message(self.socket, MessageType.INIT, config)
    
    def receive_dataset_size(self) -> int:
        """Receive dataset size."""
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.DATASET_SIZE
        return payload
    
    def send_batch_assignment(self, model_state: dict, sample_indices: Any) -> None:
        """Send batch assignment."""
        payload = {
            'model_state': model_state,
            'sample_indices': sample_indices
        }
        self._send_message(self.socket, MessageType.BATCH_ASSIGNMENT, payload)
    
    def receive_forward_result(self) -> dict:
        """Receive forward result."""
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.FORWARD_RESULT
        return payload
    
    def send_backward_signal(self, gradients: Any) -> None:
        """Send backward signal."""
        self._send_message(self.socket, MessageType.BACKWARD_SIGNAL, gradients)
    
    def receive_gradient_result(self) -> dict:
        """Receive gradient result."""
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == MessageType.GRADIENT_RESULT
        return payload
    
    def close(self) -> None:
        """Close connection."""
        self.socket.close()


class SecureNodeHandler(SecureSocketCommunicator):
    """Handler for individual node connection (secure mode).
    
    Identical to NodeHandler but uses SecureMessageType.
    """
    
    def __init__(self, sock: socket.socket, address: Tuple[str, int], node_id: int):
        """Initialize secure node handler."""
        self.socket = sock
        self.address = address
        self.node_id = node_id + 1  # 1-indexed
    
    def send_init(self, config: dict) -> None:
        """Send initialization config."""
        self._send_message(self.socket, SecureMessageType.INIT, config)
    
    def receive_dataset_size(self) -> int:
        """Receive dataset size."""
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.DATASET_SIZE
        return payload
    
    def send_batch_assignment(self, model_state: dict, sample_indices: Any) -> None:
        """Send batch assignment."""
        payload = {
            'model_state': model_state,
            'sample_indices': sample_indices
        }
        self._send_message(self.socket, SecureMessageType.BATCH_ASSIGNMENT, payload)
    
    def receive_forward_result(self) -> dict:
        """Receive forward result (share)."""
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.FORWARD_RESULT
        return payload
    
    def send_backward_signal(self, gradients: Any) -> None:
        """Send backward signal (share)."""
        self._send_message(self.socket, SecureMessageType.BACKWARD_SIGNAL, gradients)
    
    def receive_gradient_result(self) -> dict:
        """Receive gradient result (share)."""
        msg_type, payload = self._receive_message(self.socket)
        assert msg_type == SecureMessageType.GRADIENT_RESULT
        return payload
    
    def close(self) -> None:
        """Close connection."""
        self.socket.close()
