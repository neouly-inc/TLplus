"""CNN model for TL++."""

import torch
import torch.nn as nn
from typing import List, Dict


# ==============================================================================
# NODE-SIDE MODEL
# ==============================================================================


class CNNNode(nn.Module):
    """Node-side portion of the CNN.
    
    This class represents the layers that run on the node (edge device).
    The model is cut at a specified layer, and only layers up to that point
    are included here. This allows computation to be distributed between
    the node and the orchestrator.
    """
    
    def __init__(self, cut_layer: int = 1):
        """Initialize node-side model.
        
        Args:
            cut_layer: Where to cut the model (1, 2, or 3)
                      1: After first conv block (64 channels)
                      2: After second conv block (128 channels)
                      3: After third conv block + fc1 (512 features)
        
        Raises:
            TypeError: If cut_layer is not an integer
            ValueError: If cut_layer is not in range 1-3
        """
        super(CNNNode, self).__init__()
        
        # Validate input type
        if not isinstance(cut_layer, int):
            raise TypeError(f"cut_layer must be int, got {type(cut_layer).__name__}")
        
        # Validate cut_layer parameter
        if not 1 <= cut_layer <= 3:
            raise ValueError(f"cut_layer must be 1-3, got {cut_layer}")
        
        self.cut_layer = cut_layer
        
        # Block 1: 3 -> 64 -> 64 (spatial: 32x32 -> 16x16) - always included
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 64 -> 128 -> 128 (spatial: 16x16 -> 8x8)
        if cut_layer >= 2:
            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 128 -> 256 -> 256 (spatial: 8x8 -> 4x4)
        if cut_layer >= 3:
            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Classifier: flatten(256 * 4 * 4) -> 512 -> dropout
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
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
        """Forward pass up to the cut layer.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
        
        Returns:
            Activations at the cut layer (shape depends on cut_layer)
        """
        # Block 1: Conv-ReLU-Conv-ReLU-Pool (always executed)
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.pool1(x)  # 32x32 -> 16x16
        
        # Block 2: Conv-ReLU-Conv-ReLU-Pool (if cut_layer >= 2)
        if self.cut_layer >= 2:
            x = torch.relu(self.conv2_1(x))
            x = torch.relu(self.conv2_2(x))
            x = self.pool2(x)  # 16x16 -> 8x8
        
        # Block 3: Conv-ReLU-Conv-ReLU-Pool + FC (if cut_layer >= 3)
        if self.cut_layer >= 3:
            x = torch.relu(self.conv3_1(x))
            x = torch.relu(self.conv3_2(x))
            x = self.pool3(x)  # 8x8 -> 4x4
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
        
        return x
    
    def get_state_names(self) -> tuple:
        """Get names of parameters and buffers in this model.
        
        Returns:
            Tuple of (parameter_names, buffer_names)
        """
        param_names = [name for name, _ in self.named_parameters()]
        buffer_names = [name for name, _ in self.named_buffers()]
        return param_names, buffer_names


# ==============================================================================
# ORCHESTRATOR-SIDE MODEL
# ==============================================================================


class CNNOrchestrator(nn.Module):
    """Orchestrator-side model (full CNN).
    
    This class contains the complete model architecture. During training,
    it receives activations from the cut point and continues the forward pass.
    It also maintains a copy of the node-side layers for weight sharing.
    """
    
    def __init__(self, num_classes: int = 10, cut_layer: int = 1):
        """Initialize orchestrator-side model.
        
        Args:
            num_classes: Number of output classes (typically equals number of nodes)
            cut_layer: Where the model is cut between node and orchestrator
        
        Raises:
            TypeError: If num_classes or cut_layer is not an integer
            ValueError: If cut_layer is not in range 1-3 or num_classes not in 1-10
        """
        super(CNNOrchestrator, self).__init__()
        
        # Validate input types
        if not isinstance(num_classes, int):
            raise TypeError(f"num_classes must be int, got {type(num_classes).__name__}")
        if not isinstance(cut_layer, int):
            raise TypeError(f"cut_layer must be int, got {type(cut_layer).__name__}")
        
        # Validate parameters
        if not 1 <= cut_layer <= 3:
            raise ValueError(f"cut_layer must be 1-3, got {cut_layer}")
        if not 1 <= num_classes <= 10:
            raise ValueError(f"num_classes must be 1-10, got {num_classes}")
        
        self.cut_layer = cut_layer
        self.num_classes = num_classes
        
        # Define full model architecture
        # Block 1: 3 -> 64 -> 64
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 64 -> 128 -> 128
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3: 128 -> 256 -> 256
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Initialize weights for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization.
        
        Xavier initialization helps with gradient flow in deep networks.
        It scales initial weights based on the number of input/output units.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Initialize convolutional layer weights
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Initialize fully connected layer weights
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through the entire network.
        
        Used during evaluation when we have direct access to input images.
        
        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32]
        
        Returns:
            Output logits of shape [batch_size, num_classes]
        """
        # Block 1: Conv-ReLU-Conv-ReLU-Pool
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.pool1(x)  # 32x32 -> 16x16
        
        # Block 2: Conv-ReLU-Conv-ReLU-Pool
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = self.pool2(x)  # 16x16 -> 8x8
        
        # Block 3: Conv-ReLU-Conv-ReLU-Pool
        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = self.pool3(x)  # 8x8 -> 4x4
        
        # Classifier: Flatten-FC-ReLU-Dropout-FC
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def forward_from_cut(self, activations: torch.Tensor) -> torch.Tensor:
        """Forward pass starting from the cut point.
        
        Used during distributed training when receiving activations from nodes.
        
        Args:
            activations: Activations from the cut layer, shape depends on cut_layer
        
        Returns:
            Output logits of shape [batch_size, num_classes]
        """
        x = activations
        
        # If cut at layer 1, need to process blocks 2 and 3
        if self.cut_layer == 1:
            x = torch.relu(self.conv2_1(x))
            x = torch.relu(self.conv2_2(x))
            x = self.pool2(x)
            x = torch.relu(self.conv3_1(x))
            x = torch.relu(self.conv3_2(x))
            x = self.pool3(x)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
        
        # If cut at layer 2, need to process block 3
        elif self.cut_layer == 2:
            x = torch.relu(self.conv3_1(x))
            x = torch.relu(self.conv3_2(x))
            x = self.pool3(x)
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
        
        # Always process final layer
        x = self.fc2(x)
        
        return x
    
    def get_node_state(self) -> Dict[str, torch.Tensor]:
        """Extract state dict for node-side layers.
        
        Returns a dictionary containing only the parameters and buffers
        that belong to layers executed on the node side.
        
        Returns:
            Dictionary mapping parameter/buffer names to tensors
        """
        # Define which layers belong to each cut level
        layer_groups = {
            1: ['conv1_1', 'conv1_2'],
            2: ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'],
            3: ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'fc1'],
        }
        node_layers = set(layer_groups[self.cut_layer])
        
        state = {}
        
        # Extract parameters (weights and biases)
        for name, param in self.named_parameters():
            layer_name = name.split('.')[0]  # Get layer name (e.g., 'conv1_1' from 'conv1_1.weight')
            if layer_name in node_layers:
                # Detach and move to CPU for transmission
                state[name] = param.detach().cpu()
        
        # Extract buffers (e.g., running stats for BatchNorm, though not used here)
        for name, buffer in self.named_buffers():
            layer_name = name.split('.')[0]
            if layer_name in node_layers:
                state[name] = buffer.detach().cpu()
        
        return state
    
    def get_node_param_names(self) -> List[str]:
        """Get list of parameter names that belong to node-side layers.
        
        Used for gradient aggregation to know which gradients to collect from nodes.
        
        Returns:
            List of parameter names (e.g., ['conv1_1.weight', 'conv1_1.bias', ...])
        """
        # Define which layers belong to each cut level
        layer_groups = {
            1: ['conv1_1', 'conv1_2'],
            2: ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2'],
            3: ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'fc1'],
        }
        node_layers = set(layer_groups[self.cut_layer])
        
        # Filter parameter names to only include node-side layers
        return [name for name, _ in self.named_parameters()
                if name.split('.')[0] in node_layers]


# ==============================================================================
# FACTORY FUNCTIONS
# ==============================================================================


def create_node_model(cut_layer: int) -> CNNNode:
    """Factory function to create a node model.
    
    Args:
        cut_layer: Where to cut the model (1, 2, or 3)
    
    Returns:
        Initialized CNNNode instance
    
    Raises:
        TypeError: If cut_layer is not an integer
        ValueError: If cut_layer is not in valid range
    """
    return CNNNode(cut_layer=cut_layer)


def create_orchestrator_model(num_classes: int, cut_layer: int) -> CNNOrchestrator:
    """Factory function to create an orchestrator model.
    
    Args:
        num_classes: Number of output classes
        cut_layer: Where to cut the model (1, 2, or 3)
    
    Returns:
        Initialized CNNOrchestrator instance
    
    Raises:
        TypeError: If parameters are not integers
        ValueError: If parameters are not in valid ranges
    """
    return CNNOrchestrator(num_classes=num_classes, cut_layer=cut_layer)