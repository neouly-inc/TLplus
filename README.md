# TL++ | VGG-Style CNN on CIFAR-10

TL++ is a distributed deep learning framework that enables collaborative model training across multiple low-resource edge devices (nodes) while preserving data privacy. It combines traversal and split learning strategies with secure multiparty computation (MPC) to protect sensitive intermediate activations and gradients during training.

---

### 🔑 Key Advantages

**Privacy-Preserving by Design**

* ✅ Raw data never leaves edge devices (nodes)
* ✅ Intermediate activations and gradients protected using additive secret sharing
* ✅ Semi-honest security model with non-colluding orchestrator and helper node
* ✅ Configurable privacy-utility trade-off via noise parameters

**Distributed & Efficient**

* ⚡ Splits model computation between node and orchestrator at a configurable cut point
* ⚡ Minimizes communication overhead with strategic cut layer selection
* ⚡ Supports heterogeneous devices (GPU / CPU / MPS)
* ⚡ Robust to skewed non-IID data and heterogeneous data distributions

**Flexible Deployment**

* 🔄 Base mode for trusted environments (faster training, no MPC overhead)
* 🔒 Secure mode for privacy-critical applications (MPC-based)
* 🎯 Configurable model split points across 3 convolutional blocks
* 🌐 Single-machine or multi-machine deployment

**Production-Ready**

* 📊 Comprehensive logging and monitoring
* 🛡️ Gradient clipping and early stopping
* 🔧 Multiple learning rate schedulers
* ⚙️ Extensive hyperparameter control via CLI

### Use Cases

* **Healthcare**: Collaborative model training across hospitals without sharing patient data
* **Finance**: Multi-institutional fraud detection while preserving transaction privacy
* **IoT**: Edge device training with bandwidth-constrained communication
* **Mobile**: On-device learning with privacy guarantees

---

### How It Works

![Architecture Diagram](./figure/architecture.svg)

---

### Architecture Components

**Nodes (Edge Devices)**

* Store and process local private data
* Execute the bottom layers of the split model
* Send encrypted activation shares (secure mode) or raw activations (base mode) to the orchestrator
* Compute gradients for local model parameters

**Orchestrator (Central Server)**

* Coordinates distributed training across all nodes
* Executes the top layers of the split model
* Aggregates gradients and updates the global model parameters
* In secure mode: processes only one share, never sees raw activations

**Helper (Secure Mode Only)**

* Independent, non-colluding server for the two-server MPC protocol
* Processes the second share of all activations
* Enables privacy-preserving computation without reconstruction
* Critical for maintaining privacy guarantees

---

## 🧠 CNN Model

TL++ uses a custom VGG-style CNN defined in `models.py`, designed for CIFAR-10 (32×32 RGB images).

### Full Architecture

* **Block 1:** Two Conv2D layers (3→64→64 channels, 3×3 kernels, padding=1), each followed by ReLU, then 2×2 MaxPool — output: `64×16×16`
* **Block 2:** Two Conv2D layers (64→128→128 channels), each followed by ReLU, then 2×2 MaxPool — output: `128×8×8`
* **Block 3:** Two Conv2D layers (128→256→256 channels), each followed by ReLU, then 2×2 MaxPool — output: `256×4×4`
* **Classifier:** Flatten → FC(4096→512) → ReLU → Dropout(0.5) → FC(512→num_classes)

All weights are initialized with Xavier uniform initialization for stable gradient flow.

### Model Split by Cut Layer

| `cut_layer` | Node-side layers | Orchestrator-side layers | Communication tensor shape |
|:-----------:|:----------------:|:------------------------:|:--------------------------:|
| 1 (default) | Block 1 | Blocks 2–3 + Classifier | `[B, 64, 16, 16]` |
| 2 | Blocks 1–2 | Block 3 + Classifier | `[B, 128, 8, 8]` |
| 3 | Blocks 1–3 | Classifier | `[B, 512]` |

The split is implemented via two classes: `CNNNode` (edge-side) and `CNNOrchestrator` (server-side), which share the same underlying architecture.

---

## 🚀 Installation

### Prerequisites

* Python 3.8+
* PyTorch 2.0+
* torchvision
* NumPy
* tqdm

### Setup

```bash
# Clone the repository
git clone https://github.com/neouly-inc/TLplus.git
cd TLplus

# Install dependencies
pip install torch torchvision numpy tqdm

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### GPU Support (Optional)

```bash
# For NVIDIA GPUs (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (MPS) — PyTorch 2.0+ includes MPS support
# No additional installation needed
```

---

## ⚡ Quick Start

### 1. Base Mode (Single Machine, 3 Nodes)

```bash
# Terminal 1 — Orchestrator
python orchestrator.py --n_nodes 3 --cut_layer 1

# Terminal 2 — Node 1
python node.py

# Terminal 3 — Node 2
python node.py

# Terminal 4 — Node 3
python node.py
```

### 2. Secure Mode (Single Machine)

```bash
# Terminal 1 — Orchestrator
python orchestrator.py --n_nodes 2 --cut_layer 1 --secure

# Terminal 2 — Helper
python helper.py --n_nodes 2

# Terminal 3 — Node 1
python node.py --secure

# Terminal 4 — Node 2
python node.py --secure
```

### 3. Centralized Baseline

```bash
python centralized.py \
  --num_classes 10 \
  --epochs 200 \
  --lr 0.1 \
  --scheduler cosine
```

---

## 📘 Base Mode

Base mode provides direct communication between the orchestrator and nodes without any cryptographic privacy protection. Suitable for trusted environments or ablation studies.

### Orchestrator

```bash
python orchestrator.py \
  --n_nodes 3 \
  --cut_layer 1 \
  --epochs 200 \
  --train_batch_size 128 \
  --lr 0.1
```

### Nodes

```bash
# Each node auto-connects and receives configuration from the orchestrator
python node.py --host 127.0.0.1 --port 8080
```

### Distributed Setup (Multiple Machines)

#### On the Server (Orchestrator)

```bash
python orchestrator.py \
  --host 0.0.0.0 \
  --port 8080 \
  --n_nodes 3
```

#### On Edge Devices (Nodes)

```bash
# Replace SERVER_IP with the orchestrator's IP address
python node.py --host SERVER_IP --port 8080
```

### Cut Layer Selection

| Cut Layer | Node runs | Orchestrator runs | Best for |
|:---------:|:---------:|:-----------------:|:---------|
| 1 (default) | Block 1 | Blocks 2–3 + Classifier | Bandwidth-constrained scenarios |
| 2 | Blocks 1–2 | Block 3 + Classifier | Balanced computation split |
| 3 | Blocks 1–3 | Classifier | Maximum privacy (most computation on node) |

```bash
python orchestrator.py --cut_layer 2 --n_nodes 3
```

---

## 🔒 Secure Mode

Secure mode uses two-server multi-party computation (MPC) with additive secret sharing to protect intermediate activations between the edge devices and the servers.

### Security Model

* **Threat Model**: Semi-honest (honest-but-curious) adversaries
* **Assumption**: Orchestrator and helper server do not collude
* **Protection**: Intermediate activations are secret-shared; neither server alone can reconstruct them

### Setup

#### 1. Start Orchestrator

```bash
python orchestrator.py \
  --secure \
  --n_nodes 2 \
  --host 0.0.0.0 \
  --port 8080 \
  --helper_host 0.0.0.0 \
  --helper_port 8082
```

#### 2. Start Helper Server

```bash
python helper.py \
  --n_nodes 2 \
  --host 0.0.0.0 \
  --port 8081 \
  --orch_host ORCHESTRATOR_IP \
  --orch_port 8082
```

#### 3. Start Nodes

```bash
python node.py \
  --secure \
  --orch_host ORCHESTRATOR_IP \
  --orch_port 8080 \
  --helper_host HELPER_IP \
  --helper_port 8081
```

---

## 📊 Centralized Learning Baseline

`centralized.py` provides a standard centralized training baseline for fair performance comparison against TL++. It trains the identical CNN architecture on the complete CIFAR-10 dataset using a single machine with full data access — no distribution, no split learning, and no MPC overhead.

The baseline uses the same hyperparameters as TL++ by default (SGD with momentum, cosine annealing scheduler, gradient clipping, early stopping) and applies the same data augmentation pipeline (random crop, random horizontal flip, normalization). This ensures that any accuracy gap between centralized and distributed results reflects the cost of privacy and distribution rather than differences in training setup.

```bash
python centralized.py \
  --num_classes 10 \
  --epochs 200 \
  --lr 0.1 \
  --scheduler cosine
```

---

## ⚙️ Configuration

### Orchestrator Options

```
python orchestrator.py --help
```

**Network:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--host` | `127.0.0.1` | Bind address for node connections |
| `--port` | `8080` | Port for node connections |
| `--secure` | `False` | Enable secure MPC mode |
| `--helper_host` | `127.0.0.1` | Helper server host (secure mode) |
| `--helper_port` | `8082` | Helper coordination port (secure mode) |

**Secure Mode Privacy:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--activation_noise` | `0.02` | Gaussian noise scale for activation shares |
| `--gradient_noise` | `0.10` | Gaussian noise scale for gradient shares |

**Model:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--cut_layer` | `1` | CNN split point: 1, 2, or 3 |
| `--n_nodes` | `1` | Number of edge nodes |

**Training:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--epochs` | `200` | Maximum training epochs |
| `--patience` | `30` | Early stopping patience |
| `--train_batch_size` | `128` | Training batch size |
| `--test_batch_size` | `256` | Evaluation batch size |

**Optimisation:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--lr` | `0.1` | SGD learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--weight_decay` | `5e-4` | L2 regularisation |
| `--nesterov` / `--no_nesterov` | enabled | Nesterov momentum |
| `--grad_clip_norm` | `1.0` | Gradient clipping norm (0 = disabled) |

**Scheduler:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--scheduler` | `cosine` | LR scheduler: `cosine`, `step`, or `multistep` |
| `--t_max` | epochs | Cosine annealing T_max |
| `--eta_min` | `1e-6` | Cosine annealing minimum LR |
| `--step_size` | `50` | Step scheduler step size |
| `--gamma` | `0.1` | LR decay factor |
| `--milestones` | `[60,120,160]` | MultiStep scheduler milestone epochs |

**Hardware:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--no_accel` | `False` | Force CPU (disable CUDA / MPS) |

---

### Node Options

```
python node.py --help
```

**Base Mode:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--host` | `127.0.0.1` | Orchestrator host |
| `--port` | `8080` | Orchestrator port |

**Secure Mode:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--secure` | `False` | Enable secure MPC mode |
| `--orch_host` | `127.0.0.1` | Orchestrator host |
| `--orch_port` | `8080` | Orchestrator port |
| `--helper_host` | `127.0.0.1` | Helper server host |
| `--helper_port` | `8081` | Helper server port |

**Hardware:**

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--no_accel` | `False` | Force CPU |

---

### Helper Options

```
python helper.py --help
```

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--host` | `127.0.0.1` | Bind address for node connections |
| `--port` | `8081` | Port for node connections |
| `--orch_host` | `127.0.0.1` | Orchestrator host |
| `--orch_port` | `8082` | Orchestrator coordination port |
| `--n_nodes` | — | Expected number of training nodes (required) |

---

### Centralized Baseline Options

```
python centralized.py --help
```

| Flag | Default | Description |
|:-----|:-------:|:------------|
| `--num_classes` | `10` | Number of output classes |
| `--epochs` | `200` | Maximum training epochs |
| `--patience` | `30` | Early stopping patience |
| `--lr` | `0.1` | SGD learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--weight_decay` | `5e-4` | L2 regularisation |
| `--scheduler` | `cosine` | LR scheduler: `cosine`, `step`, or `multistep` |
| `--grad_clip_norm` | `1.0` | Gradient clipping norm (0 = disabled) |
| `--no_accel` | `False` | Force CPU |

---

## 📝 License

This project is licensed under the MIT License.
