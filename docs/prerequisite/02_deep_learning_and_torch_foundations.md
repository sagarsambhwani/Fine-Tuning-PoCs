# 🧱 Fine-Tuning Prerequisites: Deep Learning & PyTorch Foundations

> **Level 0 → Level 1 Prerequisite Guide**: This document covers the essential mathematical and computational foundations required before studying transformer architectures and LLM fine-tuning. We focus strictly on the core building blocks: **tensors, matrix multiplication, neural network mechanics, and PyTorch essentials**.

---

## 1. Tensors & Dimensions

In deep learning and natural language processing, all text, weights, activations, and gradients are stored in multi-dimensional numerical arrays called **Tensors**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   TENSOR TAXONOMY                                      │
│                                                                                        │
│  0D: Scalar  ──► A single number: 42.0                                                 │
│  1D: Vector  ──► A 1D list of numbers: [1.2, -0.4, 3.8] (e.g., a single token embedding)│
│  2D: Matrix  ──► A 2D grid: [[1, 2], [3, 4]] (e.g., a linear layer weight matrix)     │
│  3D: Tensor  ──► A 3D block (e.g., standard LLM sequence batch: [Batch, Seq, Hidden]) │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### The 3 Core Dimensions of Transformer Tensors

When training or fine-tuning language models, you will constantly encounter 3D tensors formatted as:

$$\mathbf{X} \in \mathbb{R}^{\text{Batch Size} \times \text{Sequence Length} \times \text{Hidden Dimension}}$$

```
Example Shape: (2, 128, 768)
  ├── 2   = Batch Size (We are processing 2 distinct text sentences simultaneously)
  ├── 128 = Sequence Length (Each sentence contains up to 128 tokens)
  └── 768 = Hidden Dimension (Every single token is represented by a 768-dimensional vector)
```

```
               Sequence Length (128 Tokens)
             ┌─────────────────────────────┐
             │ [Token 1 Embedding (768)]   │
Batch Item 1 │ [Token 2 Embedding (768)]   │ ──► Shape: (128, 768)
             │ ...                         │
             │ [Token 128 Embedding (768)] │
             └─────────────────────────────┘
             ┌─────────────────────────────┐
Batch Item 2 │ [Token 1 Embedding (768)]   │ ──► Shape: (128, 768)
             │ ...                         │
             └─────────────────────────────┘
Total Batch Shape = (2, 128, 768)
```

---

## 2. Matrix Multiplication: The Engine of Transformers

Before you can understand the Attention formula ($\text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V$) or LoRA ($B \cdot A$), you must master matrix multiplication dimensions.

### A. The Dimension Matching Rule
To multiply matrix $\mathbf{A}$ of shape $(m \times k)$ with matrix $\mathbf{B}$ of shape $(k \times n)$:
1. The **inner dimensions must match** ($k = k$).
2. The resulting matrix $\mathbf{C} = \mathbf{A} \cdot \mathbf{B}$ has shape **$(m \times n)$**.

```
    Matrix A (m × k)         Matrix B (k × n)         Output Matrix C (m × n)
  ┌───────────────────┐    ┌───────────────────┐    ┌─────────────────────────┐
m │                   │  × │ k                 │  = │ m                       │
  │         k         │  k │         n         │    │            n            │
  └───────────────────┘    └───────────────────┘    └─────────────────────────┘
        └── Inner dimensions must match (k) ──┘        └── Result shape: (m × n) ─┘
```

### B. Linear Projections in Neural Networks
When a token vector $\mathbf{x} \in \mathbb{R}^{1 \times d_{\text{in}}}$ passes through a Linear Layer with weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:

$$\mathbf{y} = \mathbf{x} \cdot \mathbf{W}$$

* **Input shape**: $(1 \times 768)$
* **Weight matrix shape**: $(768 \times 2048)$
* **Output shape**: $(1 \times 2048)$ (The layer projected the 768-dim vector into a 2048-dim space).

---

## 3. Neural Network Mechanics: How Models Learn

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                THE UNIVERSAL TRAINING LOOP                             │
│                                                                                        │
│  [Weights W] ──► [Forward Pass] ──► [Loss Calculation] ──► [Backpropagation (Gradients)]│
│       ▲                                                              │                 │
│       └────────────── [Optimizer Update (AdamW / SGD)] ──────────────┘                 │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Terminology:

1. **Parameters / Weights ($\mathbf{W}$)**: The internal learnable floating-point numbers of the model that store knowledge.
2. **Forward Pass**: Passing input tokens through the network layers to produce output predictions (logits).
3. **Loss Function ($\mathcal{L}$)**: A mathematical measure of how wrong the model's prediction is compared to the ground-truth target.
4. **Gradients ($\nabla_{\mathbf{W}} \mathcal{L}$)**: The partial derivatives indicating **how much each weight contributed to the error** and in which direction the weight should move to decrease the loss.
5. **Backpropagation**: The application of the calculus Chain Rule from the output loss layer backward through all hidden layers to calculate gradients for every parameter.
6. **Optimizer (e.g., AdamW, SGD)**: The algorithm that adjusts the weights using the calculated gradients:
   $$\mathbf{W}_{\text{new}} = \mathbf{W}_{\text{old}} - \eta \cdot \nabla_{\mathbf{W}} \mathcal{L}$$
7. **Learning Rate ($\eta$)**: A small scalar (e.g., $1 \times 10^{-4}$) controlling the step size of each weight update.
8. **Batch**: A small subset of training examples processed together in parallel.
9. **Epoch**: One complete pass through the entire training dataset.

---

## 4. PyTorch Essentials for LLMs

In real-world LLM engineering and interviews, you must understand the exact PyTorch mechanics behind model training:

```python
import torch
import torch.nn as nn

# 1. Define a minimal custom Neural Network Module
class MinimalClassifier(nn.Module):
    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super().__init__()
        # Linear projection layer (Weights: 768 x 2)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states shape: [batch_size, hidden_size]
        logits = self.classifier(hidden_states)
        return logits

# 2. Instantiate Model, Loss Function, and Optimizer
model = MinimalClassifier(hidden_size=768, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 3. Simulate a single training step
# Dummy input: 4 examples of 768-dim embeddings
inputs = torch.randn(4, 768)
# Ground-truth class labels (0 or 1)
targets = torch.tensor([1, 0, 1, 1])

# --- THE CANONICAL PYTORCH TRAINING LOOP ---
# Step A: Reset gradients from previous step (prevents accumulation)
optimizer.zero_grad()

# Step B: Forward Pass (compute predictions)
logits = model(inputs)

# Step C: Compute Loss (difference between predictions and targets)
loss = criterion(logits, targets)
print(f"Step Loss: {loss.item():.4f}")

# Step D: Backward Pass (calculate gradients via Backpropagation)
loss.backward()

# Step E: Optimizer Step (update model weights using gradients)
optimizer.step()
```

### Why Does This Matter for Fine-Tuning?
When you fine-tune open-source models using Hugging Face `transformers` or `TRL`, under the hood the trainer executes this exact sequence:
```python
logits = model(input_ids)
loss = loss_fn(logits, labels)
loss.backward()
optimizer.step()
```
When debugging training crashes (such as `loss = NaN`, gradient explosion, or out-of-memory errors), you will always trace the issue back to one of these 5 steps!
