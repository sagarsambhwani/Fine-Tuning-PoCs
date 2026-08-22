# ⚡ Module 02: PEFT — LoRA, QLoRA, DoRA & Parameter Efficiency

Parameter-Efficient Fine-Tuning (PEFT) is the industrial standard for training multi-billion parameter models on accessible hardware without sacrificing generation quality. This module dives into the mathematical derivations, memory profiles, quantization mechanisms, and architectural implementations of modern PEFT methods.

---

## 1. The PEFT Landscape & Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     PEFT TAXONOMY                                       │
│                                           │                                             │
│         ┌─────────────────────────────────┼─────────────────────────────────┐           │
│         ▼                                 ▼                                 ▼           │
│  [Reparameterization]            [Additive Adapters]               [Selective Tuning]   │
│  • LoRA (Low-Rank Adaptation)    • Prefix Tuning                   • BitFit (Bias only) │
│  • QLoRA (Quantized LoRA)        • Prompt Tuning (Soft prompts)    • Freeze top layers  │
│  • DoRA (Weight-Decomposed)      • Bottleneck Series Adapters      • Sparse parameter   │
│  • AdaLoRA (Dynamic SVD Rank)    • (IA)³ (Inhibited/Amplified)                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. LoRA: Mathematical Formulation & Mechanics

*Reference: Hu et al., 2021 ("LoRA: Low-Rank Adaptation of Large Language Models")*

### A. The Intrinsic Rank Hypothesis
Research by *Aghajanyan et al. (2020)* demonstrated that over-parameterized models reside on a **low-dimensional intrinsic manifold**. The weight updates $\Delta W$ during task adaptation have a very low intrinsic dimension (rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$).

### B. Mathematical Formulation

For a pre-trained frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

Where:
* $W_0 \in \mathbb{R}^{d \times k}$ (Frozen original weights)
* $A \in \mathbb{R}^{r \times k}$ (Down-projection adapter matrix)
* $B \in \mathbb{R}^{d \times r}$ (Up-projection adapter matrix)
* $r \ll \min(d, k)$ (Rank, typically $r \in \{8, 16, 32, 64\}$)
* $\alpha$ (Scaling hyperparameter, constant multiplier)

```
        Input Vector x (d_in)
          ┌───────┴───────┐
          │               │
          ▼               ▼
    [ Frozen W_0 ]     [ Matrix A (r × d_in) ]  <-- Initialized from N(0, σ²)
   (d_out × d_in)         │
          │               ▼
          │            [ Matrix B (d_out × r) ]  <-- Initialized to ALL ZEROS
          │               │
          │               ▼
          │            Scale: × (α / r)
          │               │
          ▼               ▼
         (+) ◄────────────┘
          │
          ▼
     Output Vector h (d_out)
```

### C. Critical Initialization & Scaling Factor Dynamics

1. **Why is Matrix $B$ initialized to $0$ and $A$ to Gaussian $\mathcal{N}(0, \sigma^2)$?**
   $$\Delta W = B \cdot A = 0 \cdot A = 0 \quad \text{at step } 0$$
   This ensures that at the start of training, the model's exact pre-trained behavior is preserved ($h = W_0 x + 0 = W_0 x$). If both were initialized to random values, initial outputs would be degraded by random noise.

2. **The Role of $\alpha$ and the $\frac{\alpha}{r}$ Scaling Factor**:
   * When tuning hyperparameter $r$, changing $r$ without scaling alters the learning dynamics.
   * Keeping $\frac{\alpha}{r}$ constant (commonly setting $\alpha = 2r$, so scaling is $2.0$) allows you to experiment with different ranks without having to drastically retune the learning rate.

---

## 3. Custom LoRA Implementation in Pure PyTorch

Writing a custom LoRA layer from scratch is a favorite coding task in Senior AI Engineer interviews.

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 1. Base frozen weight layer
        self.base_layer = nn.Linear(in_features, out_features, bias=False)
        self.base_layer.weight.requires_grad = False  # Freeze base weights

        # 2. Trainable Low-Rank Adapters
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # Zero init
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform / Gaussian init for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Zero init for B guarantees Delta W = 0 at step 0
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass
        result = self.base_layer(x)

        if self.rank > 0:
            # Low-rank forward: (x @ A.T @ B.T) * (alpha / r)
            lora_out = self.lora_dropout(x)
            lora_out = torch.matmul(lora_out, self.lora_A.t())  # [batch, seq, r]
            lora_out = torch.matmul(lora_out, self.lora_B.t())  # [batch, seq, out_features]
            result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Zero-latency inference: fuse delta weights directly into base matrix."""
        if self.rank > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.rank = 0  # Disable LoRA path
```

---

## 4. QLoRA Deep Dive: 4-Bit NormalFloat & Quantization Mechanics

*Reference: Dettmers et al., 2023 ("QLoRA: Efficient Finetuning of Quantized LLMs")*

QLoRA enables fine-tuning a **65B parameter model on a single 48GB GPU** or a **7B/14B model on a single 16GB–24GB consumer GPU** by introducing 3 key innovations:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE QLoRA TRIFECTA                                     │
│                                                                                        │
│  1. NF4 Quantization (NormalFloat4)  ──► Information-theoretically optimal for         │
│                                          normally distributed weights.                 │
│  2. Double Quantization (DQ)         ──► Quantizes quantization constants, saving      │
│                                          0.37 bits per parameter.                      │
│  3. Paged Optimizers                 ──► Pages memory to CPU RAM via CUDA Unified      │
│                                          Memory to eliminate OOM spikes.               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. NormalFloat4 (NF4)
Standard INT4 quantization uses evenly spaced buckets. However, neural network weights follow a **zero-mean Gaussian distribution** $\mathcal{N}(0, \sigma^2)$.

* **NF4 Construction**: Divides the standard normal distribution $\mathcal{N}(0, 1)$ into $2^4 = 16$ discrete bins such that **each bin contains an equal probability mass** ($q_i = \frac{1}{2} (Q_X(\frac{i}{2^k}) + Q_X(\frac{i+1}{2^k}))$).
* **Result**: Information-theoretically minimizes quantization error compared to uniform INT4 or FP4.

### B. Double Quantization (DQ)
Quantization requires a scaling factor $c_1^{\text{FP32}}$ for every block of $B_1 = 64$ parameters:
$$\text{Memory overhead of } c_1 = \frac{32\text{ bits}}{64\text{ params}} = 0.5\text{ bits/param}$$

Double Quantization performs a second 8-bit quantization on the quantization constants $c_1$ using block size $B_2 = 256$:
$$\text{Memory overhead of DQ} = \frac{8\text{ bits}}{64} + \frac{32\text{ bits}}{64 \times 256} = 0.125 + 0.00195 = 0.127\text{ bits/param}$$
* **Net Savings**: $0.5 - 0.127 \approx \mathbf{0.373\text{ bits per parameter}}$ ($\approx 3\text{ GB}$ saved on a 65B model).

### C. Computation Precision vs. Storage Precision
* **Storage Precision**: Base model weights are stored in **NF4 (4-bit)** in VRAM.
* **Computation Precision**: During the forward and backward passes, the required block of NF4 weights is **dequantized on-the-fly into BF16/FP16 directly in high-speed GPU SRAM cache**.
* The LoRA adapter matrices ($A$ and $B$) are maintained and updated in **BF16 / FP32**.

---

## 5. Modern PEFT Variants: DoRA, AdaLoRA, (IA)³

### A. DoRA (Weight-Decomposed Low-Rank Adaptation)
*Reference: Liu et al., 2024*

LoRA couples magnitude and directional updates together. In contrast, full fine-tuning alters magnitude and direction independently.

DoRA decomposes the weight matrix into **Magnitude** $m \in \mathbb{R}^{1 \times k}$ and **Direction** $V \in \mathbb{R}^{d \times k}$:

$$W = m \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_F} = m \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_F}$$

* **Advantage**: Bridges the performance gap between LoRA and Full Fine-Tuning with **zero extra inference latency** (decomposed weights merge back cleanly into base matrix).

### B. AdaLoRA (Adaptive Budget Allocation)
* Standard LoRA assigns uniform rank $r$ across all layers.
* AdaLoRA uses singular value decomposition ($P \Lambda Q$) and calculates an **importance metric** based on magnitude and gradient sensitivity:
  $$S_{i, j} = |w_{ij} \nabla_{w_{ij}} \mathcal{L}|$$
* Dynamically allocates higher rank to critical layers (e.g., intermediate cross-attention or MLP down-projection) while pruning zero-importance ranks from trivial layers.

---

## 6. GPU VRAM Consumption: The Complete Math Breakdown

| Component | Full Fine-Tuning (FP16/BF16) | Standard LoRA (BF16) | QLoRA (NF4 Base + BF16 LoRA) |
| :--- | :--- | :--- | :--- |
| **Model Weights ($W_0$)** | $2 \text{ bytes} \times N$ | $2 \text{ bytes} \times N$ | **$0.5 \text{ bytes} \times N$** |
| **LoRA Adapters ($A, B$)** | $0$ | $2 \text{ bytes} \times N_{\text{LoRA}}$ | $2 \text{ bytes} \times N_{\text{LoRA}}$ |
| **Gradients** | $2 \text{ bytes} \times N$ | $2 \text{ bytes} \times N_{\text{LoRA}}$ | $2 \text{ bytes} \times N_{\text{LoRA}}$ |
| **Optimizer States (AdamW)** | $12 \text{ bytes} \times N$ | $12 \text{ bytes} \times N_{\text{LoRA}}$ | $12 \text{ bytes} \times N_{\text{LoRA}}$ |
| **Total Static VRAM (7B Model)** | $\approx \mathbf{112 \text{ GB}}$ ($16 \times 7$) | $\approx \mathbf{14.5 \text{ GB}}$ | $\approx \mathbf{4.2 \text{ GB}}$ |

*(Note: Dynamic activation memory and KV cache are added on top depending on batch size, context length, and gradient checkpointing).*

---

## 🎯 Top Interview Q&A on PEFT & LoRA

### Q1: Should you apply LoRA to only Attention weights ($W_q, W_v$) or All Linear layers (including MLP $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$)?
**Answer**:
The original LoRA paper (2021) only applied adapters to $W_q$ and $W_v$ due to compute constraints. However, modern empirical research (e.g., QLoRA paper) proves that **applying LoRA to all linear layers (Q, K, V, O, Gate, Up, Down) at lower rank (e.g., $r=16$) consistently outperforms applying higher rank ($r=64$) to attention layers alone**. The MLP layers store factual associations and domain representations, making them essential for knowledge-intensive fine-tuning.

### Q2: What is the exact relationship between rank $r$, alpha $\alpha$, and the learning rate?
**Answer**:
The effective learning rate for the LoRA adapter update is scaled by $\frac{\alpha}{r}$:
$$\Delta W = \frac{\alpha}{r} (B \cdot A) \implies \text{Effective LR} = \eta \times \frac{\alpha}{r}$$
If you increase $r$ from $16$ to $64$ while keeping $\alpha = 16$ and learning rate $\eta$ fixed, you inadvertently reduce the update magnitude by $4\times$. To isolate the effect of rank capacity without changing update dynamics, either scale $\alpha$ proportionally with $r$ (e.g., maintain $\alpha = 2r$) or adjust the base learning rate.

### Q3: When should you prefer Full Fine-Tuning over LoRA/QLoRA?
**Answer**:
1. **Massive Domain Shift / Continued Pre-training**: When teaching an LLM an entirely new natural language or code syntax across billions of tokens, the low-rank assumption fails because widespread structural weights need fundamental reorganization.
2. **Maximum Parameter Capacity**: For competitive benchmark-topping foundation models where every fraction of a percent matters and distributed GPU clusters (H100 NVLink) are unconstrained.
3. **Inference Latency in Multi-tenant Serving**: While merged LoRA has zero overhead, serving 1,000 distinct non-merged LoRA adapters dynamically introduces modest routing overhead compared to single dedicated full fine-tuned checkpoints (though mitigated by S-LoRA/Punica).
