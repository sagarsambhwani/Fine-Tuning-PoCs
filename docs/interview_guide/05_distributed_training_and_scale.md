# 🌐 Module 05: Distributed Training, Parallelism & Systems Engineering

When scaling LLM fine-tuning from single consumer GPUs to enterprise clusters of 8×H100 or multi-node supercomputers, distributed systems engineering is essential. This module covers GPU memory internals, 3D parallelism, DeepSpeed ZeRO, PyTorch FSDP, communication primitives, and low-level kernel optimizations.

---

## 1. The GPU Memory Equation: The Exact Breakdown

To debug Out-Of-Memory (OOM) errors and architect distributed training systems, you must know the exact byte breakdown of training state memory.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               TOTAL GPU VRAM ALLOCATION                                │
│                                                                                        │
│   VRAM_total = VRAM_model + VRAM_gradients + VRAM_optimizer + VRAM_activations + VRAM_temp │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. Static Training States (Per Parameter in FP16/BF16 with AdamW)
For a model with $N$ parameters:

1. **Model Weights**: $2 \times N$ bytes (FP16/BF16)
2. **Gradients**: $2 \times N$ bytes (FP16/BF16)
3. **AdamW Optimizer States**: **$12 \times N$ bytes**
   * FP32 Master Weights: $4 \times N$ bytes (maintains numerical precision during small gradient steps)
   * FP32 Momentum ($1^{\text{st}}$ moment): $4 \times N$ bytes
   * FP32 Variance ($2^{\text{nd}}$ moment): $4 \times N$ bytes
4. **Total Static State**: **$16 \times N$ bytes** (16 GB of VRAM per 1 Billion parameters!)

### B. Dynamic Activation Memory & Gradient Checkpointing
Activations are the intermediate tensor outputs stored during the forward pass to compute gradients during the backward pass.

* **Without Gradient Checkpointing**: Memory scales with $O(L \times S \times B \times H)$ (Layers $\times$ Sequence Length $\times$ Batch Size $\times$ Hidden Dim).
* **With Activation Checkpointing (Gradient Checkpointing)**: Stored activations are discarded during the forward pass and **recomputed on-the-fly during the backward pass**.
  * **Memory Savings**: Slashes activation memory by **60% to 80%**.
  * **Compute Cost**: Adds a $\approx 25\%$ to $30\%$ compute overhead (one extra forward pass per layer).

---

## 2. Distributed Data Parallelism: DeepSpeed ZeRO & PyTorch FSDP

Standard Distributed Data Parallelism (DDP) replicates the entire model, gradients, and optimizer states across every GPU, making it impossible to train models larger than a single GPU's VRAM.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ZeRO / FSDP MEMORY SHARDING                               │
│                                                                                        │
│  Standard DDP:  [ Weights ] [ Gradients ] [ Optimizer States (12B) ]  <-- On EVERY GPU │
│                                                                                        │
│  ZeRO Stage 1:  [ Weights ] [ Gradients ] [ Sharded Optimizer (12B/N) ] (4x savings)   │
│                                                                                        │
│  ZeRO Stage 2:  [ Weights ] [ Sharded Grads ] [ Sharded Optimizer ]   (8x savings)     │
│                                                                                        │
│  ZeRO Stage 3:  [ Sharded W ] [ Sharded Grads ] [ Sharded Optimizer ] (Linear Nx)     │
│  (FSDP Full)                                                                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

| Sharding Strategy | What is Sharded? | Communication Overhead vs DDP | When to Use? |
| :--- | :--- | :--- | :--- |
| **ZeRO-1 / FSDP Stage 1** | Optimizer States ($12 \times N$) | **Zero extra communication** | Fine-tuning 7B–14B models on 8 GPUs. |
| **ZeRO-2 / FSDP Stage 2** | Optimizer States + Gradients | **Zero extra communication** | Reduces peak memory during backward pass. |
| **ZeRO-3 / FSDP Full** | Optimizer + Gradients + Model Weights | **$+50\%$ communication** (All-Gather weights before each forward & backward layer) | Training 70B+ models across multi-GPU nodes without Tensor Parallelism. |
| **ZeRO-Offload** | Offloads Optimizer/Weights to CPU RAM or NVMe | High PCIe transfer bottleneck | Training giant models on minimal GPU setups when throughput is secondary. |

---

## 3. 3D Parallelism: TP, PP, DP & Context Parallelism

When a single layer cannot fit in one GPU or when training 70B–405B models, we combine orthogonal parallelism dimensions:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 3D PARALLELISM MATRIX                                  │
│                                                                                        │
│  1. Tensor Parallelism (TP)  ──► Intra-node (Within 8x GPU via NVLink 900 GB/s)       │
│  2. Pipeline Parallelism (PP)──► Inter-node (Across nodes via InfiniBand 400 Gbps)     │
│  3. Data Parallelism (DP)    ──► Scaled across all remaining nodes via ZeRO/FSDP       │
│  4. Context Parallelism (CP) ──► Splits ultra-long sequences (128k+) across GPUs       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. Tensor Parallelism (Megatron-LM Style)
Splits individual weight matrices across GPUs:

```
Column Parallel (MLP First Layer):        Row Parallel (MLP Second Layer):
       X                                       [ X1 , X2 ]
    ┌──┴──┐                                      ┌──┴──┐
    ▼     ▼                                      ▼     ▼
  [W1]   [W2]                                  [W1]   [W2]
    │     │                                      │     │
    ▼     ▼                                      ▼     ▼
  [Y1]   [Y2] (Concat)                         [Y1] + [Y2] (AllReduce Sum)
```

* **Communication Requirement**: Requires high-speed **NVLink** (900 GB/s on H100) because an `AllReduce` collective must be performed after every transformer layer. Never run TP across slow network cables (InfiniBand/Ethernet).

### B. Pipeline Parallelism (GPipe & 1F1B Schedule)
Splits transformer layers sequentially across nodes (e.g., Layers 1–16 on GPU 0, Layers 17–32 on GPU 1).

* **The Pipeline Bubble**: In naive GPipe, downstream GPUs sit idle waiting for upstream activations.
* **1F1B (One Forward, One Backward)**: Interleaves forward and backward passes of micro-batches to reduce the idle bubble fraction:
  $$\text{Bubble Fraction } F_{\text{bubble}} = \frac{p - 1}{m + p - 1}$$
  *(Where $p$ = pipeline stages, $m$ = micro-batches. As $m \gg p$, bubble overhead approaches $0$).*

---

## 4. Collective Communication Primitives & Ring-AllReduce Math

```
Collective Primitives Overview:
• Broadcast:     One GPU sends identical tensor to all GPUs.
• Scatter:       One GPU sends unique chunks to each GPU.
• Gather:        One GPU collects chunks from all GPUs.
• AllGather:     ALL GPUs collect chunks from all GPUs.
• ReduceScatter: Reduces (sums) chunks and shards result across all GPUs.
• AllReduce:     Reduces and replicates sum across ALL GPUs (AllReduce = ReduceScatter + AllGather).
```

### Ring-AllReduce Mathematics
In a ring topology of $N$ GPUs with tensor size $S$ bytes:

1. Ring is traversed in two phases: **Scatter-Reduce** followed by **AllGather**.
2. Total data transferred per GPU:
   $$\text{Total Transferred} = 2 \times \left( \frac{N - 1}{N} \right) \times S$$
3. **Key Property**: As $N$ grows large, $\frac{N-1}{N} \to 1$. The data transferred per GPU is **independent of the number of GPUs** ($2S$), enabling near-linear scaling!

---

## 5. Hardware Acceleration & Low-Level Kernels

### A. FlashAttention-2 vs FlashAttention-3
Standard attention repeatedly transfers intermediate matrices $S = QK^\top$ ($O(N^2)$) between slow **HBM (VRAM)** and high-speed **SRAM die cache**.

```
HBM (High Bandwidth Memory): 2-3 TB/s  <-- Slow bottleneck
               ▲
               │ (Repeated R/W transfers in standard attention)
               ▼
SRAM (On-Chip Cache):        19 TB/s   <-- Lightning fast
```

* **FlashAttention Innovation**: Computes attention using **online Softmax normalization in tiled SRAM blocks**, completely avoiding writing $O(N^2)$ attention matrices to HBM.
* **FlashAttention-3**: Leverages NVIDIA Hopper (H100) asynchronous Tensor Core pipelines and FP8 matrix engines for $1.5\text{x}–2\text{x}$ speedups over FlashAttention-2.

### B. Precision Comparison: FP16 vs BF16 vs FP8

```
FP32:  [ 1 sign ][ 8 exponent ][ 23 mantissa ]   <-- Full precision (Master weights)
FP16:  [ 1 sign ][ 5 exponent ][ 10 mantissa ]   <-- Prone to underflow/overflow (Max 65,504)
BF16:  [ 1 sign ][ 8 exponent ][ 7 mantissa  ]   <-- Same dynamic range as FP32 (NO overflow!)
FP8:   [ E4M3 (Forward Pass) ] / [ E5M2 (Gradients/Backward) ]
```

---

## 🎯 Top Interview Q&A on Distributed Systems

### Q1: Why is Bfloat16 (BF16) universally preferred over FP16 for LLM training?
**Answer**:
BF16 allocates 8 bits to the exponent (identical to FP32) and 7 bits to the mantissa, giving it a dynamic range of $\sim 10^{-38}$ to $10^{38}$, compared to FP16's limited 5-bit exponent (range $10^{-5}$ to $65,504$). Because LLM training frequently encounters gradient and attention logit spikes that exceed $65,504$, FP16 requires complex dynamic loss scaling (`GradScaler`) and often suffers from sudden `NaN` crashes. BF16 completely eliminates numerical overflow without requiring loss scalers.

### Q2: What is the exact difference between Tensor Parallelism and Pipeline Parallelism?
**Answer**:
* **Tensor Parallelism (TP)**: Shards **individual operations and matrices within a layer** across GPUs. It requires high-bandwidth, ultra-low latency NVLink connections because synchronization (`AllReduce`) occurs after every single layer.
* **Pipeline Parallelism (PP)**: Shards **entire layers sequentially across GPUs**. Communication occurs only at layer boundaries (sending activations forward and gradients backward), making it tolerant to lower-bandwidth inter-node connections (InfiniBand/RoCE), at the cost of pipeline bubble latency.
