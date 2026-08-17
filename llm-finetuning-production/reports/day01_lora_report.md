# Day 1: Full Fine-Tuning vs. LoRA Parameter & Memory Benchmark Report

**Project**: 7-Day LLM Fine-Tuning & Production Deployment  
**Date**: August 2026  
**Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` (1.56B Total Parameters)  
**Notebook**: [`notebooks/day01_lora.ipynb`](../notebooks/day01_lora.ipynb)  

---

## 1. Executive Summary

This report documents the mathematical analysis, parameter accounting, and memory profiling of **Low-Rank Adaptation (LoRA)** versus **Full Parameter Fine-Tuning** applied to `Qwen/Qwen2.5-1.5B-Instruct`.

By decomposing weight update matrices $\Delta W$ into low-rank matrices $B \cdot A$ with rank $r=16$ across all attention and MLP projection layers, trainable parameters were reduced from **1.56 Billion down to 18.46 Million (1.1820%)**. Consequently, adapter weights occupy only **70.49 MB** on disk compared to **3.09 GB** for the full model checkpoint—a **97.7% disk storage reduction** with zero catastrophic forgetting of base model capabilities.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             PARAMETER DISTRIBUTION BREAKDOWN                             │
│                                                                                          │
│  Frozen Base Parameters:    [███████████████████████████████████████ ]  1.543B (98.818%) │
│  Trainable LoRA Parameters: [█                                       ]  18.46M (1.182%)  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Parameter & Memory Comparison Table

| Metric | Full Parameter Fine-Tuning | LoRA (FP16 Adapter, $r=16$) | Efficiency Gain / Impact |
| :--- | :--- | :--- | :--- |
| **Total Base Parameters** | 1,543,714,304 | 1,543,714,304 | Same base capacity |
| **Trainable Parameters** | 1,543,714,304 (100.0%) | **18,464,768 (1.1820%)** | **83.6x fewer parameters to update** |
| **Frozen Parameters** | 0 (0.0%) | **1,543,714,304 (98.818%)** | Base knowledge 100% protected |
| **Gradients Computed** | All 1.543B weights | **Only 18.46M adapter weights** | 98.8% gradient compute saved |
| **AdamW Optimizer Memory** | **~12.35 GB** ($8\text{ bytes/param}$) | **~0.147 GB** ($147.7\text{ MB}$) | **98.8% optimizer RAM saved** |
| **Checkpoint Storage (Disk)** | **~3.09 GB** (`model.safetensors`) | **70.49 MB** (`adapter_model.safetensors`) | **97.7% storage reduction** |
| **Catastrophic Forgetting** | ⚠️ High Risk | 🛡️ Zero Risk (Base frozen) | Preserves general domain knowledge |
| **Multi-Tenant Serving** | 1 full model per task | 1 base model + $N$ swap adapters | Massive cost reduction in production |

---

## 3. Mathematical Breakdown of Memory Footprint

The total VRAM required during training is governed by:
$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{gradients}} + \text{VRAM}_{\text{optimizer}} + \text{VRAM}_{\text{activations}}$$

### A. AdamW Optimizer State Explosion in Full Fine-Tuning
AdamW tracks two 32-bit floating-point states for every single trainable parameter:
1. **First Moment (Momentum)**: $m_t$ (4 bytes/param)
2. **Second Moment (Variance)**: $v_t$ (4 bytes/param)

* **For Full Fine-Tuning**:
  $$\text{VRAM}_{\text{optimizer}} = 1.543 \times 10^9 \text{ params} \times 8 \text{ bytes} \approx \mathbf{12.35 \text{ GB}}$$
  When combined with 16-bit model weights (3.09 GB) and gradients (3.09 GB), the memory instantly surpasses **18.5 GB** (causing immediate Out-Of-Memory on 16GB T4/V100 GPUs) before even allocating activation memory!

* **For LoRA ($r=16$)**:
  $$\text{VRAM}_{\text{optimizer}} = 1.846 \times 10^7 \text{ params} \times 8 \text{ bytes} \approx \mathbf{147.7 \text{ MB}}$$
  Optimizer memory drops from **12.35 GB to 147.7 MB**, making fine-tuning comfortably feasible on standard hardware.

---

## 4. Layer-by-Layer LoRA Adapter Target Distribution

Applying LoRA across **all linear attention and MLP layers** rather than attention-only (`q_proj`, `v_proj`) provides superior adaptation quality and downstream task performance:

| Layer Category | Target Modules | Module Dimension ($d_{\text{in}} \times d_{\text{out}}$) | LoRA Rank $r$ | LoRA Parameters per Layer |
| :--- | :--- | :--- | :--- | :--- |
| **Self-Attention Query** | `q_proj` | $1536 \times 1536$ | 16 | $(1536 \times 16) + (16 \times 1536) = 49,152$ |
| **Self-Attention Key** | `k_proj` | $1536 \times 256$ (GQA) | 16 | $(1536 \times 16) + (16 \times 256) = 28,672$ |
| **Self-Attention Value** | `v_proj` | $1536 \times 256$ (GQA) | 16 | $(1536 \times 16) + (16 \times 256) = 28,672$ |
| **Self-Attention Output** | `o_proj` | $1536 \times 1536$ | 16 | $(1536 \times 16) + (16 \times 1536) = 49,152$ |
| **MLP Gate Projection** | `gate_proj` | $1536 \times 8960$ | 16 | $(1536 \times 16) + (16 \times 8960) = 167,936$ |
| **MLP Up Projection** | `up_proj` | $1536 \times 8960$ | 16 | $(1536 \times 16) + (16 \times 8960) = 167,936$ |
| **MLP Down Projection** | `down_proj` | $8960 \times 1536$ | 16 | $(8960 \times 16) + (16 \times 1536) = 167,936$ |

$$\text{Total LoRA Parameters across 28 Transformer Blocks} = \mathbf{18,464,768}$$

---

## 5. Production Architectural Benefits

1. **Modular Multi-Tenant Adapter Swapping**:
   In production serving, a single base model instance can reside in GPU memory while distinct, task-specific 70 MB LoRA adapters (e.g. Legal, Medical, JSON extraction) are dynamically activated per request.
2. **Zero-Overhead Inference Deployment (`merge_and_unload`)**:
   During production packaging (Day 6), adapter weights can be permanently fused into the base weights via $W' = W_0 + \frac{\alpha}{r} B A$, achieving **zero added inference latency**.

---

## 6. Key Takeaways for Technical Interviews

* **Why is matrix $B$ initialized to zero and matrix $A$ to Gaussian?**  
  Because $\Delta W = B \cdot A = 0 \cdot A = 0$ at step 0. The model starts with the exact behavior of the base model with zero initial perturbation.
* **What is the significance of the scaling factor $\frac{\alpha}{r}$?**  
  It stabilizes gradient magnitudes when experimenting with different rank values ($r$), allowing rank tuning without aggressively retuning learning rates.
