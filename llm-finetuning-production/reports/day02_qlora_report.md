# Day 2: LoRA vs. QLoRA Experimental Benchmark Report

## 1. Experiment Overview
This report documents the empirical comparison between **Standard LoRA (16-bit)** and **QLoRA (4-bit NF4 with Double Quantization)** trained on `Qwen/Qwen2.5-1.5B-Instruct` on a Google Colab T4 GPU (16GB VRAM).

## 2. Experimental Setup
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Hardware**: Google Colab NVIDIA T4 (15.0 GB VRAM)
- **Batch Size**: 2 (with Gradient Accumulation = 4, Effective Batch Size = 8)
- **Sequence Length**: 512 tokens
- **LoRA Hyperparameters**: $r=16$, $\alpha=32$, $\text{dropout}=0.05$, all linear attention & MLP modules targeted.

---

## 3. Empirical Comparison Table

| Metric | Full Fine-Tuning (Estimated) | Standard LoRA (FP16) | QLoRA (NF4 4-bit) |
|---|---|---|---|
| **Base Model Precision** | 16-bit (FP16/BF16) | 16-bit (FP16) | 4-bit (NF4) |
| **Base Weight Memory** | ~3.1 GB | ~3.1 GB | ~1.1 GB |
| **Optimizer State Memory (AdamW)** | ~12.4 GB | ~0.08 GB | ~0.08 GB (Paged 8-bit) |
| **Peak GPU VRAM during Training** | OOM (>16 GB on T4) | `NOT RUN` (est. ~7.8 GB) | `NOT RUN` (est. ~3.6 GB) |
| **Total Model Parameters** | 1,543,714,816 | 1,543,714,816 | 1,543,714,816 |
| **Trainable Parameters** | 1,543,714,816 (100%) | `NOT RUN` (~18.4M) | `NOT RUN` (~18.4M) |
| **Trainable Ratio (%)** | 100.0% | `NOT RUN` (~1.19%) | `NOT RUN` (~1.19%) |
| **Training Throughput (samples/sec)** | N/A (OOM) | `NOT RUN` | `NOT RUN` |
| **Final Training Loss** | N/A | `NOT RUN` | `NOT RUN` |

*(Note: Values marked `NOT RUN` will be updated after executing `notebooks/day02_qlora.ipynb` in Colab).*

---

## 4. Key Takeaways & Interview Explanations
1. **NF4 (NormalFloat4)**: Quantizes weights assuming a standard zero-mean normal distribution, achieving lower quantization error than uniform 4-bit integers (INT4).
2. **Double Quantization**: Quantizes the quantization constants themselves (FP32 $\to$ FP8), reducing memory by ~0.37 bits per parameter (~70MB saved on 1.5B).
3. **Paged Optimizers**: Uses CUDA Unified Memory to automatically page optimizer states to CPU RAM during memory spikes, preventing out-of-memory crashes.
