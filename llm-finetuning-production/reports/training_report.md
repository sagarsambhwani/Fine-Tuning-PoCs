# Domain-Specific Structured JSON SFT Training Report

## 1. Executive Summary
This report summarizes the end-to-end Supervised Fine-Tuning (SFT) run executed in **Day 4** for structured entity extraction from natural language order prompts into validated JSON.

## 2. Training Run Specifications
- **Target Task**: Unstructured Text $\to$ Structured Order JSON (`customer`, `quantity`, `product`, `amount`, `delivery_day`)
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Training Method**: SFT + QLoRA (4-bit NF4)
- **Dataset Size**:
  - Training: 1,200 samples
  - Validation: 150 samples
  - Held-out Test: 150 samples
- **Max Sequence Length**: 512 tokens
- **Epochs**: 3
- **Effective Batch Size**: 8 (Batch size per device: 2, Gradient Accumulation: 4)
- **Learning Rate**: $2 \times 10^{-4}$ (Cosine schedule with 5% warmup)
- **Optimizer**: `paged_adamw_8bit`

---

## 3. Training Dynamics & Checkpoint Log

| Step | Epoch | Training Loss | Validation Loss | Peak VRAM (GB) |
|---|---|---|---|---|
| 50 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 100 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 150 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| 200 | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |
| Final (450) | `NOT RUN` | `NOT RUN` | `NOT RUN` | `NOT RUN` |

- **Total Training Time**: `NOT RUN`
- **Final Train Loss**: `NOT RUN`
- **Saved Adapter Artifact**: `models/adapters/qwen-1.5b-order-extractor/`

*(Metrics to be populated automatically from `reports/training_metrics.json` after running `notebooks/day04_domain_finetuning.ipynb`)*
