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

| Step | Epoch | Training Loss | Validation Loss | Output Entropy | Mean Token Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.33 | `0.221277` | `0.209245` | `0.218825` | `93.6320%` |
| 100 | 0.67 | `0.187893` | `0.188795` | `0.192108` | `93.8839%` |
| 150 | 1.00 | `0.183324` | `0.189227` | `0.188797` | `93.8849%` |
| 200 | 1.33 | `0.182930` | `0.183771` | `0.184747` | `93.9071%` |
| 250 | 1.67 | `0.180359` | `0.181885` | `0.181847` | `94.0048%` |
| 300 | 2.00 | `0.175832` | `0.181277` | `0.179077` | `93.9235%` |
| 350 | 2.33 | `0.169802` | `0.179257` | `0.176712` | `93.9041%` |
| 400 | 2.67 | `0.173595` | `0.178655` | `0.179041` | `93.8906%` |
| **450 (Final)** | **3.00** | **`0.177998`** | **`0.178609`** | **`0.178762`** | **`93.9201%`** |

- **Total Training Time**: 4,108.96 seconds (~1 hour, 8 minutes, 28 seconds)
- **Training Throughput**: 0.876 samples/second on NVIDIA T4 GPU (FP16 compute)
- **Total Tokens Trained**: 477,021 active tokens
- **Saved Adapter Artifact**: `models/adapters/qwen-1.5b-order-extractor/`

*(Metrics populated from empirical run of `notebooks/day04_domain_finetuning.ipynb` and `reports/training_metrics.json`)*
