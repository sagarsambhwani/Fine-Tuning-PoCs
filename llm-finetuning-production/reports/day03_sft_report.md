# Day 3: Supervised Fine-Tuning (SFT) & Token Loss Masking Report

**Project**: 7-Day LLM Fine-Tuning & Production Deployment  
**Date**: August 2026  
**Target Model**: `Qwen/Qwen2.5-1.5B-Instruct`  
**Notebook**: [`notebooks/day03_sft.ipynb`](../notebooks/day03_sft.ipynb)  

---

## 1. Executive Summary

This report documents the empirical evaluation of **ChatML template tokenization** and **Assistant Response Loss Masking (`labels = -100`)** on `Qwen/Qwen2.5-1.5B-Instruct` for structured JSON extraction.

By applying prompt masking, **45 out of 84 tokens (53.57%)** corresponding to the system instructions and user input are masked with `ignore_index = -100`. Backpropagation gradients are concentrated exclusively on the **39 active completion tokens (46.43%)**, preventing the model from wasting parameter updates learning arbitrary user phrasing and enforcing strict JSON output adherence with exact stopping on `<|im_end|>`.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             SEQUENCE TOKEN ALLOCATION (84 TOKENS)                        │
│                                                                                          │
│  Masked Prompt Tokens (Labels = -100): [████████████████████████         ] 45 (53.57%)   │
│  Active Gradient Tokens (Loss > 0):    [██████████████████               ] 39 (46.43%)   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Empirical Tokenization & Masking Metrics

| Metric | Measured Value | Architectural Significance |
| :--- | :--- | :--- |
| **Model Tokenizer** | `Qwen/Qwen2.5-1.5B-Instruct` | Byte-Pair Encoding (BPE) with special control tokens |
| **Vocabulary Size** | `151,643` | Native support for multi-lingual and structured tokens |
| **EOS Token / ID** | `<|im_end|>` (ID: `151645`) | Primary stop delimiter for single-turn structured extraction |
| **Total Sequence Length** | **84 tokens** | Compact input-output representation |
| **Prompt Length (Masked)** | **45 tokens** | System prompt + User input + ChatML delimiters |
| **Target Completion (Active)** | **39 tokens** | Raw JSON string + `<|im_end|>` |
| **Prompt Loss Masking Ratio** | **53.57%** | Reduces noisy gradient updates by >50% per step |
| **Active Target Loss Ratio** | **46.43%** | 100% focused on JSON keys, values, types, and closing brace |

---

## 3. Rendered ChatML Structure & Token Partitioning

### A. Rendered Jinja ChatML String
```text
<|im_start|>system
You are an expert entity extraction system. Extract structured order information into JSON.<|im_end|>
<|im_start|>user
John ordered 3 laptops for $2400 and wants delivery on Friday.<|im_end|>
<|im_start|>assistant
{"customer": "John", "quantity": 3, "product": "laptops", "amount": 2400.0, "delivery_day": "Friday"}<|im_end|>
```

### B. Active Gradient Token Inspection
The exact token slice where `labels != -100` computes backpropagation gradients:
```json
{"customer": "John", "quantity": 3, "product": "laptops", "amount": 2400.0, "delivery_day": "Friday"}<|im_end|>
```

---

## 4. Mathematical Mechanics of Loss Masking

### Standard Causal LM vs. Masked Instruction Tuning

1. **Standard Unmasked Causal LM (Pretraining Objective)**:
   Computes auto-regressive cross-entropy across all tokens $t \in [1, N]$:
   $$\mathcal{L}_{\text{unmasked}} = -\frac{1}{N} \sum_{t=1}^{N} \log P(x_t \mid x_{<t}; \theta)$$
   *Problem*: Penalizes the model for failing to predict the exact phrasing of the user's prompt, leading to suboptimal weight updates on domain-specific tasks.

2. **Assistant-Masked SFT Objective (Day 3 Implementation)**:
   Setting `labels[1:K] = -100` where $K=45$ (prompt length) and $N=84$ (total length):
   $$\mathcal{L}_{\text{masked}} = -\frac{1}{N - K} \sum_{t=K+1}^{N} \log P(x_t \mid x_{<t}; \theta)$$
   In PyTorch's `torch.nn.CrossEntropyLoss(ignore_index=-100)`, tokens with value `-100` contribute $0$ to the loss and receive $\nabla_\theta \mathcal{L} = 0$.

---

## 5. Key Interview & Engineering Insights

1. **Why is the `<|im_end|>` token included in the active gradient slice?**
   If `<|im_end|>` was masked, the model would learn how to write the JSON content, but would never learn *when to stop generating*. This would cause generation loops, conversational runaway, or hallucinated extra fields.
2. **How does this integrate with `SFTTrainer` and `DataCollatorForCompletionOnlyLM`?**
   In TRL, `DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant\n", tokenizer=tokenizer)` automatically locates the response prefix and sets all preceding token IDs in `labels` to `-100`.
3. **Data Format Readiness**:
   The processed Hugging Face `Dataset` with standard `messages` column (`system`, `user`, `assistant`) is ready for direct batch consumption in the Day 4 training loop.
