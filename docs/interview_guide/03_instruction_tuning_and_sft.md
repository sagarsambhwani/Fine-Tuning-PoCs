# 🛠️ Module 03: Instruction Tuning, SFT & Data Engineering

Supervised Fine-Tuning (SFT) transforms a raw causal language model into an instruction-following assistant. In production and interviews, **data quality, synthetic curation, data packing, and training stability** are the true differentiators between mediocre and state-of-the-art models.

---

## 1. Instruction Data Formats & Schemas

### A. Format Comparison

```
1. Alpaca Format (Single-Turn)
{"instruction": "Extract names", "input": "Alice met Bob.", "output": "Alice, Bob"}

2. ShareGPT Format (Multi-Turn Conversations)
{"conversations": [
    {"from": "human", "value": "How do vaccines work?"},
    {"from": "gpt", "value": "Vaccines train the immune system..."},
    {"from": "human", "value": "What are mRNA vaccines?"},
    {"from": "gpt", "value": "mRNA vaccines deliver genetic instructions..."}
]}

3. Hugging Face Standard Messages Format (OpenAI / ChatML compatible)
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"}
]}
```

---

## 2. Synthetic Data Engineering & Quality Curation

The modern consensus (LIMA: *Less Is More for Alignment*, Zhou et al., 2023) is that **1,000 carefully curated, high-quality samples outperform 50,000 noisy, low-quality web examples**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             SYNTHETIC DATA PIPELINE                                    │
│                                                                                        │
│  [Seed Tasks] ──► [Evol-Instruct] ──► [Filter & Judge] ──► [SemDeDup] ──► [Golden SFT] │
│   (100 core)      (Deepen/Broaden)    (LLM Rubric/Perp)    (Clustering)     (5k curated)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. Evol-Instruct (WizardLM Paradigm)
Evol-Instruct uses a powerful teacher LLM (e.g., GPT-4) to evolve simple seed prompts into complex multi-step reasoning challenges:

1. **In-Depth Evolution (Deepening Complexity)**:
   * **Add Constraints**: *"Add a requirement that the solution must run in $O(1)$ space."*
   * **Deepening**: *"Explain the quantum mechanical mechanism behind this chemical bond."*
   * **Concretizing**: *"Replace general concepts with a concrete enterprise banking scenario."*
   * **Multi-Step Reasoning**: *"Rewrite this single-step logic question into a 4-step deduction puzzle."*
2. **In-Breadth Evolution (Domain Diversity)**:
   * Mutate topic completely to underrepresented domains (e.g., legal compliance, bioinformatics, embedded systems).

### B. Deduplication: MinHash LSH vs. SemDeDup

* **MinHash LSH (Syntactic Deduplication)**: Hashes $k$-shingle substrings to find lexical duplicates (Jaccard similarity $> 0.8$). Fast, scales to billions of tokens.
* **SemDeDup (Semantic Deduplication)**:
  1. Embeds instruction texts using an embedding model (e.g., `text-embedding-3-large`).
  2. Clusters embeddings with $K$-Means.
  3. Within each cluster, computes pairwise cosine similarity.
  4. If cosine similarity $> 0.92$, drops the duplicate, retaining only the highest-quality example.
  * **Benefit**: Removes semantic duplicates that use completely different phrasing.

### C. Benchmark Decontamination (Preventing Data Leakage)
Before training, evaluate $N$-gram overlap (e.g., 13-gram exact matches) between training datasets and standard test benchmarks (MMLU, GSM8K, HumanEval, ARC). Any matched prompts must be purged to ensure legitimate benchmark evaluation.

---

## 3. Training Optimization: Packing vs. Padding

### A. The Padding Inefficiency
In standard batching, sequences are padded to `max_seq_length` with `<pad>` tokens:

```
Batch Item 1: [Token, Token, Token, <pad>, <pad>, <pad>, <pad>, <pad>]  <-- 60% wasted compute!
Batch Item 2: [Token, Token, Token,  Token, Token, Token, Token, Token]
```

### B. Sequence Packing (Sample Packing)
Packing concatenates multiple short conversations into a single fixed context window (e.g., 4096 tokens), separated by `<|endoftext|>` or `<|im_end|>`.

```
Packed Window (4096 tokens):
[ Conv A: Prompt -> Reply <eos> ][ Conv B: Prompt -> Reply <eos> ][ Conv C: Prompt... ]
```

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    packing=True,                    # Enables sample packing
    max_seq_length=4096,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    dataset_text_field="text"
)
```

> [!IMPORTANT]
> **Cross-Contamination Prevention**: When packing multiple sequences without 2D block-diagonal attention masks, Conversation B can attend to Conversation A. With FlashAttention-2 `varlen` (variable length) or `position_ids` reset, attention is strictly prevented from crossing sequence boundaries.

---

## 4. Hyperparameters & Learning Rate Schedulers

```
Cosine Decay with Warmup vs. WSD (Warmup-Stable-Decay)

   Learning Rate
        │          ┌───────────────────┐ (Stable)
        │         /                     \
        │        /                       \  <-- Rapid Decay (Annealing)
        │       /                         \
        │      / (Warmup)                  \
        └─────┴─────────────────────────────┴────────► Training Steps
```

| Hyperparameter | Recommended Range (LoRA) | Recommended Range (Full FT) | Notes / Rationale |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | $1 \times 10^{-4} - 5 \times 10^{-4}$ | $1 \times 10^{-5} - 2 \times 10^{-5}$ | LoRA updates fewer parameters, requiring higher gradient steps. |
| **LR Scheduler** | Cosine with Warmup | Cosine with Warmup / WSD | Warmup prevents gradient explosion in early steps. |
| **Warmup Ratio** | $0.03 - 0.05$ (3–5% of total steps) | $0.03 - 0.05$ | Too high wastes budget; too low destabilizes AdamW states. |
| **Weight Decay** | $0.01 - 0.1$ | $0.01 - 0.1$ | Regularizes weights; set to $0$ on bias and LayerNorm. |
| **Gradient Clipping** | `max_grad_norm = 1.0` | `max_grad_norm = 1.0` | Critical for clipping outlier activation spikes. |
| **Effective Batch Size** | $32 - 128$ sequences | $64 - 256$ sequences | $\text{Batch Size} = \text{Per-Device BS} \times \text{Grad Accum} \times \text{GPUs}$. |

---

## 5. Catastrophic Forgetting & Mitigation

When an LLM is fine-tuned aggressively on a narrow domain (e.g., Python code or legal QA), it loses general reasoning, conversational flow, and general knowledge.

### Mitigation Strategies:
1. **Replay Buffer (Data Mixing)**: Mix **5% to 15% general instruction data** (e.g., UltraChat, OpenOrca, ShareGPT) into the specialized domain dataset.
2. **LoRA Parameter Isolation**: LoRA preserves frozen base weights $W_0$. If severe degradation occurs, adapter weights can be scaled down via $\alpha$ or selectively disabled.
3. **KL-Divergence Regularization**: Add a penalty to the loss function that penalizes deviation from the base model's token distribution:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \gamma \mathbb{D}_{\text{KL}}(\pi_\theta(x) \parallel \pi_{\text{base}}(x))$$

---

## 6. Production Debugging: Diagnosing Training Anomalies

### Failure Mode 1: Loss Spikes to `NaN`
* **Root Cause 1**: FP16 Numerical Overflow. In standard FP16, max representable value is $65,504$. Attention logits $\frac{QK^\top}{\sqrt{d_k}}$ or Adam variance can overflow.
  * **Fix**: Use **Bfloat16 (BF16)** (same dynamic range as FP32) or enable `torch.cuda.amp.GradScaler`.
* **Root Cause 2**: Corrupted / Empty sequence in batch producing division by zero in loss normalization.
  * **Fix**: Sanitize dataset to ensure zero empty response strings.

### Failure Mode 2: Loss Drops to $\approx 0.001$ in First 20 Steps
* **Root Cause**: Label leakage. The training script failed to mask input prompts (`labels = -100`), or the target output was accidentally duplicated into the prompt field.

### Failure Mode 3: Repetitive Output Loops at Inference ("I am sorry, I am sorry, I am...")
* **Root Cause**: Overfitting on short sequences, high learning rate, or absence of `<|im_end|>` EOS tokens during training. The model never learned when to terminate generation.

---

## 🎯 Top Interview Q&A on SFT & Instruction Tuning

### Q1: How does TRL's `SFTTrainer` differ from the standard Hugging Face `Trainer`?
**Answer**:
1. **Integrated Loss Masking**: `SFTTrainer` natively supports `DataCollatorForCompletionOnlyLM` to mask prompts automatically.
2. **Sample Packing**: Natively packs multiple small sequences into a continuous `max_seq_length` buffer (`packing=True`) to eliminate padding waste.
3. **PEFT/LoRA Integration**: Accepts `peft_config` directly in initialization and manages trainable adapter wrappers seamlessly.
4. **Dataset Formatting**: Automatically applies tokenizer chat templates (`apply_chat_template`) to raw dictionary datasets.

### Q2: Why is effective batch size critical, and how do you calculate Gradient Accumulation Steps?
**Answer**:
Small batch sizes (e.g., 1 or 2) lead to high gradient variance, causing noisy updates and training instability. Gradient accumulation aggregates gradients over $K$ forward/backward micro-steps before calling `optimizer.step()`:

$$\text{Effective Batch Size} = (\text{per\_device\_train\_batch\_size}) \times (\text{gradient\_accumulation\_steps}) \times (\text{num\_gpus})$$

For example, on a single GPU with VRAM only allowing `batch_size = 2`, setting `gradient_accumulation_steps = 32` achieves an effective batch size of $2 \times 32 \times 1 = 64$, providing smooth, stable gradient convergence.
