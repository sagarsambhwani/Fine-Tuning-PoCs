# 🧠 Module 01: Foundations, Tokenization & Architecture Mechanics

This module covers the core foundational building blocks of Large Language Models (LLMs) and fine-tuning: pre-training paradigms, subword tokenization algorithms, vocabulary manipulation, loss masking, attention variants, and positional embeddings.

---

## 1. The Post-Training Hierarchy

Understanding where fine-tuning fits into the LLM lifecycle is the #1 question in technical screening rounds.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   LLM TRAINING PIPELINE                                     │
│                                                                                             │
│  [Pre-Training] ────────► [Continued Pre-training] ──► [Supervised Fine-Tuning] ──► [Alignment]│
│  • Web-scale (trillions) • Domain text (10B-100B)      • QA, Instructions (10k-1M) • Preferences   │
│  • Next-token prediction • Next-token prediction       • Next-token (masked prompt)• DPO/GRPO/RLHF │
│  • Learns world knowledge• Learns domain vocabulary    • Learns task & structure   • Learns values │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Phase | Input Data | Objective | Key Loss | Typical Learning Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Pre-Training** | Trillions of raw tokens (Common Crawl, Books, GitHub) | Learn universal linguistic patterns, grammar, and general knowledge | Causal Cross-Entropy over all tokens | $1 \times 10^{-4} - 3 \times 10^{-4}$ |
| **Continued Pre-Training (CPT)** | Billions of domain-specific tokens (Legal, Medical, Finance) | Adapt base model to specialized domain terminology and syntax | Causal Cross-Entropy over all domain tokens | $2 \times 10^{-5} - 5 \times 10^{-5}$ |
| **Supervised Fine-Tuning (SFT)** | Thousands to millions of high-quality (Prompt, Response) pairs | Format model into an instruction-following assistant or tool caller | Cross-Entropy **masked on prompt tokens** (`labels=-100`) | $1 \times 10^{-5} - 2 \times 10^{-5}$ (FFT) / $1 \times 10^{-4} - 5 \times 10^{-4}$ (LoRA) |
| **Preference Alignment** | Pairwise (Prompt, Chosen, Rejected) or Verifiable Tasks | Align tone, safety, reasoning honesty, eliminate hallucinations | DPO / IPO / KTO / GRPO loss | $5 \times 10^{-7} - 5 \times 10^{-6}$ |

---

## 2. Tokenization Algorithms Deep Dive

Tokenization converts discrete raw text into integer IDs. Understanding tokenization is vital because **LLMs do not see characters or words; they see token IDs**.

```
Raw Text: "Unstoppable"
Byte-Pair Encoding: ["Un", "stopp", "able"] ──► Token IDs: [2849, 19284, 602]
```

### A. Subword Tokenization Algorithms Comparison

| Algorithm | Used By | Mechanics | Splitting Rule |
| :--- | :--- | :--- | :--- |
| **Byte-Pair Encoding (BPE)** | GPT-2/3/4, LLaMA, Mistral, RoBERTa | Bottom-up frequency-based merge of most frequent byte/character pairs. | Frequency count of adjacent pairs. |
| **WordPiece** | BERT, DistilBERT, Electra | Bottom-up, but merges pairs that **maximize the likelihood of the language model data** (Information Gain / Mutual Information). Uses `##` prefix for subwords. | Maximizing $\frac{P(w_i w_j)}{P(w_i)P(w_j)}$. |
| **Unigram (SentencePiece)** | T5, ALBERT, Gemma | Top-down: Starts with a massive vocabulary and iteratively prunes tokens that minimize the increase in training corpus perplexity. | Loss minimization / Perplexity delta. |
| **Byte-level BPE (BBPE)** | GPT-4, LLaMA 3, Qwen | Operates on raw UTF-8 bytes (256 base vocabulary). Can represent **any Unicode string** without ever producing an Out-Of-Vocabulary (`<unk>`) token. | Byte-level frequency merges. |

### B. Custom Vocabulary Expansion & Embedding Resizing

When fine-tuning on a specialized domain (e.g., medical formulas or code syntax like `<<<CALL_API>>>`), standard tokenizers may split unique entities into 5–10 subword fragments. Adding custom tokens prevents fragmentation and improves inference latency.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")

# 1. Add new domain tokens
new_tokens = ["<custom_tool_start>", "<custom_tool_end>", "COVID-19_SARS"]
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens.")

# 2. CRITICAL STEP: Resize model embedding layer
model.resize_token_embeddings(len(tokenizer))

# 3. Best Practice: Initialize new token embeddings with mean of existing embeddings
# Avoids random noise gradients destabilizing early training
with torch.no_grad():
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    
    # Calculate mean embedding
    mean_input_emb = input_embeddings[:-num_added].mean(dim=0)
    mean_output_emb = output_embeddings[:-num_added].mean(dim=0)
    
    # Initialize new rows
    for i in range(1, num_added + 1):
        input_embeddings[-i] = mean_input_emb
        output_embeddings[-i] = mean_output_emb
```

> [!WARNING]
> **Interview Trap**: If you add new tokens and call `model.resize_token_embeddings(len(tokenizer))`, by default PyTorch initializes the new rows with **random Gaussian noise**. If you freeze the base model (as in LoRA) and forget to train the embedding layer (`modules_to_save=["embed_tokens", "lm_head"]`), the new tokens will produce **pure random garbage**!

---

## 3. Chat Templates & Loss Masking Mechanics

### A. The ChatML Standard Format
Modern instruction-tuned LLMs use special delimiter tokens to distinguish system prompts, user turns, and assistant replies.

```xml
<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Write a binary search in Python.<|im_end|>
<|im_start|>assistant
def binary_search(arr, target):
    ...<|im_end|>
```

### B. Loss Masking (`labels = -100`)
In Supervised Fine-Tuning (SFT), we **only want to calculate gradients on the assistant's response**, not on the system prompt or user query. Calculating loss on user tokens forces the model to memorize user questions rather than learning how to respond.

In PyTorch, `torch.nn.CrossEntropyLoss(ignore_index=-100)` ignores tokens where the target label is `-100`.

```
Tokenized Sequence:
Tokens:   [ <|im_start|>, system, \n, You..., <|im_end|>, <|im_start|>, user, \n, Write..., <|im_end|>, <|im_start|>, assistant, \n, def..., <|im_end|> ]
Labels:   [    -100,      -100,  -100, -100,   -100,       -100,      -100,-100, -100,    -100,        -100,        -100,   -100, def..., <|im_end|> ]
                                                                                                                   └──── Only these compute loss! ───┘
```

#### Code Implementation: Custom Data Collator with Loss Masking

```python
import torch

def create_masked_labels(input_ids, tokenizer, response_template="<|im_start|>assistant\n"):
    labels = input_ids.clone()
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
    
    batch_size, seq_len = input_ids.shape
    for i in range(batch_size):
        # Find the starting index of the assistant response
        seq = input_ids[i].tolist()
        # Default: mask everything
        labels[i, :] = -100
        
        # Locate response template in sequence
        for idx in range(len(seq) - len(response_token_ids) + 1):
            if seq[idx : idx + len(response_token_ids)] == response_token_ids:
                start_response_idx = idx + len(response_token_ids)
                # Unmask from start of response to end of sequence
                labels[i, start_response_idx:] = input_ids[i, start_response_idx:]
                break
    return labels
```

---

## 4. Modern Attention Architectures (MHA vs MQA vs GQA)

In generative inference and fine-tuning, Key-Value (KV) cache size is the main bottleneck for long context sequences.

```
Multi-Head Attention (MHA)      Grouped-Query Attention (GQA)      Multi-Query Attention (MQA)
    (e.g., LLaMA-1, GPT-3)             (e.g., LLaMA-3, Mistral)               (e.g., Falcon)
    
    Q Q Q Q   Q Q Q Q                  Q Q Q Q   Q Q Q Q                  Q Q Q Q   Q Q Q Q
    │ │ │ │   │ │ │ │                  └──┬──┘   └──┬──┘                  └────┬────┘
    K K K K   K K K K                     K         K                          K
    V V V V   V V V V                     V         V                          V
  (8 Q, 8 K, 8 V heads)               (8 Q, 2 K, 2 V groups)             (8 Q, 1 K, 1 V head)
```

| Architecture | Query Heads ($H_Q$) | Key/Value Heads ($H_{KV}$) | KV Cache Memory Savings | Quality vs MHA |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Head Attention (MHA)** | $N$ | $N$ | $1 \times$ (Baseline) | $100\%$ (Baseline) |
| **Grouped-Query Attention (GQA)** | $N$ | $G$ (where $1 < G < N$) | $\frac{N}{G} \times$ (e.g., $8\times$ reduction if $N=32, G=4$) | $\approx 99\%$ of MHA quality |
| **Multi-Query Attention (MQA)** | $N$ | $1$ | $N \times$ reduction | Minor degradation in complex reasoning |

---

## 5. Positional Encodings & Context Length Extension

### A. Rotary Position Embeddings (RoPE)
RoPE (*Su et al., 2021*) encodes relative position by rotating Query and Key vectors in 2D coordinate planes using complex numbers:

$$\mathbf{q}_m = \mathbf{R}_{\Theta, m}^d \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n}^d \mathbf{W}_k \mathbf{x}_n$$

$$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = \mathbf{x}_m^\top \mathbf{W}_q^\top \mathbf{R}_{\Theta, n-m}^d \mathbf{W}_k \mathbf{x}_n$$

* **Property**: The inner product depends **only on the relative distance $(m - n)$**, not absolute positions $m$ and $n$.

### B. Extending Context Length: RoPE Scaling Techniques

1. **Linear RoPE Scaling**: Scales down position indices by factor $s = \frac{L_{\text{new}}}{L_{\text{old}}}$. Smooth but causes high-frequency feature loss.
2. **NTK-Aware RoPE Scaling**: Leaves high-frequency dimensions unchanged (preserving local grammar) while scaling low-frequency dimensions (preserving long-range dependencies).
3. **YaRN (Yet another RoPE extensioN)**: Applies temperature correction to attention Softmax entropy alongside dimension-wise NTK interpolation, allowing stable fine-tuning up to 128k+ tokens with only 400 steps of fine-tuning.

---

## 🎯 Top Interview Q&A on Foundations & Tokenization

### Q1: Why can't we just perform Full Fine-Tuning on raw text using standard Causal LM loss for instruction following?
**Answer**:
Raw Causal LM loss computes cross-entropy over all tokens equally. If trained on raw conversation text without loss masking, the model allocates substantial gradient capacity toward predicting the user's prompt (which contains diverse syntax and unpredictable phrasing). This causes two major failures:
1. **Prompt Memorization & Hallucination**: The model learns to generate user questions rather than answering them.
2. **Gradient Inefficiency**: Up to 60–80% of backpropagation compute is wasted optimizing tokens that the model will never be asked to generate at test time. Loss masking (`labels=-100` on prompt) guarantees that 100% of gradient updates optimize the conditional probability $P(\text{Response} \mid \text{Prompt})$.

### Q2: Explain the token healing problem and why prompt delimiters matter.
**Answer**:
Subword tokenizers greedily merge characters. If a user prompt ends with a space or partial word (e.g., `The capital of France is `), the tokenizer might encode `is ` as a distinct single token ID (`[is_space]`). However, in training, `Paris` might have been tokenized as `[ space_Paris]`. Because the trailing space was consumed by the previous token, the model cannot generate `[ space_Paris]` and gets forced into sub-optimal generation paths. Chat templates with explicit boundaries (`<|im_end|>\n<|im_start|>assistant\n`) provide deterministic, unambiguous state boundaries that prevent token fragmentation.

### Q3: What happens to the GPU memory when sequence length doubles from 2,048 to 4,096 tokens during standard self-attention?
**Answer**:
1. **Activation Memory for Attention Matrix**: Vanilla attention stores the attention score matrix $S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times S \times S}$. Doubling sequence length $S$ quadruples ($4\times$) the attention matrix memory ($O(S^2)$).
2. **FlashAttention Optimization**: With FlashAttention-2, intermediate attention matrices are never written to HBM; they are computed online in tile chunks in fast SRAM. Thus, FlashAttention scales memory **linearly** ($O(S)$) with sequence length instead of quadratically.
