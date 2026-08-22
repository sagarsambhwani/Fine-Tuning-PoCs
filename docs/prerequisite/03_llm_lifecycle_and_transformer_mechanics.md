# 🚀 Fine-Tuning Prerequisites: The Complete LLM Lifecycle & Progressive Transformer Mechanics

> **Level 0 → Level 2 Lifecycle Blueprint**: This document bridges foundational deep learning with the advanced modules in the interview guide. It walks through every stage of the LLM lifecycle in chronological order: **Text → Tokens → Embeddings → Attention → Pre-training → SFT → LoRA/QLoRA → Alignment → Scaling → Serving → Evaluation**.

---

## 🗺️ The Continuous LLM Lifecycle Mental Model

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE LLM LIFECYCLE ROADMAP                              │
│                                                                                        │
│  [1. Raw Text] ──► [2. Tokens] ──► [3. Embeddings] ──► [4. Transformer Blocks]         │
│                                                                │                       │
│  [8. Serving] ◄── [7. Evaluation] ◄── [6. Alignment] ◄── [5. Pre-training / SFT]      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. What an LLM Actually Does

### A. Step 1: Text $\to$ Tokens
Computers and neural networks cannot process raw characters or words directly. A **Tokenizer** splits text into chunks called **Tokens** and assigns each token a unique integer **Token ID** from its vocabulary.

```
Raw Text:    "The cat is playing"
Tokenized:   ["The", " cat", " is", " play", "ing"]
Token IDs:   [ 464,   3797,   318,    711,    278 ]
```

* **Vocabulary**: The fixed dictionary of tokens the model knows (e.g., 32,000 tokens in LLaMA-1, 128,000 in LLaMA-3).
* **Subword Splitting**: Words not seen during tokenizer training are broken into smaller subword pieces (e.g., `"playing"` $\to$ `["play", "ing"]`), preventing Out-of-Vocabulary errors.

### B. Step 2: Tokens $\to$ Embeddings
Each integer Token ID is looked up in an **Embedding Matrix** to retrieve a dense numerical vector:

```
Token ID: 3797 (" cat")
     │
     ▼ (Look up row 3797 in Embedding Matrix)
Embedding Vector: [ 0.12, -0.45, 0.89, ..., 0.03 ]  <-- Shape: (1, 768)
```

### C. Step 3: Predicting the Next Token
An autoregressive LLM is fundamentally a **Next-Token Probability Predictor**:

```
Input Context: "The cat sat on the"
                          │
                          ▼
            Transformer Neural Network
                          │
                          ▼
                 Unnormalized Logits
                          │
                          ▼ (Softmax Function)
            Next-Token Probability Distribution
              ├── "mat"   : 72%
              ├── "couch" : 14%
              ├── "floor" :  8%
              └── "the"   :  1%
```

---

## 2. Transformer Fundamentals & Attention Intuition

### A. Why Does Attention Exist?
In a sentence, a word's meaning depends heavily on its context:
> *"The dog chased the ball because **it** was moving."*

What does **"it"** refer to? (The ball). How does the neural network know to link "it" to "ball" rather than "dog"? **Self-Attention** allows every token to look at and extract information from every other token in the sequence.

### B. Queries, Keys, and Values (Q, K, V)
Instead of abstract mathematical variables, think of attention like a database search:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE Q, K, V ANALOGY                                       │
│                                                                                        │
│  • Query (Q): What information am I (the current token) looking for?                   │
│  • Key   (K): What information do I (other tokens) contain or advertise?               │
│  • Value (V): What actual content will I provide if you select me?                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

```
                     Attention Calculation Formula:
             Attention(Q, K, V) = Softmax( (Q · Kᵀ) / √d_k ) · V

  1. Q · Kᵀ   ──► Compatibility Scores (How relevant is token J to token I?)
  2. / √d_k   ──► Scaling factor (Prevents dot products from exploding into flat Softmax)
  3. Softmax  ──► Turns raw scores into probability weights (summing to 1.0)
  4. · V      ──► Blends the Value vectors according to the computed weights
```

### C. Multi-Head Attention (MHA) $\to$ Grouped-Query Attention (GQA)
* **Multi-Head Attention (MHA)**: Instead of computing a single attention calculation, the model runs multiple independent attention "heads" in parallel. Head 1 might focus on grammar, Head 2 on subject-verb agreement, and Head 3 on pronoun references.
* **Grouped-Query Attention (GQA)**: In generative serving, storing Key and Value states for every head consumes too much VRAM. GQA shares a single Key/Value head across multiple Query heads, **slashing KV Cache memory by $8\times$** while maintaining near-identical quality.

### D. Causal Attention Masking
When generating text autoregressively, the model must **never peek into future tokens**:

```
Token 1 ("The")  ──► Can ONLY see ["The"]
Token 2 ("cat")  ──► Can ONLY see ["The", "cat"]
Token 3 ("sat")  ──► Can ONLY see ["The", "cat", "sat"]
```
This is enforced by applying an upper-triangular mask of $-\infty$ to the attention score matrix before Softmax.

---

## 3. Training an LLM: Pre-Training vs. SFT

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PRE-TRAINING VS FINE-TUNING                               │
│                                                                                        │
│  Pre-Training (Base Model):                                                            │
│  • Trained on trillions of web tokens.                                                 │
│  • Prompt: "How to bake a cake?"                                                       │
│  • Completion: "...Chapter 2: Buying flour, Chapter 3: Ovens..." (Document completion) │
│                                                                                        │
│  Supervised Fine-Tuning (SFT / Instruct Model):                                        │
│  • Trained on structured (Instruction, Response) dialogue pairs.                       │
│  • Prompt: "How to bake a cake?"                                                       │
│  • Completion: "Here is a step-by-step recipe: 1. Preheat oven to 350°F..."            │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### The SFT Loss Masking Rule (`labels = -100`)
During SFT, we format inputs using chat templates (e.g., `<|im_start|>user\n...\n<|im_start|>assistant\n...`).
* **Critical Rule**: Gradients and cross-entropy loss must **only be computed on the assistant's response tokens**.
* Prompt tokens are assigned a target label of `-100`, instructing PyTorch to skip them during loss calculation.

---

## 4. Parameter-Efficient Fine-Tuning: Full FT $\to$ LoRA $\to$ QLoRA

### The Scaling Dilemma
* A 7B parameter model contains $7,000,000,000$ weights.
* In Full Fine-Tuning with AdamW, storing weights, gradients, and optimizer states requires **$\approx 112\text{ GB}$ of VRAM** (exceeding standard 24GB or 80GB GPUs).

```
Full Fine-Tuning:
  Every single weight W is updated ──► Requires massive GPU VRAM (16 bytes/param)

LoRA (Low-Rank Adaptation):
  Keep base weights W_0 FROZEN ────► Learn a tiny decomposition update ΔW = (α/r) * (B · A)
  Only 0.1% to 1% of parameters are trainable!
```

### The LoRA Building Blocks Explained:
* $\mathbf{W_0}$: The original base model weights (Frozen, non-trainable).
* $\mathbf{A}$: A small bottleneck down-projection matrix (Initialized with Gaussian noise).
* $\mathbf{B}$: A small bottleneck up-projection matrix (Initialized with **zeros**, ensuring $\Delta W = 0$ at step 0).
* $\mathbf{r}$ (Rank): The compression dimension (Typical starting points: $r = 8, 16, 32, 64$).
* $\boldsymbol{\alpha}$ (Scaling): Multiplier keeping update strength stable across different ranks.

### QLoRA: Quantizing the Base Model
```
Standard LoRA:  [ 16-bit BF16 Base Model (14 GB) ] + [ Trainable LoRA Adapters (50 MB) ]
QLoRA:          [  4-bit NF4 Base Model (3.5 GB) ] + [ Trainable LoRA Adapters (50 MB) ]
                └── Slashes base model memory by 75% without degrading generation quality!
```

---

## 5. Preference Alignment: DPO & GRPO

```
SFT Model ──► Can generate fluent answers, but may hallucinate or follow bad instructions.
     │
     ▼ (Preference Alignment)
Aligned Model ──► Prefers safe, accurate, concise, and logically verified answers.
```

* **Direct Preference Optimization (DPO)**: Uses pairs of $(x, y_{\text{chosen}}, y_{\text{rejected}})$. Maximizes the log-probability of the preferred response while minimizing the rejected response, without needing a separate reward model or complex reinforcement learning loops.
* **Group Relative Policy Optimization (GRPO - DeepSeek-R1)**: A critic-free RL algorithm that samples a group of candidate responses per prompt, evaluates them with **verifiable deterministic rules** (code compilers, math solvers), and updates the policy based on group-normalized advantage.

---

## 6. GPU Memory Breakdown & Serving Dynamics

### A. The 7B Parameter Memory Rule
When an interviewer asks: *"Why can't I fine-tune a 7B model on my 16GB GPU?"*

$$\text{Memory} = \underbrace{2 \text{ bytes (Weights)}}_{\text{14 GB}} + \underbrace{2 \text{ bytes (Gradients)}}_{\text{14 GB}} + \underbrace{12 \text{ bytes (AdamW Master + Momentum + Variance)}}_{\text{84 GB}} = \mathbf{112 \text{ GB}}$$

### B. High-Throughput Serving & KV Caching
* **Prefill Phase**: Processes input prompt tokens in parallel (Compute-bound).
* **Decode Phase**: Generates tokens autoregressively one by one (Memory-bandwidth bound).
* **PagedAttention (vLLM)**: Organizes Key-Value cache in non-contiguous virtual memory blocks (like an OS), eliminating memory fragmentation and boosting serving concurrency by $2\text{x}–4\text{x}$.

---

## 🎯 Progressive Interview Checkpoints (Staircase Model)

Test your knowledge progressively at each milestone:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PROGRESSIVE CHECKPOINTS                                 │
│                                                                                        │
│  Level 0 (Beginner):     "What problem does attention solve?"                          │
│  Level 1 (Intermediate): "What are Q, K, and V, and how do they interact?"             │
│  Level 2 (Advanced):     "Why do we divide QKᵀ by √d_k before Softmax?"               │
│  Level 3 (Senior/Lead):  "How does Grouped-Query Attention reduce KV cache in vLLM?"   │
│  System Design:          "Design a serving pipeline for 1,000 fine-tuned LoRA tenants."│
└────────────────────────────────────────────────────────────────────────────────────────┘
```
