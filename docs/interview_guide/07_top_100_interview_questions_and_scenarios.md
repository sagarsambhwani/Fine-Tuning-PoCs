# 🏆 Module 07: 100+ Interview Questions, Coding Implementations & System Design

This module is the ultimate test bank for **Senior and Staff AI Engineer interviews**. It contains comprehensive conceptual questions, step-by-step mathematical derivations, complete PyTorch coding implementations from scratch, production debugging post-mortems, and end-to-end Machine Learning System Design (MLSD) case studies.

---

## 🎯 Part 1: Top 50 Rapid-Fire Interview Questions & Answers

### 1. Foundations & Architecture
1. **Q: What is the primary difference between causal language modeling and masked language modeling?**
   * **A**: Causal LM (e.g., GPT, LLaMA) uses a lower-triangular causal attention mask to predict the next token $P(x_t \mid x_{<t})$ autoregressively. Masked LM (e.g., BERT) allows bidirectional attention to predict randomly masked tokens $P(x_{\text{mask}} \mid x_{\backslash \text{mask}})$. Causal LMs are natural text generators, whereas MLMs excel at feature extraction and classification.
2. **Q: Why does standard Softmax attention have $O(N^2)$ time and memory complexity?**
   * **A**: Because every token computes an attention score against every other token in the sequence ($Q K^\top \in \mathbb{R}^{N \times N}$), requiring storing and computing an $N \times N$ matrix.
3. **Q: How does Grouped-Query Attention (GQA) reduce KV cache memory without hurting performance?**
   * **A**: It shares a single Key and Value head across a group of Query heads (e.g., 8 Query heads per 1 KV head), reducing KV cache size by $8\times$ while retaining near-full Multi-Head Attention capacity.
4. **Q: What is Rotary Position Embedding (RoPE) and why is it superior to learned positional embeddings?**
   * **A**: RoPE encodes positions by rotating Query and Key vectors in complex 2D planes. The inner product $\langle \mathbf{q}_m, \mathbf{k}_n \rangle$ naturally depends only on the relative distance $(m-n)$, enabling smooth extrapolation to longer sequence lengths.
5. **Q: What is the purpose of the EOS (End of Sequence) token during training?**
   * **A**: It teaches the model when to terminate generation. Missing EOS tokens during SFT causes the model to generate infinite repetitive loops until reaching `max_new_tokens`.

### 2. PEFT, LoRA & QLoRA
6. **Q: What is the Intrinsic Rank Hypothesis?**
   * **A**: The empirical discovery that weight updates $\Delta W$ during domain adaptation reside on a low-dimensional manifold, meaning a low-rank matrix $B \cdot A$ ($r \ll d$) can capture $>95\%$ of full fine-tuning performance.
7. **Q: Why is matrix $B$ initialized to zero in LoRA?**
   * **A**: To guarantee that $\Delta W = B \cdot A = 0$ at step 0, preserving the exact pre-trained base model behavior at the start of training.
8. **Q: What is the role of the $\alpha$ parameter in LoRA?**
   * **A**: $\alpha$ acts as a constant scaling multiplier ($\frac{\alpha}{r}$). Keeping $\frac{\alpha}{r}$ constant decouples learning rate dynamics from changes in rank $r$.
9. **Q: How does NormalFloat4 (NF4) differ from standard INT4?**
   * **A**: INT4 uses linear spacing. NF4 uses quantile bins constructed such that each bucket has equal probability under a standard normal distribution $\mathcal{N}(0, 1)$, minimizing information-theoretic quantization error.
10. **Q: What are Paged Optimizers in QLoRA?**
    * **A**: They leverage CUDA Unified Memory to automatically page optimizer states from GPU VRAM to CPU RAM during activation memory spikes, preventing sudden Out-Of-Memory (OOM) crashes.
11. **Q: How does DoRA improve upon LoRA?**
    * **A**: DoRA decomposes weights into independent Magnitude and Direction components, updating Direction via low-rank matrices while scaling Magnitude, mirroring the learning dynamics of full fine-tuning.
12. **Q: Can you merge QLoRA weights directly into a 4-bit base model?**
    * **A**: No. Merging requires dequantizing the 4-bit base weights to FP16/BF16, adding $\Delta W = \frac{\alpha}{r}BA$, and saving as an FP16 checkpoint or re-quantizing to AWQ/GGUF.

### 3. SFT & Data Engineering
13. **Q: Why must input prompt tokens be masked (`labels = -100`) during SFT?**
    * **A**: To prevent the model from wasting gradient updates on predicting the user's prompt, focusing 100% of capacity on conditional response generation.
14. **Q: What is Sample Packing in SFT?**
    * **A**: Concatenating multiple short conversations into a single fixed context window (e.g., 4096 tokens) separated by EOS delimiters, eliminating wasted computation on `<pad>` tokens.
15. **Q: What is SemDeDup?**
    * **A**: Semantic Deduplication that embeds text, clusters examples using K-Means, and prunes near-identical semantic duplicates based on cosine similarity thresholds.
16. **Q: How does Evol-Instruct create complex synthetic datasets?**
    * **A**: It uses an LLM to evolve seed prompts in-depth (adding constraints, deepening technical detail) and in-breadth (mutating topics).
17. **Q: What is Catastrophic Forgetting and how do you mitigate it?**
    * **A**: The loss of general capabilities when fine-tuning on a narrow task. Mitigated by mixing 5–10% general instruction replay data or using PEFT.

### 4. Preference Alignment & RL (DPO / GRPO / RLHF)
18. **Q: What is the Bradley-Terry preference model?**
    * **A**: A probabilistic model defining the probability that response $y_w$ is preferred over $y_l$ as $P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$.
19. **Q: Why does DPO eliminate the need for an explicit Reward Model?**
    * **A**: DPO mathematically substitutes the optimal reward function $r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ directly into the Bradley-Terry objective, yielding a closed-form cross-entropy loss.
20. **Q: What is the purpose of the $\beta$ parameter in DPO?**
    * **A**: It controls the penalty against drifting from reference policy $\pi_{\text{ref}}$. A lower $\beta$ pushes preference optimization harder but risks degeneration.
21. **Q: What is Length Bias in DPO and how is it addressed?**
    * **A**: DPO sums token log-probabilities, favoring longer outputs. Solved by length-normalized scoring (SimPO) or length-penalized rewards.
22. **Q: How does GRPO differ from PPO?**
    * **A**: GRPO eliminates the Critic/Value model entirely. It samples a group of $G$ outputs per prompt and computes relative advantage normalized against the group mean.
23. **Q: What is the difference between a PRM and an ORM?**
    * **A**: Outcome Reward Models (ORMs) score only the final response. Process Reward Models (PRMs) score every intermediate reasoning step, enabling Monte Carlo Tree Search (MCTS).

### 5. Distributed Systems & Scale
24. **Q: What are the three components of AdamW optimizer state memory?**
    * **A**: FP32 Master Weights (4 bytes/param), FP32 Momentum (4 bytes/param), and FP32 Variance (4 bytes/param) = 12 bytes per parameter.
25. **Q: Explain DeepSpeed ZeRO stages 1, 2, and 3.**
    * **A**: ZeRO-1 shards optimizer states ($4\times$ memory reduction); ZeRO-2 shards optimizer states and gradients ($8\times$ reduction); ZeRO-3 shards optimizer states, gradients, and model weights ($N\times$ linear reduction).
26. **Q: What is Activation Checkpointing?**
    * **A**: Discarding intermediate layer activations during the forward pass and recomputing them during the backward pass, trading ~25% compute for ~70% VRAM savings.
27. **Q: Why does Tensor Parallelism require NVLink?**
    * **A**: TP performs an `AllReduce` communication after every single transformer layer. High-speed NVLink (900 GB/s) is required to prevent extreme communication bottlenecks.
28. **Q: What is the data transfer cost per GPU in Ring-AllReduce?**
    * **A**: $2 \times \frac{N-1}{N} \times S$, which approaches $2S$ as $N \to \infty$, making bandwidth cost independent of cluster size.

### 6. Serving, Merging & Quantization
29. **Q: How does PagedAttention in vLLM eliminate memory fragmentation?**
    * **A**: It allocates KV cache memory in non-contiguous physical pages (like OS virtual memory), cutting memory waste from ~70% to under 4%.
30. **Q: What is RadixAttention in SGLang?**
    * **A**: It maintains a radix tree of KV caches across requests, enabling instant re-use of shared system prompts and multi-turn conversation history.
31. **Q: How does AWQ differ from GPTQ?**
    * **A**: AWQ protects the top 1% salient weight channels based on activation magnitudes before INT4 quantization, whereas GPTQ optimizes weight rounding against the inverse Hessian.
32. **Q: What is Task Arithmetic in model merging?**
    * **A**: Computing task vectors $\tau = \theta_{\text{ft}} - \theta_{\text{base}}$ and linearly combining them ($\theta_{\text{merged}} = \theta_{\text{base}} + \sum \lambda_i \tau_i$) without retraining.
33. **Q: What is TIES-Merging?**
    * **A**: A model merging technique that trims small-magnitude noise, elects majority parameter signs, and merges disjoint parameters to eliminate multi-task interference.

---

## 💻 Part 2: Complete Coding Implementations from Scratch

### 1. Custom LoRA Linear Layer with Merge & Unload (Pure PyTorch)

```python
import math
import torch
import torch.nn as nn

class LoRALinearLayer(nn.Module):
    """
    Production-grade LoRA implementation in pure PyTorch.
    Supports dynamic rank, scaling factor, and zero-latency weight merging.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable Low-Rank Adapters
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Zero initialization for B ensures Delta W = 0 at step 0
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged or self.rank == 0:
            return self.base_layer(x)

        base_out = self.base_layer(x)
        lora_out = self.dropout(x)
        # Low rank matrix multiplication: (x @ A.T) @ B.T * scaling
        lora_out = torch.matmul(lora_out, self.lora_A.t())
        lora_out = torch.matmul(lora_out, self.lora_B.t()) * self.scaling
        return base_out + lora_out

    def merge(self):
        """Fuses adapter weights directly into base weight matrix."""
        if not self.merged and self.rank > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge(self):
        """Recovers original base weight matrix."""
        if self.merged and self.rank > 0:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= delta_w
            self.merged = False
```

---

### 2. Custom Completion-Only Data Collator for SFT with Loss Masking

```python
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class CompletionOnlyDataCollator:
    """
    Data collator that masks prompt tokens with labels = -100 so that
    loss is strictly computed on the assistant's completion.
    """
    tokenizer: Any
    response_template: str = "<|im_start|>assistant\n"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )
        template_len = len(response_token_ids)

        for i in range(len(labels)):
            seq = batch["input_ids"][i].tolist()
            labels[i, :] = -100  # Default mask everything
            
            # Find the starting index of the assistant response template
            for idx in range(len(seq) - template_len + 1):
                if seq[idx : idx + template_len] == response_token_ids:
                    start_response_idx = idx + template_len
                    # Unmask tokens from assistant start to end of sequence
                    labels[i, start_response_idx:] = batch["input_ids"][i, start_response_idx:]
                    break

            # Ensure padding tokens are always masked
            if self.tokenizer.pad_token_id is not None:
                labels[i][batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch
```

---

## 🔬 Part 3: Production Debugging Scenarios (Post-Mortem Cases)

### Scenario 1: Training Loss Suddenly Explodes to `NaN` at Step 420
* **Context**: You are fine-tuning a 14B model using standard FP16 mixed precision on 8x A100 GPUs.
* **Diagnosis Steps**:
  1. Check gradient norms logged to Weights & Biases right before step 420.
  2. Inspect the data batch at step 420 for corrupted empty strings, extremely long repetitive tokens, or invalid Unicode sequences.
  3. Verify Attention Softmax logit scale ($QK^\top / \sqrt{d_k}$) exceeding FP16 maximum value ($65,504$).
* **Solution**:
  * Switch precision from `fp16=True` to `bf16=True`.
  * Set `max_grad_norm = 1.0` in `TrainingArguments`.
  * Filter empty text instances from the dataset pipeline.

---

### Scenario 2: Severe Catastrophic Forgetting After Legal Domain Fine-Tuning
* **Context**: A 7B model fine-tuned on 20,000 legal contracts scores 92% on legal extraction but drops from 68% to 24% on GSM8K math reasoning and begins hallucinating simple facts.
* **Root Cause**: Over-aggressive domain updates on full weights with too high learning rate ($5 \times 10^{-5}$) without general replay data.
* **Solution**:
  1. Switch to **LoRA ($r=16, \alpha=32$)** to freeze core base model representations.
  2. Apply **Data Replay**: Mix 10% general reasoning data (UltraChat / OpenHermes) into the legal dataset.
  3. Reduce learning rate to $1 \times 10^{-4}$ with a cosine decay scheduler and early stopping.

---

## 🏛️ Part 4: Machine Learning System Design (MLSD) Case Studies

### System Design 1: Enterprise Multi-Tenant Multi-LoRA Platform

```
                                  ENTERPRISE MULTI-LORA ARCHITECTURE
                                  
    [ 10,000 Inbound Requests/sec ]
                  │
                  ▼
       [ API Gateway & Router ] ──► Extract Tenant ID & Model Tag
                  │
                  ▼
        [ SGLang / vLLM Engine ]
                  │
  ┌───────────────┴───────────────┐
  │ Physical GPU VRAM             │
  │  [ Base Model: Qwen-72B FP8 ] │  <-- Single Shared Base Instance (80 GB)
  │                               │
  │ Dynamic LoRA Cache (SRAM):    │
  │  • Tenant A Adapter (50 MB)   │  <-- Swapped in < 2ms via custom CUDA kernels
  │  • Tenant B Adapter (50 MB)   │
  │  • Tenant C Adapter (50 MB)   │
  └───────────────┬───────────────┘
                  │
                  ▼
        [ High-Speed Redis NVMe ] ◄── 1,000+ Tenant LoRA Adapter Weights (S3 Backup)
```

* **Key Requirements**: Serve 1,000 distinct enterprise fine-tuned models with $< 30\text{ms}$ Time-To-First-Token and $< \$0.001$ cost per request.
* **Core Architecture**:
  1. Deploy a **single shared base foundation model** (e.g., LLaMA-3.3-70B in FP8) across 2x H100 GPUs.
  2. Maintain a local NVMe and high-speed RAM tier holding 1,000 lightweight LoRA adapters ($50\text{ MB}$ each).
  3. Use **vLLM Multi-LoRA / S-LoRA dynamic kernel batching**: Incoming requests from different tenants are batched together into the same forward pass; base weights execute concurrently while custom LoRA GEMM kernels apply tenant-specific adapter matrices.
  4. Implement an LRU cache for high-frequency tenant adapters in GPU VRAM.
