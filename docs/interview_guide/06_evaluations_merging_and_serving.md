# 🚀 Module 06: Evaluation, Model Merging, Quantization & Production Serving

Building a fine-tuned model is only half the engineering equation. The other half is **evaluating domain robustness, merging model capabilities without retraining, quantizing for low-latency inference, and architecting high-throughput serving systems**.

---

## 1. LLM Evaluation & Benchmarking Methodologies

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE 3-TIER EVALUATION STACK                               │
│                                                                                        │
│  Tier 1: Deterministic Metrics ──► Perplexity, Exact Match (EM), JSON Schema Validity   │
│  Tier 2: Academic Benchmarks   ──► MMLU (Knowledge), GSM8K (Math), HumanEval (Coding)  │
│  Tier 3: LLM-as-a-Judge        ──► Pairwise Elo (MT-Bench, Arena-Hard) & Custom Rubrics│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. Academic Benchmarks & What They Measure

| Benchmark | Domain | Metric | Target Quality Assessed |
| :--- | :--- | :--- | :--- |
| **MMLU / MMLU-Pro** | 57 Multi-domain subjects | 5-shot Multiple Choice Accuracy | Broad world knowledge & factual recall |
| **GSM8K / MATH** | Grade school & Olympiad Math | Exact Match (Chain of Thought) | Multi-step mathematical reasoning |
| **HumanEval / MBPP** | Python Coding Challenges | Pass@1 (Unit test verification) | Algorithmic code synthesis |
| **MT-Bench / Arena-Hard**| Multi-turn Open-Ended Prompts| GPT-4 Judge Score (1–10) | Conversational coherence & instruction follow |

### B. LLM-as-a-Judge: Best Practices & Bias Mitigation
Using a frontier model (e.g., GPT-4o) to judge candidate model outputs requires rigorous guardrails against known biases:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               JUDGE BIAS MITIGATIONS                                   │
│                                                                                        │
│  1. Position Bias      ──► Swap response order (A/B and B/A) and average scores.       │
│  2. Verbosity Bias     ──► Instruct judge to penalize fluff; normalize score by length.│
│  3. Self-Enhancement   ──► Never reveal model identity / signature in prompt.          │
│  4. Score Calibration  ──► Provide explicit 5-level scoring rubrics with anchor examples│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Production-Grade LLM-as-a-Judge Prompt Template

```markdown
You are an expert evaluator judging the accuracy, adherence, and clarity of an AI response.

[Context / Ground Truth]:
{context}

[User Query]:
{query}

[Model Response]:
{response}

Evaluate the response on a scale of 1 to 5 using the following criteria:
- Score 5: Flawless accuracy, fully follows all constraints, clear reasoning, zero hallucinations.
- Score 3: Partially correct, minor factual inaccuracy or missed a secondary constraint.
- Score 1: Factually incorrect, severe hallucination, or completely failed instruction.

Provide a step-by-step critique followed by your final score in JSON format:
```json
{"critique": "<step_by_step_reasoning>", "score": <int_1_to_5>}
```
```

---

## 2. Model Merging: Combining Expertise Without Retraining

Model Merging combines the weights of multiple specialized models into a single checkpoint without requiring any GPU training compute.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODEL MERGING TECHNIQUES                                 │
│                                                                                        │
│  1. SLERP (Spherical Linear Interpolation) ──► Preserves geometric vector magnitude   │
│  2. Task Arithmetic (Task Vectors)         ──► θ_target = θ_base + λ_1 τ_1 + λ_2 τ_2   │
│  3. TIES-Merging                           ──► Trims noise, resolves sign interference │
│  4. DARE (Drop And REscale)                ──► Bernoulli sparsity pruning (90% dropped)│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. Task Arithmetic & Task Vectors
A task vector $\tau_i$ represents the directional delta learned during fine-tuning on task $i$:

$$\tau_i = \theta_{\text{ft}, i} - \theta_{\text{base}}$$

To create a multi-task model combining Coding ($\tau_{\text{code}}$) and Math ($\tau_{\text{math}}$):

$$\theta_{\text{merged}} = \theta_{\text{base}} + \lambda_{\text{code}} \tau_{\text{code}} + \lambda_{\text{math}} \tau_{\text{math}}$$

### B. TIES-Merging (*Yadav et al., 2023*)
When merging multiple task vectors, parameter interference degrades capabilities. TIES solves this in 3 steps:
1. **Trim**: Keeps only the top $k\%$ ($20\%$) largest magnitude parameter updates in each task vector, setting the rest to $0$.
2. **Elect Signs**: Resolves conflicting updates (where Task A wants $+0.5$ and Task B wants $-0.4$) by computing the majority sign across models.
3. **Disjoint Merge**: Averages only the parameters that align with the elected majority sign.

### C. DARE (Drop And REscale)
Applies extreme random dropout (e.g., $p = 0.90$) to task vectors $\tau_i$ and rescales the remaining $10\%$ by $\frac{1}{1-p}$. Eliminates up to $90\%$ of delta weights with zero accuracy loss, allowing seamless merging of 10+ models.

---

## 3. Post-Training Quantization (PTQ) for Production Serving

```
Weight Precision vs Memory & Speed:
• FP16 / BF16:  16 bits/param (2.0 GB per 1B params)  ──► Baseline Training
• FP8 (E4M3):    8 bits/param (1.0 GB per 1B params)  ──► Modern Standard Serving
• AWQ / GPTQ:    4 bits/param (0.5 GB per 1B params)  ──► High Compression / Low VRAM
• GGUF (Q4_K_M): 4.5 bits/param                       ──► CPU / Metal / Edge Inference
```

| Quantization Method | Target Layer | Calibration Data Required? | Key Mechanism | Best Serving Engine |
| :--- | :--- | :--- | :--- | :--- |
| **AWQ** (*Activation-aware Weight Quantization*) | Weights ($W$) | **Yes** (128 samples) | Protects the top $1\%$ salient weights that align with high-magnitude activation channels. | vLLM / SGLang |
| **GPTQ** (*Exact Second-Order Optimization*) | Weights ($W$) | **Yes** | Uses second-order Taylor expansion inverse Hessian ($H^{-1}$) to compensate for quantization errors. | vLLM / AutoGPTQ |
| **GGUF (k-quants)** | Weights ($W$) | No / Optional | Block-wise mixed quantization (Q4_K_M uses 4-bit for attention, 6-bit for critical MLP layers). | llama.cpp / Ollama |
| **FP8 (W8A8)** | Weights & Activations | Optional (Dynamic) | Native Hopper/Ada hardware FP8 matrix acceleration. | vLLM / TensorRT-LLM |

---

## 4. High-Throughput Production Serving Architectures

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODERN LLM SERVING STACK                                 │
│                                                                                        │
│   [ Client Inbound Requests ]                                                          │
│                │                                                                       │
│                ▼                                                                       │
│   [ SGLang: RadixAttention ]  ◄── LRU Tree prefix caching (re-uses system prompt KV)   │
│                │                                                                       │
│                ▼                                                                       │
│   [ vLLM Serving Engine ]     ◄── Continuous Batching & Chunked Prefill                │
│        ├── PagedAttention     ◄── Zero memory fragmentation for KV Cache               │
│        ├── Multi-LoRA Manager ◄── S-LoRA / Punica dynamic adapter routing              │
│        └── Speculative Engine ◄── Small draft model (2x-3x speedup)                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### A. vLLM & PagedAttention
* **The Problem**: Traditional serving pre-allocates contiguous memory for the maximum sequence length (e.g., 4096 tokens), wasting **$60\%–80\%$ of KV cache VRAM** in memory fragmentation.
* **PagedAttention**: Manages KV cache like virtual memory in Operating Systems. KV cache is divided into fixed-size **pages/blocks (e.g., 16 tokens)** and mapped dynamically to non-contiguous physical GPU VRAM.
  * **Result**: Near-zero memory waste ($< 4\%$), enabling **$2\text{x}–4\text{x}$ larger concurrency batch sizes**.

### B. Multi-LoRA Dynamic Serving (S-LoRA / Punica)
Instead of deploying 50 distinct base models for 50 fine-tuned customer use cases (requiring 50 GPUs):
* Deploy **a single base model instance in GPU VRAM**.
* Maintain 50 LoRA adapters ($50\text{ MB}$ each) in host RAM.
* Dynamically load and batch LoRA adapters on-the-fly per incoming query token inside custom CUDA kernels (Punica / S-LoRA).
* **Cost Reduction**: $50\times$ infrastructure cost reduction.

### C. Speculative Decoding
LLM generation is **memory bandwidth bound** (loading weights for 1 token generation takes as much time as loading weights for 64 tokens).
* **Mechanism**: A tiny draft model (e.g., 0.5B) rapidly generates $K=5$ candidate tokens. The large target model (e.g., 70B) verifies all 5 tokens **in a single parallel forward pass**.
* **Speedup**: $2.0\text{x}–3.5\text{x}$ throughput improvement with **zero loss in mathematical output distribution**.

---

## 🎯 Top Interview Q&A on Evaluation, Merging & Serving

### Q1: How does AWQ differ from GPTQ, and why is AWQ preferred for online serving?
**Answer**:
* **GPTQ** optimizes weight rounding by minimizing mean squared error against the inverse Hessian matrix ($H^{-1}$) of activations. However, GPTQ treats all weights uniformly, occasionally corrupting critical outlier channels.
* **AWQ** recognizes that only **$0.1\%–1\%$ of weight channels are salient** (handling high-magnitude activation spikes). AWQ protects these critical channels by scaling them before INT4 quantization.
* **Serving Benefit**: AWQ weights retain higher fidelity at low bits and enable significantly faster on-the-fly dequantization kernels (e.g., Marlin/AWQ CUDA kernels in vLLM) compared to GPTQ.

### Q2: What is "Chunked Prefill" and why is it essential for low-latency serving?
**Answer**:
In LLM serving, incoming requests have two phases: **Prefill** (compute-bound, processing input prompt tokens) and **Decode** (memory-bandwidth bound, generating output tokens one by one).
A giant input prompt (e.g., 32k tokens) monopolizes the GPU Tensor Cores for seconds, causing high latency spikes (Time-To-First-Token and Inter-Token Latency) for all existing concurrent decoding streams.
**Chunked Prefill** breaks massive prefill prompts into smaller chunks (e.g., 512 tokens) and co-schedules them alongside decoding steps, ensuring stable latency and high GPU utilization.
