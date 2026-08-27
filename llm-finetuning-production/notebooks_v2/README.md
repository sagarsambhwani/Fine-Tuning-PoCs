# 🚀 Phase 2: Post-Training Alignment & High-Throughput Serving Sprint (Days 8–14)

Welcome to **Phase 2 (`notebooks_v2`)**! Having mastered foundational LoRA, QLoRA, SFT, and basic FastAPI deployment in Days 1–7, this advanced curriculum covers the modern **state-of-the-art post-training and inference engineering stack** used by leading AI teams.

---

## 🗺️ 7-Day Hands-On Syllabus

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHASE 2: ADVANCED ALIGNMENT & SERVING                                  │
│                                                                                                        │
│  [Day 8: DPO]        ──► Pairwise Preference Alignment (TRL DPOTrainer, implicit reward beta)          │
│  [Day 9: ORPO & KTO] ──► Reference-free single-step alignment & Kahneman-Tversky prospect theory       │
│  [Day 10: GRPO]      ──► Group Relative Policy Optimization (The DeepSeek-R1 reasoning revolution)     │
│  [Day 11: vLLM]      ──► PagedAttention & Continuous Batching (OpenAI-compatible high-throughput)      │
│  [Day 12: Multi-LoRA]──► Concurrent adapter routing on a single shared base model                     │
│  [Day 13: AWQ & FP8] ──► Activation-aware 4-bit serving quantization & kernel benchmarking             │
│  [Day 14: Agentic]   ──► Grammar-constrained logits masking (Outlines) & Multi-turn Tool Calling       │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗓️ Day-by-Day Breakdown

### 🎯 Day 8: Direct Preference Optimization (DPO)
* **Notebook**: [`notebooks_v2/day08_dpo_alignment.ipynb`](day08_dpo_alignment.ipynb)
* **Report**: [`reports/day08_dpo_alignment_report.md`](../reports/day08_dpo_alignment_report.md)
* **Status**: **Completed & Empirically Verified** ✅
* **Focus**: Moving beyond cross-entropy SFT to steer model tone, eliminate conversational preambles (`"Sure! Here is the JSON..."`), markdown fences (` ```json `), and enforce negative constraints.
* **Key Concepts**:
  * The mathematical limitation of SFT: SFT only learns positive likelihood $\sum \log P(y|x)$; it cannot penalize negative patterns.
  * Bradley-Terry preference model: $P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$.
  * DPO loss formulation: Eliminates the separate Reward Model and PPO value network by expressing the reward implicitly:
    $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
  * The role of temperature parameter $\beta=0.1$ (regularization strength against reference model drift).
* **Empirical Discovery**:
  * **DPO directly from Base**: 28.00% Preference Accuracy, Mean Reward Margin -2.0543 (Policy collapsed into story continuation because base model never had domain knowledge).
  * **DPO on top of SFT (The 2-Stage Standard)**: >95% Preference Accuracy, +1.8420 Mean Reward Margin, 100% clean zero-preamble JSON extraction.
* **Deliverable**: Generated 250 pairwise comparison triplets, trained QLoRA adapter via TRL `DPOTrainer`, and documented the 2-Stage alignment standard.

---

### 🎯 Day 9: Reference-Free Alignment (ORPO & KTO)
* **Notebook**: `notebooks_v2/day09_orpo_kto.ipynb`
* **Focus**: Eliminating the GPU memory overhead of keeping a frozen reference model in VRAM during alignment.
* **Key Concepts**:
  * **ORPO (Odds Ratio Preference Optimization)**: Combines standard SFT cross-entropy loss with an odds-ratio penalty in a single training loop:
    $$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}$$
  * **KTO (Kahneman-Tversky Optimization)**: Direct alignment on unpaired binary feedback (`desirable` vs. `undesirable`) matching human utility curves.
* **Deliverable**: Train an ORPO model in a single pass on a consumer GPU without allocating dual-model VRAM.

---

### 🎯 Day 10: Reinforcement Learning for Reasoning (GRPO)
* **Notebook**: `notebooks_v2/day10_grpo_reasoning.ipynb`
* **Focus**: The breakthrough reinforcement learning algorithm behind **DeepSeek-R1**.
* **Key Concepts**:
  * Why standard PPO is memory-prohibitive: It requires an Actor, Critic (Value Network), Reference Model, and Reward Model simultaneously.
  * **GRPO (Group Relative Policy Optimization)**:
    * Samples a group of $G$ candidate completions for each prompt: $\{o_1, o_2, \dots, o_G\}$.
    * Computes rule-based deterministic rewards (e.g. unit test passes, exact JSON matching, math verification).
    * Normalizes rewards within the group (mean 0, variance 1) to derive advantage $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$, eliminating the Critic network entirely!
  * Training `<think> ... </think>` Chain-of-Thought reasoning.
* **Deliverable**: Implement a rule-based GRPO loop teaching the model to verify its own intermediate extraction reasoning steps before outputting final JSON.

---

### 🎯 Day 11: Production Serving with vLLM & PagedAttention
* **Notebook**: `notebooks_v2/day11_vllm_serving.ipynb`
* **Focus**: Transitioning from PyTorch sequential generation to production-scale serving engines.
* **Key Concepts**:
  * **PagedAttention**: Treating the Key-Value (KV) cache like virtual memory pages, cutting memory waste from 60–80% down to under 4%.
  * **Continuous Batching (Iteration-level scheduling)**: Interleaving incoming requests dynamically at every decoding step rather than padding to the longest sequence.
  * **Prefix Caching**: Automatic caching and reuse of KV states for shared system prompts.
* **Deliverable**: Spin up an OpenAI-compatible vLLM serving daemon, benchmark concurrency under 16 simultaneous client streams, and measure Time-To-First-Token (TTFT).

---

### 🎯 Day 12: Concurrent Multi-LoRA Dynamic Serving
* **Notebook**: `notebooks_v2/day12_multilora_serving.ipynb`
* **Focus**: Serving dozens of specialized fine-tuned models on a single GPU without duplicating base weights.
* **Key Concepts**:
  * How S-LoRA / vLLM LoRA dynamically batches heterogeneous adapter requests in the same forward pass.
  * Routing requests via HTTP headers: `model="qwen-order-extractor"` vs `model="qwen-support-classifier"` vs `model="qwen-sql-generator"`.
  * VRAM footprint comparison: 1 Shared Base Model + 5 Adapters (3.3 GB) vs. 5 Merged Standalone Models (15.5 GB).
* **Deliverable**: Serve two distinct adapters on a single vLLM instance and run concurrent multi-task requests against them.

---

### 🎯 Day 13: Advanced Serving Quantization (AWQ & FP8)
* **Notebook**: `notebooks_v2/day13_awq_quantization.ipynb`
* **Focus**: Achieving 4-bit serving speedups without perplexity degradation.
* **Key Concepts**:
  * Why QLoRA (NF4) is for training, while AWQ / GPTQ are for serving.
  * **AWQ (Activation-aware Weight Quantization)**: Finding the top 1% salient weight channels by profiling input activation distributions and protecting them in higher precision while quantizing the rest to INT4.
  * Native **FP8 (Float8 E4M3 / E5M2)** serving on modern Tensor Cores.
* **Deliverable**: Quantize our Day 6 merged model into AWQ format with `AutoAWQ`, verify that zero JSON extraction accuracy is lost, and measure 2.5x throughput gain on vLLM.

---

### 🎯 Day 14: Structured Logits Masking & Agentic Function Calling
* **Notebook**: `notebooks_v2/day14_agentic_function_calling.ipynb`
* **Focus**: Making hallucinations mathematically impossible through constrained decoding, and building multi-turn agentic tool execution.
* **Key Concepts**:
  * **Grammar-Guided Decoding (Outlines / Guidance)**: Compiling a regex or Pydantic schema into a Finite State Machine (FSM), masking out invalid token logits at each forward step so the model can physically *only* generate valid syntax.
  * Multi-turn tool calling loops:
    `User Request -> LLM emits Tool Call -> Python executes API -> LLM receives Observation -> LLM synthesizes Final Answer`.
* **Deliverable**: Build a complete agent that extracts orders, calls a real inventory database API, and handles out-of-stock scenarios gracefully.

---

## 🛠️ Required Stack for Phase 2

```bash
pip install -q trl vllm autoawq outlines openai datasets peft accelerate
```
