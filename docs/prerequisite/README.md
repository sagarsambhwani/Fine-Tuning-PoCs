# 📚 Fine-Tuning Prerequisites: Progressive Curriculum

This directory contains the foundational, progressive prerequisite curriculum designed to take you from core deep learning basics up to advanced LLM fine-tuning and interview mastery.

---

## 🗺️ Prerequisite Curriculum Structure

```
docs/prerequisite/
├── 📄 README.md                                             # Prerequisite Overview & Navigation
├── 📄 01_pedagogical_framework_and_design_principles.md    # 9-Step Learning Cascade, 4 Difficulty Levels & Chapter Blueprint
├── 📄 02_deep_learning_and_torch_foundations.md            # Tensors, Matrix Multiplication, Neural Nets & PyTorch Training Loop
└── 📄 03_llm_lifecycle_and_transformer_mechanics.md        # Tokens, Attention (Q/K/V), SFT, LoRA/QLoRA, DPO/GRPO & Serving Intuition
```

---

## 📑 Module Directory

| Part | Document Link | Core Topics & Learning Objectives |
| :--- | :--- | :--- |
| **01** | [**Pedagogical Framework & Design Principles**](01_pedagogical_framework_and_design_principles.md) | The 9-step learning progression (Problem $\to$ Intuition $\to$ Terminology $\to$ Mechanics $\to$ Math $\to$ Code $\to$ Trade-offs $\to$ Interview), 4 difficulty levels (Levels 0–3), the 3-tier interview answer structure (10s, 30s, 2min), strict terminology rules, and the 16-step chapter template. |
| **02** | [**Deep Learning & PyTorch Foundations**](02_deep_learning_and_torch_foundations.md) | Tensors & dimensions (`[Batch, Seq, Hidden]`), Matrix multiplication dimension matching rules ($XW$), Neural network mechanics (weights, forward pass, loss, gradients, backpropagation, optimizer, learning rate), and the canonical PyTorch training loop. |
| **03** | [**LLM Lifecycle & Transformer Mechanics**](03_llm_lifecycle_and_transformer_mechanics.md) | The full LLM lifecycle: Text $\to$ Tokens (BPE) $\to$ Embeddings $\to$ Next-Token Prediction; Self-Attention intuition ($Q, K, V$ database analogy); Causal masking; Pretraining vs SFT (`labels=-100`); Full Fine-Tuning $\to$ LoRA ($W_0 + \Delta W$) $\to$ QLoRA; Preference alignment (DPO/GRPO); GPU memory math; and high-throughput serving (vLLM PagedAttention). |

---

## 🔗 Connection to Advanced Interview Modules

Once you have mastered the foundational concepts in this directory, proceed directly to the advanced technical modules in [`docs/interview_guide/`](../interview_guide/README.md):

* [**Module 01: Foundations & Tokenization**](../interview_guide/01_foundations_and_tokenization.md)
* [**Module 02: PEFT — LoRA, QLoRA & DoRA**](../interview_guide/02_peft_lora_and_qlora.md)
* [**Module 03: Instruction Tuning & SFT**](../interview_guide/03_instruction_tuning_and_sft.md)
* [**Module 04: Preference Alignment & RL (DPO / GRPO)**](../interview_guide/04_preference_alignment_dpo_grpo_rlhf.md)
* [**Module 05: Distributed Training & Scale**](../interview_guide/05_distributed_training_and_scale.md)
* [**Module 06: Evaluation, Merging & Serving**](../interview_guide/06_evaluations_merging_and_serving.md)
* [**Module 07: 100+ Interview Questions & System Design**](../interview_guide/07_top_100_interview_questions_and_scenarios.md)
