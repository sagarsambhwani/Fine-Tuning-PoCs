# 📊 Track 1: Pure LLM Fine-Tuning Pipeline (QLoRA + SFT) Empirical Report
## Domain Adaptation on Corporate Litigation Knowledge Graphs & Formal Ontologies

**Author & Experimenter**: Sagar Sambhwani  
**Target Model**: `Qwen/Qwen2.5-1.5B-Instruct` (Base) → `Qwen-1.5B-Legal-Graph-Adapter` (Track 1 Fine-Tuned)  
**Hardware & Environment**: Google Colab Tesla T4 GPU (16GB VRAM, CUDA 12.2)  
**Training Regime**: 4-bit NormalFloat (NF4) QLoRA ($r=16, \alpha=32$, all 7 linear projections), 3 Epochs (225 steps, effective batch size 16)  
**Dataset**: 1,200 Domain-Supervised Samples (Anti-Leakage Partitioned, Zero Target Leakage into Test Benchmark)  
**Test Suite**: 50 Unseen Litigation Cases per Benchmark Task ($N=150$ total evaluations)

---

## 1. Executive Summary & Benchmark Head-to-Head

In Phase 1, our zero-training baseline demonstrated that untuned foundation LLMs struggle significantly on structured graph tasks due to informal relation hallucinations and inverted edge directions. In **Track 1**, we evaluated the **Pure LLM Paradigm**: supervised fine-tuning (SFT) using parameter-efficient QLoRA.

The results show a massive improvement across ontology alignment and deductive extraction, while uncovering the **Pure LLM Inductive Ceiling** on topological link prediction:

| Benchmark Task / Metric | Untuned Base (`Qwen2.5-1.5B`) | Track 1 Fine-Tuned (`QLoRA SFT`) | Delta ($\Delta$) | Performance Status |
| :--- | :---: | :---: | :---: | :---: |
| **Task A: Triplet Extraction Precision** | $30.67\%$ | **$100.00\%$** | **$+69.33\%$** | 🎯 Perfect Schema Adherence |
| **Task A: Triplet Extraction Recall** | $21.33\%$ | **$100.00\%$** | **$+78.67\%$** | 🎯 Zero Missing Relations |
| **Task A: Triplet Macro F1 Score** | $25.07\%$ | **$100.00\%$** | **$+74.93\%$** | 🎯 $4.0\times$ Improvement |
| **Task B: 2-Hop Deductive Accuracy** | $92.00\%$ | **$100.00\%$** | **$+8.00\%$** | 🎯 Flawless 2-Hop Routing |
| **Task B: 3-Hop Deductive Accuracy** | $86.00\%$ | **$90.00\%$** | **$+4.00\%$** | 📈 Strong Deep-Chain Reasoning |
| **Task C: Link Prediction Accuracy** | $24.00\%$ | **$36.00\%$** | **$+12.00\%$** | ⚠️ Inductive Ceiling Hit |

---

## 2. Training Dynamics & Loss Convergence

Training was executed over **225 optimization steps** (3 epochs across 1,200 domain samples with gradient accumulation $= 4$ and batch size $= 4$).

### Convergence Log

| Step | Training Loss | Validation Loss | Entropy | Mean Token Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **50** | $0.1321$ | $0.1032$ | $0.1068$ | $97.18\%$ |
| **100** | $0.0851$ | $0.0814$ | $0.0908$ | $97.40\%$ |
| **150** | $0.0784$ | $0.0786$ | $0.0846$ | $97.38\%$ |
| **200** | $0.0748$ | $0.0766$ | $0.0797$ | $97.40\%$ |
| **225 (Final)** | **$0.0755$** | **$0.0766$** | **$0.0797$** | **$97.41\%$** |

- **Total Training Duration**: $31.51$ minutes on single Tesla T4 GPU.
- **Overfitting Diagnosis**: Validation loss smoothly decreased from $0.1032 \rightarrow 0.0766$ without diverging from training loss ($0.0755$), indicating optimal generalization on domain vocabulary without memorization artifacts.

---

## 3. Deep Dive by Task

### Task A: Information Extraction & Formal Ontology Alignment
- **Untuned Defect**: The base model generated arbitrary informal strings (e.g., `"sued"`, `"represented_by"`, `"presided over by"` instead of formal ontology predicates like `FILES_CLAIM_AGAINST`, `ADJUDICATED_BY`, `APPLIES_PRECEDENT`, `ESTABLISHES_REMEDY`, `LIABLE_FOR`).
- **Fine-Tuned Resolution**: The adapter internalized the closed-world ontology schema. Precision, Recall, and Macro F1 reached **$100.00\%$** across all 50 test cases with $0\%$ JSON parsing failures.

### Task B: Multi-Hop Deductive Reasoning
- **2-Hop Reasoning**: Jumped from $92.00\% \rightarrow 100.00\%$ ($+8.00\%$). The model reliably traverses intermediate nodes (e.g., `Plaintiff → Court → Presiding Judge`).
- **3-Hop Reasoning**: Improved from $86.00\% \rightarrow 90.00\%$ ($+4.00\%$). The model successfully resolves complex legal chains (e.g., `Claim → Adjudicator → Precedent Applied → Remedy Established`).

### Task C: Inductive Link Prediction & Graph Completion
- **Performance**: Improved from $24.00\% \rightarrow 36.00\%$ ($+12.00\%$).
- **The "Pure LLM Inductive Ceiling"**:
  - While fine-tuning improved relation discrimination over random guessing ($20\%$), the pure text-based LLM struggles to perform topological graph completion when entities share latent structural proximity but lack explicit contextual cues in the prompt.
  - LLMs process textual linear tokens, lacking an explicit permutation-invariant graph inductive bias (such as message passing over node neighbor aggregations).

---

## 4. Key Architectural & Production Insights

1. **Turing T4 Mixed Precision Stability**:
   - `fp16=False` and `bf16=False` in `SFTConfig` combined with `bnb_4bit_compute_dtype=torch.float16` and explicit `torch.float32` LoRA parameter casting prevents CUDA `GradScaler` unscale exceptions on Tesla T4 hardware.
2. **LoRA Efficiency**:
   - Only **$18.45\text{ M}$ parameters** ($1.20\%$ of base model) were trained, retaining full base conversational capabilities while mastering domain ontology constraints.
3. **Motivation for Track 2 (Graph Neural Networks & Hybrid Architectures)**:
   - The contrast between Task A/B ($90-100\%$) and Task C ($36\%$) empirically confirms the thesis of this research series: **Text-based LLMs excel at syntax, translation, and structured extraction, but require graph-native representations (e.g., GAT/GCN or GraphRAG hybrid retrieval) for high-accuracy inductive topological inference.**

---

## 5. Artifacts & Deliverables

- **Trained LoRA Weights**: `models/qwen-1.5b-legal-graph-adapter/`
- **Track 1 Notebook**: [`llm-graph-ontology/notebooks/02_track1_pure_llm_finetuning.ipynb`](file:///e:/Downloads/Fine-tune/llm-graph-ontology/notebooks/02_track1_pure_llm_finetuning.ipynb)
- **Evaluation Dataset**: `llm-graph-ontology/data/benchmark/`
- **Next Milestone**: **Track 2: Graph Neural Networks (GNN / GAT) for Inductive Graph Embedding & Link Prediction**.
