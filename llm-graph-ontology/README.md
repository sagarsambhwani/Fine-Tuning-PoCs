# 🧠 Phase 7: Knowledge Graph & Ontology Reasoning Benchmark

An empirical research project evaluating whether **Pure Autoregressive LLM Fine-Tuning** can perform structural graph and ontology reasoning, or if **Hybrid Multimodal Graph Neural Networks (GNN + LLM)** are fundamentally required.

---

## 🔬 The Core Research Question
> *"Can an autoregressive LLM (`Qwen2.5-1.5B-Instruct` + QLoRA) solve multi-hop ontology traversal and link prediction through linear text prompting alone, or does it hit an insurmountable inductive ceiling without explicit graph topological inductive bias?"*

---

## 📊 Comprehensive Empirical Benchmark Results

### 1. The Head-to-Head Architectural Comparison (Task C Link Prediction)

Evaluated on the official **50-sample Task C Link Prediction Benchmark**:

| Architectural Paradigm | Link Prediction Accuracy | Model Size / Footprint | Convergence / Training Time | Inductive Status |
| :--- | :---: | :---: | :---: | :--- |
| **1. Untuned Base Model** (`Qwen2.5-1.5B-Instruct`) | **24.00%** | 1.5B params | Zero-Shot Baseline | Near random guess (~20% across 5 relations) |
| **2. Track 1 Pure LLM Fine-Tuned** (QLoRA $r=16, \alpha=32$) | **36.00%** | 18.5M trainable params | ~20 min (T4 GPU) | **Inductive Ceiling Hit** (Lacks graph adjacency) |
| **3. Track 2 Hybrid GNN Encoder** (RGCN + DistMult) | **82.00%** | **~45k trainable params** | **1.94s (Standard CPU)** | **🎯 Ceiling Broken (+46.00% gain!)** |

> **Key Finding**: While pure LLM fine-tuning improves syntax and domain extraction (**100% on Task A entity extraction**), it fails on topological link prediction (**36.00%**). The Relational GNN encoder leverages neighborhood message passing to shatter this bottleneck (**82.00%**), converging >500x faster on consumer CPU hardware.

---

### 2. Multi-Stage Stress Testing & Out-Of-Distribution Benchmarks

We evaluated the models across 4 rigorous benchmark suites:

| Benchmark Suite | Dataset / Domain | Base Model | Fine-Tuned Adapter / GNN | Key Empirical Insight |
| :--- | :--- | :---: | :---: | :--- |
| **In-Domain Clean** | Synthetic Legal KG (50 samples each) | Task A: 84%<br>Task B (2-hop): 60%<br>Task C: 24% | **Task A: 100%**<br>**Task B (2-hop): 100%**<br>**Task B (3-hop): 90%**<br>Task C (GNN): **82.00%** | Specialization masters in-domain ontology extraction and topological reasoning. |
| **Adversarial Stress Test** | Syntax mutations, Distractor facts, Inversions | 46.67% F1 | **61.67% F1** | Fine-tuned LLM proved resilient to distractors (+15.00%), but suffered 80% directional inversions on passive voice and 100% remedy hallucination on dismissed claims. |
| **Public Real-World WebNLG** | 50 human RDF triples across 5 general domains | **8.00%** F1 | 2.00% F1 | Extreme legal domain specialization caused catastrophic out-of-domain forgetting on general non-legal entities. |
| **Stanford LegalBench** | 30 real-world Federal Court Complaints | **93.33%** F1 | 80.00% F1 | Base model generalized better on raw human-drafted legal text; fine-tuned adapter overfit to strict synthetic token boundaries. |

---

## 🏛️ System Architecture

### Track 1: Pure LLM Fine-Tuning (Text-Only)
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Tuning Method**: QLoRA (4-bit NF4 quantization, Double Quantization, Rank $r=16$, Alpha $\alpha=32$, LoRA Dropout $0.05$)
- **Targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (18.5M params)
- **Loss Masking**: Assistant token response masking (`DataCollatorForCompletionOnlyLM`)

### Track 2: Hybrid GNN + LLM Multimodal Architecture
- **Topological Encoder**: 2-layer Relational Graph Convolutional Network (`RGCNConv`, hidden dim 128, 5 relation types, basis-decomposition)
- **Link Predictor**: Bilinear DistMult head ($f(u, r, v) = h_u^T \operatorname{diag}(W_r) h_v$)
- **Multimodal Projector**: 2-layer MLP (`Linear(128, 512) -> GELU -> LayerNorm -> Linear(512, 1536)`) aligning GNN node representations directly to `Qwen2.5-1.5B` token embedding space $\mathbb{R}^{1536}$ for soft prefix prompt injection.

---

## 📁 Project Structure

```
llm-graph-ontology/
├── configs/
│   ├── track1_pure_llm.yaml             # QLoRA hyperparameters & training configuration
│   └── track2_gnn_llm.yaml              # RGCN architecture & MLP projector configuration
├── data/
│   ├── benchmark/                       # Empirical evaluation test suites
│   │   ├── legal_task_a_entity_extract.json
│   │   ├── legal_task_b_multihop_reason.json
│   │   ├── legal_task_c_link_pred.json
│   │   ├── legal_adversarial_benchmark.json
│   │   ├── webnlg_benchmark.json
│   │   ├── legalbench_cpu_benchmark.json
│   │   └── track2_hybrid_gnn_results.json
│   ├── processed/
│   │   └── graph_node_mapping.json      # Entity and relation ID mappings
│   └── schema/                          # Formal legal ontology schema (OWL / Turtle / JSON-LD)
├── notebooks/
│   ├── 01_ontology_data_generation.ipynb
│   ├── 02_track1_pure_llm_finetuning.ipynb
│   ├── 03_adversarial_stress_test.ipynb
│   ├── 04_webnlg_real_world_benchmark.ipynb
│   ├── 05_legalbench_cpu_evaluation.ipynb
│   └── 06_track2_gnn_llm_hybrid.ipynb  # End-to-end Track 2 Colab CPU notebook
├── reports/
│   ├── adversarial_stress_test_report.md
│   ├── webnlg_evaluation_report.md
│   ├── legalbench_cpu_evaluation_report.md
│   └── track2_hybrid_gnn_report.md     # Official Track 1 vs Track 2 empirical report
└── src/
    ├── data/
    │   ├── graph_dataset.py             # PyG Data graph constructor & negative sampler
    │   └── generate_*.py                # Synthetic & benchmark dataset generators
    ├── models/
    │   ├── gnn_encoder.py               # 2-layer RGCN encoder & DistMult link predictor
    │   └── projector.py                 # Multimodal MLP Projector (R^128 -> R^1536)
    └── training/
        └── train_track1_pure_llm.py     # QLoRA SFT training script
```

---

## 🚀 Reproduction & Quick Start

### Running the Hybrid GNN Locally or in Google Colab (CPU)
```bash
# 1. Activate your virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Build the PyG graph dataset
python llm-graph-ontology/src/data/graph_dataset.py

# 3. Train the Relational GNN (completes in ~2 seconds on CPU)
python llm-graph-ontology/src/models/gnn_encoder.py

# 4. Verify the Multimodal Projector (R^128 -> R^1536)
python llm-graph-ontology/src/models/projector.py
```
Or open [`notebooks/06_track2_gnn_llm_hybrid.ipynb`](file:///e:/Downloads/Fine-tune/llm-graph-ontology/notebooks/06_track2_gnn_llm_hybrid.ipynb) directly in Google Colab and execute with standard CPU runtime.
