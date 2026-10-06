# 📋 Step-by-Step Execution Plan: Pure LLM vs. Hybrid GNN+LLM Benchmark

This plan breaks down the entire project into manageable, self-contained steps. Each step has clear inputs, code deliverables, verification criteria, and expected outputs.

---

## 🗺️ Master Progression Overview

```
[Phase 1: Foundation & Data] ──► [Phase 2: Track 1 (Pure LLM)] ──► [Phase 3: Track 2 (GNN + LLM)] ──► [Phase 4: Head-to-Head Benchmark]
```

---

## 📌 Phase 1: Ontology Schema & Graph Dataset Construction

### Step 1.1: Define Formal Domain Ontology Schema
* **What we do**: Create a formal ontology definition (`ontology_schema.json`) with strict entity classes, allowed relation predicates, and domain/range type constraints (e.g. Biomedical or Supply Chain KG).
* **Deliverable**: `llm-graph-ontology/data/ontology_schema.json`
* **Verification**: Validate schema structure with Pydantic / JSON schema validator.

### Step 1.2: Synthetic Knowledge Graph Generator & Splitter
* **What we do**: Write a deterministic generator that constructs a connected Knowledge Graph and produces three benchmark tasks:
  1. **Task A (IE)**: Text paragraph $\rightarrow$ List of formal triples `(Subject, Predicate, Object)`.
  2. **Task B (Multi-Hop)**: Graph path context + Query $\rightarrow$ Deductive conclusion.
  3. **Task C (Link Prediction)**: Subgraph with missing edge $\rightarrow$ Target relation.
* **Deliverable**: `llm-graph-ontology/src/data/graph_dataset_generator.py`
* **Verification**: Generate `train.jsonl` (1,200), `val.jsonl` (150), `test.jsonl` (150) with automated 0% data leakage checks.

### Step 1.3: Graph Verbalization & Serialization Formatter
* **What we do**: Implement converters that format graph structures into standard text representations:
  * Turtle / RDF syntax
  * Triple list format
  * ChatML `<|im_start|>` conversational prompts
* **Deliverable**: `llm-graph-ontology/src/data/graph_serializer.py`
* **Verification**: Unit tests verifying bidirectional conversion (Graph $\leftrightarrow$ Text).

---

## 📌 Phase 2: Track 1 — Pure LLM Fine-Tuning (Language-Native)

### Step 2.1: Track 1 Configuration & Baseline Zero-Shot Evaluation
* **What we do**: Configure `configs/track1_pure_llm.yaml`. Run untuned `Qwen2.5-1.5B-Instruct` on the held-out test split across Tasks A, B, and C to establish our baseline scores.
* **Deliverable**: `notebooks/01_track1_baseline.ipynb`, `reports/track1_baseline_metrics.json`
* **Verification**: Baseline metrics computed and saved (typically ~30–55% accuracy on strict ontology triples).

### Step 2.2: Track 1 QLoRA + SFT Training Pipeline
* **What we do**: Fine-tune `Qwen2.5-1.5B-Instruct` with 4-bit NF4 quantization and LoRA targeting all linear projection layers on the serialized graph dataset.
* **Deliverable**: `src/training/train_track1.py`, `notebooks/02_track1_training.ipynb`
* **Verification**: Training completes under 16GB VRAM, loss converges, adapter saved to `models/adapters/track1_pure_llm`.

### Step 2.3: Track 1 Evaluation & Metric Calculation
* **What we do**: Evaluate fine-tuned Track 1 model on the test split:
  * Macro F1 score on Triplet Information Extraction.
  * Multi-hop deduction accuracy.
  * Link prediction Hits@1 and MRR.
* **Deliverable**: `reports/track1_finetuned_metrics.json`
* **Verification**: Quantitative before-and-after comparison logged.

---

## 📌 Phase 3: Track 2 — Hybrid GNN + LLM Architecture Fusion

### Step 3.1: GNN Encoder Module (`torch_geometric`)
* **What we do**: Build a lightweight Graph Neural Network (2-layer GCN/GAT) that takes node feature vectors $X \in \mathbb{R}^{N \times d_{\text{in}}}$ and adjacency edge index $E \in \mathbb{R}^{2 \times |E|}$ to compute topological node embeddings $Z \in \mathbb{R}^{N \times 128}$.
* **Deliverable**: `src/models/gnn_encoder.py`
* **Verification**: Unit tests on synthetic PyG `Data` objects confirming shape transformations.

### Step 3.2: Multimodal Graph-to-Language Projector Layer
* **What we do**: Implement a 2-layer MLP projection module mapping continuous graph embeddings into LLM token space:
  $$\mathbf{W}_{\text{proj}}: \mathbb{R}^{128} \rightarrow \mathbb{R}^{1536}$$
* **Deliverable**: `src/models/graph_projector.py`
* **Verification**: Verify projected graph tensors match Qwen token embedding dimensions and can be concatenated to input token embeddings.

### Step 3.3: Hybrid Forward Pass & Training Pipeline
* **What we do**: Implement the joint training loop:
  1. GNN encodes the subgraph.
  2. Projector maps node representations to "virtual graph tokens".
  3. Virtual graph tokens are prepended to prompt text token embeddings.
  4. Qwen-1.5B computes autoregressive loss and backpropagates gradients into the Projector and LoRA weights.
* **Deliverable**: `src/training/train_track2.py`, `notebooks/03_track2_hybrid_training.ipynb`
* **Verification**: Hybrid forward/backward pass executes without OOM on 16GB VRAM; checkpoints saved to `models/adapters/track2_hybrid`.

### Step 3.4: Track 2 Evaluation on Identical Benchmark
* **What we do**: Run the hybrid model on the exact same test dataset across Tasks A, B, and C.
* **Deliverable**: `reports/track2_hybrid_metrics.json`
* **Verification**: Compute Task A F1, Task B multi-hop accuracy, and Task C link prediction.

---

## 📌 Phase 4: Head-to-Head Comparative Analysis

### Step 4.1: Side-by-Side Benchmark & Latency Profiling
* **What we do**: Measure both systems on:
  1. **Task Accuracy**: F1 on Triplet IE, Multi-hop accuracy, Link prediction MRR.
  2. **Memory Footprint**: Peak training and inference VRAM.
  3. **Inference Latency**: Milliseconds per query and tokens/second.
  4. **Context Window Scalability**: How performance changes as graph size scales (10 nodes vs 50 nodes vs 100 nodes).
* **Deliverable**: `notebooks/04_comparative_benchmark.ipynb`, `src/evaluation/comparative_benchmark.py`
* **Verification**: Complete comparison table generated directly from empirical runs.

### Step 4.2: Master Report & Architectural Takeaways
* **What we do**: Write the final engineering report detailing:
  * Where Pure LLM won (e.g. semantic language nuances, entity synonyms).
  * Where GNN+LLM won (e.g. complex multi-hop cycles, graph connectivity).
  * Production recommendations for enterprise Knowledge Graphs.
* **Deliverable**: `reports/comparative_benchmark_report.md`, root `README.md`
* **Verification**: Fully reproducible, interview-ready documentation with no fabricated metrics.

---

## 🏁 Summary Checklist

| Step | Task Name | Status |
| :--- | :--- | :--- |
| **1.1** | Define Formal Domain Ontology Schema | ⚪ Up Next |
| **1.2** | Synthetic Knowledge Graph Dataset Generator | ⚪ Pending |
| **1.3** | Graph Verbalizer & Serializer | ⚪ Pending |
| **2.1** | Track 1 Configuration & Zero-Shot Baseline | ⚪ Pending |
| **2.2** | Track 1 QLoRA + SFT Training | ⚪ Pending |
| **2.3** | Track 1 Evaluation Metrics | ⚪ Pending |
| **3.1** | GNN Topological Encoder (PyG) | ⚪ Pending |
| **3.2** | Multimodal Graph Projector Layer | ⚪ Pending |
| **3.3** | Hybrid Joint Training Loop | ⚪ Pending |
| **3.4** | Track 2 Evaluation Metrics | ⚪ Pending |
| **4.1** | Head-to-Head Benchmark & Profiling | ⚪ Pending |
| **4.2** | Master Report & Portfolio Documentation | ⚪ Pending |
