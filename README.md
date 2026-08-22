# Fine-Tuning Proofs-of-Concept (PoCs) & Production LLM Engineering

A comprehensive repository demonstrating hands-on fine-tuning methodologies, from foundational transformer classification and custom tokenization up to parameter-efficient fine-tuning (LoRA / QLoRA), instruction tuning (SFT), structured JSON extraction, and containerized FastAPI production deployment.

---

## 🗺️ Repository Structure & Projects

```
Fine-tune/
│
├── 🚀 llm-finetuning-production/        # ⭐ 7-Day LLM Fine-Tuning + Production Deployment
│   ├── notebooks/                      # 7 Colab notebooks (Day 1 LoRA to Day 7 Docker)
│   ├── src/                            # Modular Python package (data, training, eval, inference)
│   ├── api/                            # FastAPI serving application (Lifespan model loading)
│   ├── configs/                        # training_config.yaml (Qwen-2.5-1.5B, QLoRA, LoRA)
│   ├── data/                           # Processed Train (1200), Val (150), Test (150) JSONL
│   ├── tests/                          # Automated pytest suite (12 passing tests)
│   ├── Dockerfile                      # Production container with healthcheck
│   └── README.md                       # Full 7-Day Portfolio & Interview Reference
│
├── 📌 distilbert-base-uncased/          # PoC 1.1: Classical ML vs. Transformer Fine-Tuning
│   ├── train.py
│   ├── inference.py
│   └── README.md
│
├── 📌 multilabel-classification/        # PoC 1.2: Multi-Label Classification
│   ├── train.py                        # BCEWithLogitsLoss, Sigmoid multi-label head
│   └── README.md
│
├── 📌 tokenization-custom-vocab/        # PoC 1.3: Tokenization & Custom Vocabulary
│   ├── custom_tokenizer.py             # BPE / WordPiece, adding custom tokens
│   └── README.md
│
└── 📚 docs/                             # Deep-dive theoretical guides & roadmaps
    ├── prerequisite/                   # 🧱 Progressive Prerequisite Foundations & Pedagogical Blueprint
    ├── interview_guide/                # 🎯 Comprehensive Fine-Tuning A to Z Interview Guide (Modules 1-7)
    ├── fine_tuning_roadmap.md          # Multi-phase fine-tuning progression
    ├── senior_ai_engineer_handbook.md  # 7-Pillar Senior AI Engineer production guide
    ├── adapter_deployment_strategies.md# Full fusion, Multi-LoRA, AWQ, GGUF, FastAPI
    ├── day01_lora_explained.md         # LoRA & PEFT mathematical foundations
    ├── day02_qlora_explained.md        # QLoRA 4-bit NF4 quantization deep dive
    └── day03_sft_explained.md          # ChatML & Assistant Loss Masking mechanics
```

---

## 🌟 Featured Project: 7-Day LLM Fine-Tuning & Serving Sprint

Located in [**`llm-finetuning-production/`**](file:///e:/Downloads/Fine-tune/llm-finetuning-production/README.md):

> **"I took an open-source 1.5B parameter instruction model (`Qwen/Qwen2.5-1.5B-Instruct`), prepared a domain-specific dataset with strict anti-leakage guarantees, fine-tuned it using QLoRA and SFT on a Google Colab T4 GPU, evaluated it against an empirical zero-shot base baseline on held-out test data, merged and optimized the adapter weights, and packaged the model behind a FastAPI service inside a containerized Docker GPU environment."**

### 7-Day Progression Overview:
- **Day 1: LoRA & PEFT Fundamentals** — Low-rank matrix adaptation $W' = W_0 + (\alpha/r)BA$, parameter efficiency calculation (1.18% trainable parameters), adapter saving/loading.
- **Day 2: QLoRA & 4-bit Quantization** — NormalFloat4 (NF4), double quantization, paged optimizers, VRAM profiling.
- **Day 3: Supervised Fine-Tuning (SFT)** — Chat templates (`<|im_start|>`), assistant response loss masking (`labels = -100`), TRL SFTTrainer.
- **Day 4: Domain-Specific Fine-Tuning** — Natural language to structured JSON order extraction, 1,200 training samples, zero-leakage test set.
- **Day 5: Empirical LLM Evaluation** — Zero-shot Base vs Fine-Tuned benchmark on held-out test split (JSON validity, exact match, field-level accuracy).
- **Day 6: LoRA Adapter Merge & Quantized Inference** — `merge_and_unload()` export, standalone Safetensors weights, latency & throughput benchmarking.
- **Day 7: FastAPI + Docker Deployment** — Production REST API (`/health`, `/predict`), Lifespan model loading, Pydantic validation, Docker containerization.

---

## 🛠️ Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Run unit tests
pytest llm-finetuning-production/tests -v

# 3. Explore Day 1 LoRA Notebook
jupyter notebook llm-finetuning-production/notebooks/day01_lora.ipynb
```
