# 🚀 7-Day LLM Fine-Tuning + Production-Style Deployment Portfolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/docs/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%26%20QLoRA-green.svg)](https://github.com/huggingface/peft)
[![TRL](https://img.shields.io/badge/TRL-SFTTrainer-orange.svg)](https://github.com/huggingface/trl)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An end-to-end, production-oriented LLM engineering project that guides you from **LoRA/PEFT fundamentals** to a fine-tuned, evaluated, merged, quantized, and containerized API serving system.

---

## 📑 Portfolio Narrative

> **“I took an open-source 1.5B parameter instruction model (`Qwen/Qwen2.5-1.5B-Instruct`), prepared a domain-specific dataset with strict anti-leakage guarantees, fine-tuned it using QLoRA and SFT on a Google Colab T4 GPU, evaluated it against an empirical zero-shot base baseline on held-out test data, merged and optimized the adapter weights, and packaged the model behind a FastAPI service inside a containerized Docker GPU environment.”**

---

## 🏗️ End-to-End Pipeline Architecture

```
                                  [ Unstructured Natural Language Input ]
                                                    ↓
                                      [ Data Curation & Anti-Leakage ]
                                      (1,200 Train | 150 Val | 150 Test)
                                                    ↓
                                  ┌─────────────────────────────────────┐
                                  │   Zero-Shot Base Model Benchmark   │
                                  │ (JSON Validity, Exact Match, Field) │
                                  └─────────────────────────────────────┘
                                                    ↓
                                        [ 4-Bit NF4 Quantization ]
                                        (BitsAndBytes + Double Quant)
                                                    ↓
                                    [ LoRA Low-Rank Adaptation ]
                                    (W' = W_0 + (α/r)·B·A on all linear)
                                                    ↓
                                    [ Supervised Fine-Tuning (SFT) ]
                                    (Assistant Loss Masking: labels=-100)
                                                    ↓
                                  ┌─────────────────────────────────────┐
                                  │   Fine-Tuned Empirical Evaluation   │
                                  │   (Base vs Fine-Tuned Test Split)   │
                                  └─────────────────────────────────────┘
                                                    ↓
                                        [ merge_and_unload() ]
                                   (Standalone Safetensors Model Export)
                                                    ↓
                                  ┌─────────────────────────────────────┐
                                  │     FastAPI Serving Architecture    │
                                  │ (Lifespan Loading + Pydantic Schema)│
                                  └─────────────────────────────────────┘
                                                    ↓
                                        [ Docker Containerization ]
                                         (Healthcheck + GPU / CPU)
                                                    ↓
                                      [ Validated Structured JSON ]
```

---

## 🗓️ 7-Day Hands-On Progression

| Day | Topic | Key Deliverables & Artifacts | Primary Notebook / Script |
|---|---|---|---|
| **Day 1** | **LoRA & PEFT Fundamentals** | Math formulation $W' = W + (\alpha/r)BA$, parameter-efficiency calculation, adapter saving & reloading. | [`notebooks/day01_lora.ipynb`](notebooks/day01_lora.ipynb) |
| **Day 2** | **QLoRA & 4-bit Quantization** | NormalFloat4 (NF4), Double Quantization, memory profiling report vs standard LoRA. | [`notebooks/day02_qlora.ipynb`](notebooks/day02_qlora.ipynb), [`reports/day02_qlora_report.md`](reports/day02_qlora_report.md) |
| **Day 3** | **SFT & Chat Templates** | Chat templates (`<|im_start|>`), assistant response loss masking (`labels=-100`), TRL SFTTrainer. | [`notebooks/day03_sft.ipynb`](notebooks/day03_sft.ipynb) |
| **Day 4** | **Domain-Specific Fine-Tuning** | Unstructured text $\to$ structured JSON dataset, strict train/val/test splits, QLoRA SFT training run. | [`notebooks/day04_domain_finetuning.ipynb`](notebooks/day04_domain_finetuning.ipynb), [`reports/training_report.md`](reports/training_report.md) |
| **Day 5** | **Empirical Evaluation** | Zero-shot Base vs Fine-Tuned benchmark on held-out test split, JSON validity, exact match, field F1. | [`notebooks/day05_evaluation.ipynb`](notebooks/day05_evaluation.ipynb), [`reports/evaluation_report.md`](reports/evaluation_report.md) |
| **Day 6** | **Merge & Quantize Inference** | `merge_and_unload()`, standalone weights export, latency (ms) & throughput (tokens/sec) benchmarking. | [`notebooks/day06_quantization.ipynb`](notebooks/day06_quantization.ipynb) |
| **Day 7** | **FastAPI + Docker Deployment** | Production-style FastAPI API (`/health`, `/predict`), Dockerfile, deployment guide, end-to-end smoke test. | [`notebooks/day07_deployment.ipynb`](notebooks/day07_deployment.ipynb), [`deployment/day07_deployment.md`](deployment/day07_deployment.md) |

---

## 🖥️ Hardware & Model Strategy

- **Baseline Hardware Target**: Google Colab (NVIDIA T4 16GB VRAM) / Linux / Local GPU
- **Primary Model**: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache 2.0 open license, native chat template, state-of-the-art structured JSON capability).
- **Fallback Model**: `Qwen/Qwen2.5-0.5B-Instruct` (for low-resource environments).
- **Memory Footprint**:
  - FP16 Base Model: ~3.1 GB
  - 4-Bit NF4 Quantized Base Model: **~1.1 GB**
  - Trainable Adapter Parameters: **~18.4M params (~1.19% of total)**
  - Peak Training VRAM with QLoRA: **~3.6 GB** (Fits effortlessly on 16GB T4 with headroom for batch size 2-4 and gradient accumulation 4).

---

## 📂 Repository Structure

```
llm-finetuning-production/
├── README.md                          # Master Portfolio & Project Documentation
├── requirements.txt                   # Pinned dependency manifest
├── .gitignore                         # Excludes checkpoints, large weights, caches
├── Dockerfile                         # Production-style container definition
├── docker-compose.yml                 # Local & VM multi-service orchestration
│
├── configs/
│   └── training_config.yaml           # Centralized training, LoRA & serving configuration
│
├── data/
│   ├── raw/                           # Raw generation traces
│   ├── processed/                     # Train (1,200), Val (150), Test (150) JSONL splits
│   └── README.md                      # Data schemas & anti-leakage documentation
│
├── notebooks/
│   ├── day01_lora.ipynb               # Day 1: LoRA Math & Parameter Efficiency
│   ├── day02_qlora.ipynb              # Day 2: QLoRA, NF4 & Memory Benchmarking
│   ├── day03_sft.ipynb                # Day 3: Instruction Tuning & Loss Masking
│   ├── day04_domain_finetuning.ipynb  # Day 4: Domain-Specific QLoRA + SFT Pipeline
│   ├── day05_evaluation.ipynb         # Day 5: Base vs Fine-Tuned Benchmark
│   ├── day06_quantization.ipynb       # Day 6: Adapter Merging & Inference Benchmarks
│   └── day07_deployment.ipynb         # Day 7: FastAPI & Docker Smoke Testing
│
├── src/
│   ├── data/
│   │   ├── dataset_generator.py       # Deterministic synthetic order generator
│   │   ├── formatter.py               # Tokenizer chat template formatter
│   │   └── validator.py               # Schema conformance & JSON parsing
│   ├── training/
│   │   ├── lora_config.py             # PEFT LoRA configuration & parameter inspection
│   │   ├── qlora_config.py            # BitsAndBytes 4-bit NF4 configuration
│   │   └── trainer.py                 # SFTTrainer runner with metrics logging
│   ├── evaluation/
│   │   ├── metrics.py                 # JSON validity, exact match, field accuracy metrics
│   │   └── benchmark.py               # Test split evaluation runner
│   ├── inference/
│   │   ├── engine.py                  # Low-latency structured inference engine
│   │   └── merger.py                  # merge_and_unload() standalone weight exporter
│   └── utils/
│       ├── env_detector.py            # GPU VRAM & library environment auditor
│       └── logger.py                  # Structured logging utility
│
├── api/
│   ├── main.py                        # FastAPI application with Lifespan loader
│   ├── schemas.py                     # Pydantic request & response models
│   └── inference.py                   # API model manager & inference singleton
│
├── tests/
│   ├── test_data.py                   # Data generation & schema validation tests
│   ├── test_inference.py              # Metric calculation & JSON parser tests
│   └── test_api.py                    # FastAPI endpoint tests using TestClient
│
├── deployment/
│   ├── README.md                      # RunPod, Modal, EC2 & HF Spaces guide
│   └── day07_deployment.md            # Smoke testing & verification log
│
├── models/
│   └── README.md                      # Guide to adapter and merged model weights
│
└── reports/
    ├── day02_qlora_report.md          # LoRA vs QLoRA memory comparison
    ├── training_report.md             # SFT training dynamics & loss log
    ├── evaluation_report.md           # Base vs Fine-Tuned quantitative table
    └── deployment_report.md           # Serving latency & throughput benchmarks
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/llm-finetuning-production.git
cd llm-finetuning-production

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 3. Generate Domain Dataset
```bash
python src/data/dataset_generator.py --train 1200 --val 150 --test 150 --seed 42
```

### 4. Run SFT Fine-Tuning (or open `notebooks/day04_domain_finetuning.ipynb` in Colab)
```bash
python src/training/trainer.py
```

### 5. Merge Adapter Weights
```bash
python src/inference/merger.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir models/adapters/qwen-1.5b-order-extractor \
  --output_dir models/merged/qwen-1.5b-order-extractor
```

### 6. Launch FastAPI Server Locally
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Usage & Example Payloads

### Healthcheck Probe
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model": "models/merged/qwen-1.5b-order-extractor",
  "device": "cuda"
}
```

### Structured Extraction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "John ordered 3 laptops for $2400 and wants delivery on Friday."}'
```
**Response:**
```json
{
  "data": {
    "customer": "John",
    "quantity": 3,
    "product": "laptops",
    "amount": 2400.0,
    "delivery_day": "Friday"
  },
  "raw_response": "{\"customer\": \"John\", \"quantity\": 3, \"product\": \"laptops\", \"amount\": 2400.0, \"delivery_day\": \"Friday\"}",
  "is_valid_json": true,
  "error": null,
  "latency_ms": 112.4,
  "model_name": "models/merged/qwen-1.5b-order-extractor"
}
```

---

## 🐳 Docker Deployment & Smoke Testing

### Build Container
```bash
docker build -t llm-json-extractor:latest .
```

### Run Container (NVIDIA GPU)
```bash
docker run --gpus all -d -p 8000:8000 --name json-extractor-api llm-json-extractor:latest
```

### Run Container (CPU Mode)
```bash
docker run -d -p 8000:8000 -e DEVICE=cpu --name json-extractor-api llm-json-extractor:latest
```

---

## 🎯 Resume Readiness Evaluation Checklist

Mark a skill as verified **only** after personally executing the corresponding notebook/module and inspecting the empirical outputs:

- [ ] **LoRA**: Understand low-rank matrix decomposition $W' = W + (\alpha/r)BA$, freezing base weights, and parameter efficiency.
- [ ] **QLoRA**: Understand 4-bit NormalFloat (NF4), double quantization, paged optimizers, and VRAM reduction.
- [ ] **PEFT**: Configured `LoraConfig`, `get_peft_model`, saved and reloaded adapters.
- [ ] **SFT**: Formatted instruction datasets with chat templates and applied assistant loss masking (`labels=-100`).
- [ ] **Dataset Preparation**: Generated stratified datasets with anti-leakage verification.
- [ ] **LLM Evaluation**: Benchmarked Base vs. Fine-Tuned models on held-out test splits (JSON validity, exact match, field accuracy).
- [ ] **Adapter Merging**: Fused adapter weights using `merge_and_unload()` into standalone Safetensors.
- [ ] **FastAPI Serving**: Built REST endpoints (`/health`, `/predict`) with Lifespan weight preloading.
- [ ] **Docker Containerization**: Wrote production Dockerfile with healthchecks and smoke tested end-to-end.
