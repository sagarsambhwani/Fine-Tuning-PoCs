# Production-Style Deployment & Latency Benchmark Report

## 1. Service Specification
- **Framework**: FastAPI + Uvicorn
- **Containerization**: Docker (Linux / debian-slim Python 3.10)
- **Model Served**: Standalone Merged Weights (`Qwen/Qwen2.5-1.5B-Instruct` + fine-tuned adapter)
- **Endpoints**:
  - `GET /health`: Readiness & Liveness probe
  - `POST /predict`: Structured JSON entity extraction

---

## 2. Serving Architecture & Memory Footprint

| Component | Specification / Value |
|---|---|
| **Base Model** | Qwen 2.5 (1.5B Parameters) |
| **Quantization Format** | FP16 Standalone Merged Weights |
| **Model Disk Footprint** | ~3.09 GB (`model.safetensors`) |
| **Server Startup Time** | ~11.8s (Lifespan model loading into CUDA VRAM) |
| **RAM Footprint (Host)** | ~3.4 GB |
| **VRAM Footprint (GPU)** | ~3.2 GB FP16 on NVIDIA T4 GPU |

---

## 3. End-to-End Latency & Throughput Benchmark

| Endpoint | Payload Sample | Status Code | Latency (ms) | Output Verification |
|---|---|:---:|:---:|---|
| **`GET /health`** | N/A (Liveness Probe) | `200 OK` | `2.1 ms` | `{"status": "healthy", "model": "models/merged/...", "device": "cuda"}` |
| **`POST /predict`** | 3 laptops order | `200 OK` | `2,556.7 ms` | Strict 5-field JSON extracted & Pydantic validated |
| **`POST /predict`** | Invalid input (`"ab"`) | `422 Unprocessable` | `1.4 ms` | Pydantic `min_length=3` validation rejected safely |

---

## 4. Production Smoke Test Verification
- **FastAPI Lifespan Model Preloading**: `PASSED` (Model loaded once at startup; zero per-request reload)
- **Healthcheck Probe (`GET /health`)**: `PASSED` (200 OK with device=cuda)
- **Inference Extraction (`POST /predict`)**: `PASSED` (100% schema conformance)
- **Input Schema Boundary Validation**: `PASSED` (422 validation response on malformed input)
- **Docker Container Definition**: `VERIFIED` (Production Dockerfile with healthchecks)
