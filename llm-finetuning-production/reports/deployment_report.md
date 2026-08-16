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
| **Quantization Format** | FP16 Merged / 4-bit NF4 |
| **Model Disk Footprint** | ~3.1 GB (FP16) / ~1.1 GB (4-bit) |
| **Server Startup Time** | `NOT RUN` (est. 4.2s) |
| **RAM Footprint (Host)** | `NOT RUN` (est. 3.8 GB) |
| **VRAM Footprint (GPU)** | `NOT RUN` (est. 3.2 GB FP16 / ~1.6 GB 4-bit) |

---

## 3. End-to-End Latency & Throughput Benchmark

| Concurrency (Workers) | Median Latency (p50) | 95th Percentile (p95) | 99th Percentile (p99) | Tokens / Sec |
|---|---|---|---|---|
| **1 Worker (Sequential)** | `NOT RUN` ms | `NOT RUN` ms | `NOT RUN` ms | `NOT RUN` |
| **4 Workers (Concurrent)** | `NOT RUN` ms | `NOT RUN` ms | `NOT RUN` ms | `NOT RUN` |
| **8 Workers (Stress Test)** | `NOT RUN` ms | `NOT RUN` ms | `NOT RUN` ms | `NOT RUN` |

---

## 4. Production Smoke Test Verification
- Container Build: `NOT RUN`
- Healthcheck Passed: `NOT RUN`
- Valid JSON returned on Sample Request: `NOT RUN`
