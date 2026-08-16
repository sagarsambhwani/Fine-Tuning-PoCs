# Production-Style Deployment Guide

This guide details how to serve the fine-tuned structured JSON extraction LLM as a containerized API across various cloud and local GPU environments.

## Deployment Target Comparison

| Environment | Use Case | GPU / Specs | Estimated Cost | Recommended For |
|---|---|---|---|---|
| **Local / Workstation** | Testing & Smoke Tests | CPU or local NVIDIA GPU | $0 | Rapid verification |
| **RunPod** | Serverless / Pod | RTX 4090 / A4000 (16-24 GB) | ~$0.20–$0.40/hr | Production testing |
| **Modal** | Serverless Python GPU | T4 / A10G / L4 | Pay per second | Scalable microservices |
| **Hugging Face Spaces** | Portfolio Demo UI | T4 GPU ($0.60/hr) or ZeroGPU | Low / Free Tier | Public portfolio demo |
| **AWS EC2 (g4dn.xlarge)** | Dedicated Enterprise VM | T4 16GB VRAM | ~$0.52/hr | Enterprise deployment |

---

## 1. Local Testing & Verification

Run the FastAPI service directly:
```bash
# Start API locally with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test the `/health` endpoint:
```bash
curl -X GET http://localhost:8000/health
```

Test the `/predict` endpoint:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Sarah bought 5 monitors for $1500 and requested delivery on Monday."}'
```

---

## 2. Docker Container Deployment

### Build the Image
```bash
docker build -t llm-json-extractor:latest .
```

### Run the Container (with NVIDIA GPU support)
```bash
docker run --gpus all -d -p 8000:8000 --name json-extractor-api llm-json-extractor:latest
```

### Run the Container (CPU Fallback Mode)
```bash
docker run -d -p 8000:8000 -e DEVICE=cpu --name json-extractor-api llm-json-extractor:latest
```

---

## 3. Modal Serverless Deployment (Optional)

Deploying to Modal requires minimal configuration:
```python
import modal

app = modal.App("llm-order-extractor")
image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "accelerate", "fastapi", "pydantic", "uvicorn"
)

@app.function(image=image, gpu="T4")
@modal.asgi_app()
def fastapi_app():
    from api.main import app as web_app
    return web_app
```

Deploy with:
```bash
modal deploy deployment/modal_app.py
```
