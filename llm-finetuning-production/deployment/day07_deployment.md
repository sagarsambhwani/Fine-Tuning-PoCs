# Day 7: Deployment & Smoke Testing Log

## Overview
This document records the verification and smoke testing procedures for the containerized FastAPI serving system.

## Smoke Testing Checklist

- [ ] **1. Unit & Integration Tests**:
  ```bash
  pytest tests/ -v
  ```
  Expected: All tests for data schemas, inference regex/parsers, and FastAPI test client pass.

- [ ] **2. Local API Launch**:
  ```bash
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  ```
  Expected: Fast startup, model weights loaded once into memory during lifespan handler.

- [ ] **3. Endpoint Verification (Health)**:
  ```bash
  curl -s http://localhost:8000/health
  ```
  Expected output:
  ```json
  {"status": "healthy", "model": "qwen-1.5b-order-extractor", "device": "cuda"}
  ```

- [ ] **4. Endpoint Verification (Inference)**:
  ```bash
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "John ordered 3 laptops for $2400 and wants delivery on Friday."}'
  ```
  Expected output:
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
    "latency_ms": 128.45,
    "model_name": "qwen-1.5b-order-extractor"
  }
  ```

- [ ] **5. Docker Build & Run**:
  ```bash
  docker build -t llm-json-extractor:latest .
  docker run -d -p 8000:8000 --name test-api llm-json-extractor:latest
  curl -s http://localhost:8000/health
  docker stop test-api && docker rm test-api
  ```

## Smoke Test Results Log
*Status: READY FOR EXECUTION (Marked NOT RUN until learner executes on target hardware)*
- Test Date: `NOT RUN`
- Target Environment: `NOT RUN` (e.g. Local CPU / Colab / RunPod T4)
- Average /predict Latency: `NOT RUN` ms
- Container Build Time: `NOT RUN`
- Pass/Fail Status: `NOT RUN`
