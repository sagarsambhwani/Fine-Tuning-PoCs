# Comprehensive Guide: Production Deployment Strategies for Trained LoRA & QLoRA Adapters

Deploying fine-tuned Parameter-Efficient Fine-Tuning (PEFT) models in production involves architectural trade-offs between **throughput, latency, hardware costs, multi-tenancy, and operational complexity**.

This document outlines the **5 primary production deployment patterns** for LoRA/QLoRA adapters, when to use each, step-by-step implementation recipes, and a production decision matrix.

---

## 🗺️ Architectural Overview

```
                               TRAINED LoRA ADAPTERS (70 MB)
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
    STRATEGY 1                          STRATEGY 2                          STRATEGY 3
 [Full Fusion / Merge]              [Dynamic Multi-LoRA]                [Edge / Quantized]
 • Permanent base weight fusion     • 1 Base Model + N dynamic adapters • GGUF / AWQ / GPTQ
 • Zero added latency               • Multi-tenant shared GPU serving   • CPU / Edge / Ollama / llama.cpp
 • vLLM, TGI, Triton, TensorRT-LLM  • vLLM, S-LoRA, Predibase LoRAX     • Local on-premise execution
         │                                   │                                   │
         └───────────────────────────────────┼───────────────────────────────────┘
                                             ▼
                                    STRATEGY 4 & 5
                        [Production Serving & Infrastructure]
                        • FastAPI Containerized Microservice (Docker)
                        • Serverless GPU Scale-to-Zero (vLLM / Modal / Triton)
```

---

## 1. Strategy 1: Permanent Weight Fusion (`merge_and_unload`)

### Concept
Permanently compute the low-rank delta matrix $\Delta W = \frac{\alpha}{r}(B \cdot A)$ and add it directly into the base weights $W_0$:
$$W_{\text{merged}} = W_0 + \frac{\alpha}{r}(B \cdot A)$$

The resulting checkpoint is saved as a standard, standalone Hugging Face `Safetensors` model with **zero PEFT dependency**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 WEIGHT FUSION WORKFLOW                                   │
│                                                                                          │
│  Base Model (FP16/BF16)  ──┐                                                             │
│                            ├─►  merge_and_unload()  ──►  Fused Standalone Safetensors     │
│  LoRA Adapter (16-bit)   ──┘                             (Zero inference overhead!)      │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Python Implementation
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "models/adapters/qwen-1.5b-order-extractor"
output_dir = "models/merged/qwen-1.5b-order-extractor-fused"

# 1. Load base model in unquantized 16-bit precision
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu",  # Merge on CPU or GPU
    trust_remote_code=False
)

# 2. Attach trained adapter
model = PeftModel.from_pretrained(base_model, adapter_dir)

# 3. Fuse weights mathematically
fused_model = model.merge_and_unload()

# 4. Save standalone checkpoint & tokenizer
fused_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(output_dir)
print(f"✅ Merged standalone weights saved to: {output_dir}")
```

> [!WARNING]
> **QLoRA Warning**: You **cannot** directly merge a 16-bit LoRA adapter into 4-bit `bitsandbytes` quantized base weights without dequantizing first. Always load the base model in `float16` or `bfloat16` on CPU/GPU before running `merge_and_unload()`.

### Pros & Cons
* ✅ **Zero Latency Overhead**: Identical inference speed to native base models.
* ✅ **Universal Compatibility**: Compatible with all high-performance engines (vLLM, TensorRT-LLM, TGI, ONNX, llama.cpp).
* ❌ **VRAM Multiplicity**: If you have 10 domain tasks, you must run 10 separate 3–14 GB model instances.

---

## 2. Strategy 2: Dynamic Multi-Tenant Adapter Swapping (Multi-LoRA)

### Concept
Keep a **single copy of the base model** in GPU VRAM and dynamically swap or apply lightweight adapters (50–100 MB) on the fly based on the user's request.

```
Incoming Requests:
  Req 1 (Order Extractor) ──┐
  Req 2 (Legal Classifier)  ├─►  [ vLLM Engine + Base Qwen-7B (VRAM) ]  ──►  Batched GPU Execution
  Req 3 (Medical Entity)   ──┘         ▲         ▲          ▲
                                       │         │          │
                                   [Adapter 1] [Adapter 2] [Adapter 3] (RAM / Disk Cache)
```

### High-Throughput Engines Supporting Multi-LoRA

#### A. vLLM Multi-LoRA Serving
vLLM supports heterogeneous batching where different requests in the exact same batch can use different LoRA adapters:

```bash
# Launch vLLM server with Multi-LoRA enabled
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora \
    --max-loras 8 \
    --max-lora-rank 32 \
    --lora-modules \
        order_extractor=/models/adapters/qwen-1.5b-order-extractor \
        customer_support=/models/adapters/qwen-1.5b-support-bot
```

**Client Request via OpenAI-compatible API**:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Dynamically target the order extractor adapter
response = client.chat.completions.create(
    model="order_extractor",  # Specify the adapter name!
    messages=[
        {"role": "system", "content": "Extract order info into JSON."},
        {"role": "user", "content": "Alice bought 3 monitors for $900."}
    ],
    temperature=0.0
)
print(response.choices[0].message.content)
```

#### B. S-LoRA & Predibase LoRAX (Specialized Multi-LoRA Kernels)
* Uses Unified Paging for LoRA adapters in GPU memory.
* Supports **thousands of fine-tuned adapters** on a single GPU with <5% overhead compared to the base model alone.

### Pros & Cons
* ✅ **Massive Cost Reduction**: Serve 50+ specialized task models on a single GPU instead of 50 separate GPUs (>95% infrastructure cost savings).
* ✅ **Zero Cold-Start for New Tasks**: Adapters load in milliseconds without rebooting the server.
* ❌ **Throughput Trade-off**: Heterogeneous batching incurs a slight (~5–10%) latency penalty compared to a purely fused monolithic model.

---

## 3. Strategy 3: Quantized High-Speed Inference (AWQ / GPTQ / Marlin)

### Concept
While `bitsandbytes` (NF4) is optimal for **training on budget GPUs**, it is not optimized for high-throughput production serving. For production inference, we merge the adapter in FP16 and quantize into **AWQ (Activation-aware Weight Quantization)** or **GPTQ / Marlin**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             PRODUCTION QUANTIZATION PIPELINE                             │
│                                                                                          │
│  [Train QLoRA (NF4)] ──► [Merge in FP16] ──► [AutoAWQ INT4] ──► [vLLM Marlin Kernel]    │
│   (Budget Training)       (Full Precision)    (4-bit Inference)   (3.5x Higher Tok/s)    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### AutoAWQ Conversion Recipe
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "models/merged/qwen-1.5b-order-extractor-fused"
quant_path = "models/quantized/qwen-1.5b-order-extractor-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# 1. Load fused model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Quantize using calibration dataset
model.quantize(tokenizer, quant_config=quant_config)

# 3. Save AWQ weights
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"✅ Production AWQ model saved to: {quant_path}")
```

### Benchmarks (1.5B–7B Models)
* **VRAM Footprint**: ~1.1 GB (1.5B) / ~4.5 GB (7B).
* **Serving Speed**: ~3.2x faster token generation than unquantized FP16 on NVIDIA Ampere/Ada Tensor Cores.

---

## 4. Strategy 4: Edge & Local CPU Deployment (GGUF + llama.cpp / Ollama)

### Concept
Convert the fused model to **GGUF** format to enable ultra-lightweight execution on consumer CPUs, Apple Silicon Macs, or edge appliances without dedicated GPUs.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                GGUF EXPORT & SERVING                                     │
│                                                                                          │
│  Fused HF Model ──► convert_hf_to_gguf.py ──► llama-quantize (Q4_K_M) ──► Ollama / CPU   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Conversion:
```bash
# 1. Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && pip install -r requirements.txt

# 2. Convert HuggingFace fused directory to GGUF (FP16)
python convert_hf_to_gguf.py ../models/merged/qwen-1.5b-order-extractor-fused/ \
    --outfile models/qwen-1.5b-order-extractor-f16.gguf

# 3. Quantize to 4-bit (Q4_K_M)
./llama-quantize models/qwen-1.5b-order-extractor-f16.gguf \
    models/qwen-1.5b-order-extractor-q4_k_m.gguf Q4_K_M

# 4. Create Ollama Model File
cat <<EOF > Modelfile
FROM ./models/qwen-1.5b-order-extractor-q4_k_m.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.0
EOF

# 5. Build and run in Ollama
ollama create order-extractor -f Modelfile
ollama run order-extractor "John bought 3 laptops for $2400."
```

---

## 5. Strategy 5: Containerized FastAPI Microservice

### Concept
Package the model behind a production-ready asynchronous Python REST API using **Lifespan event handlers** (zero cold-start per request), Pydantic validation, health checks, and Docker containerization.

### FastAPI Architecture (`api/main.py`)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Global model state holder
class ModelService:
    model = None
    tokenizer = None

model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: Load model weights once into GPU VRAM
    base_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_id = "models/adapters/qwen-1.5b-order-extractor"
    
    model_service.tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model_service.model = PeftModel.from_pretrained(base_model, adapter_id)
    model_service.model.eval()
    yield
    # SHUTDOWN: Clean up GPU memory
    del model_service.model
    torch.cuda.empty_cache()

app = FastAPI(title="Order Extractor LLM Service", lifespan=lifespan)

class ExtractionRequest(BaseModel):
    text: str = Field(..., example="Alice bought 2 phones for $600 delivery Monday.")

class OrderOutput(BaseModel):
    customer: str
    quantity: int
    product: str
    amount: float
    delivery_day: str

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

@app.post("/v1/extract", response_model=OrderOutput)
async def extract_order(req: ExtractionRequest):
    messages = [
        {"role": "system", "content": "Extract structured order information into JSON."},
        {"role": "user", "content": req.text}
    ]
    prompt = model_service.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = model_service.tokenizer(prompt, return_tensors="pt").to(model_service.model.device)
    
    with torch.no_grad():
        outputs = model_service.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,
            do_sample=False,
            eos_token_id=model_service.tokenizer.eos_token_id
        )
    
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = model_service.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    import json
    try:
        return json.loads(response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM JSON: {response_text}")
```

### Docker Containerization (`Dockerfile`)
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 Comprehensive Production Decision Matrix

| Dimension | Strategy 1: Fused Standalone | Strategy 2: Multi-LoRA (vLLM) | Strategy 3: AWQ Quantized | Strategy 4: GGUF Edge | Strategy 5: FastAPI Microservice |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Best Used For** | High-traffic single task | Many tasks on 1 GPU | Max GPU throughput | CPU / Local / Mobile | Full custom backend logic |
| **Inference Latency** | ⭐⭐⭐⭐⭐ Lowest | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Ultra Fast | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Fast |
| **Throughput (req/s)** | ⭐⭐⭐⭐⭐ Highest | ⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐⭐ Highest | ⭐⭐ Low–Moderate | ⭐⭐⭐ Moderate |
| **VRAM per Model** | Full Model (~3–14 GB) | **~70 MB per adapter** | Compressed (~1.1–4.5 GB) | **0 GB GPU (Uses CPU RAM)** | Full Model (~3–14 GB) |
| **Multi-Tenancy** | ❌ 1 Model per Port | ✅ **100+ Adapters on 1 Port** | ❌ 1 Model per Port | ❌ 1 Model per Process | ⚠️ Manual Swapping |
| **Engine Stack** | vLLM / TGI / TensorRT | vLLM / S-LoRA / LoRAX | vLLM / AutoAWQ | llama.cpp / Ollama | Uvicorn / PyTorch / PEFT |
| **Operational Complexity**| Low | Medium | Medium | Very Low | Low–Medium |

---

## 🎯 Recommended Production Blueprint

1. **For Enterprise SaaS (Multiple Tools/Customers)**:
   * **Use Strategy 2 (vLLM Multi-LoRA)**. Host one base model (e.g. `Qwen2.5-7B` or `Llama-3.1-8B`) and mount dozens of task-specific 70 MB LoRA adapters.
2. **For High-Traffic Single Task (e.g., Structured Order Extraction)**:
   * **Use Strategy 1 + 3 (Merge $\rightarrow$ AWQ $\rightarrow$ vLLM)**. Merge the adapter weights into the base model, quantize with AWQ, and deploy behind vLLM with PagedAttention.
3. **For Local / Desktop / On-Premise (Zero GPU Hardware)**:
   * **Use Strategy 4 (Merge $\rightarrow$ GGUF $\rightarrow$ Ollama)**. Export as GGUF Q4_K_M and run on standard CPU servers.
