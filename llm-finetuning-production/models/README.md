# Model Artifacts Directory

This directory holds adapter checkpoints, merged full weights, and quantized models. Large binary files (`*.safetensors`, `*.bin`, `*.pt`, `*.gguf`) are excluded from version control via `.gitignore`.

## Layout

```
models/
├── adapters/
│   └── qwen-1.5b-order-extractor/      # LoRA/PEFT adapter weights (`adapter_model.safetensors`, `adapter_config.json`)
├── merged/
│   └── qwen-1.5b-order-extractor/      # Merged standalone base + adapter model for fast serving
└── quantized/
    └── qwen-1.5b-order-extractor-4bit/ # Quantized standalone weights for reduced VRAM production inference
```

## How to Export & Merge
To merge LoRA adapters into standalone weights:
```bash
python src/inference/merger.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_dir models/adapters/qwen-1.5b-order-extractor \
  --output_dir models/merged/qwen-1.5b-order-extractor
```
