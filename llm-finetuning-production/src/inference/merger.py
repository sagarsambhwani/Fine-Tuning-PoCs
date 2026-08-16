"""
LoRA Adapter Merger & Exporter
Fuses trained low-rank adapter matrices (B @ A) directly into base model weights (W + dW),
unloading PEFT overhead to produce standalone model checkpoints for high-throughput serving.
"""
import os
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.utils.logger import setup_logger

logger = setup_logger("adapter_merger")

def merge_adapter_to_base(
    base_model_id: str,
    adapter_dir: str,
    output_dir: str,
    torch_dtype: str = "float16"
):
    """Merges LoRA adapter into base weights and saves standalone model."""
    logger.info(f"Loading base model: {base_model_id} in {torch_dtype}...")
    dtype = torch.float16 if torch_dtype in ("float16", "fp16") else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="cpu", # CPU merge avoids VRAM spikes
        low_cpu_mem_usage=True
    )

    logger.info(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    logger.info(f"Attaching LoRA adapter from {adapter_dir}...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Executing merge_and_unload()...")
    merged_model = lora_model.merge_and_unload()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged standalone model to {output_dir}...")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merge and export completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_dir", type=str, default="models/adapters/qwen-1.5b-order-extractor")
    parser.add_argument("--output_dir", type=str, default="models/merged/qwen-1.5b-order-extractor")
    parser.add_argument("--dtype", type=str, default="float16")

    args = parser.parse_args()
    merge_adapter_to_base(args.base_model, args.adapter_dir, args.output_dir, args.dtype)
