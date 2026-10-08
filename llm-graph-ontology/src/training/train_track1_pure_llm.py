"""
Track 1: Pure LLM Fine-Tuning Pipeline for Graph & Ontology Adaptation.
Trains Qwen2.5-1.5B-Instruct using QLoRA (4-bit NF4) + SFT on legal ontology tasks.
"""
import os
import sys
import json
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def parse_args():
    parser = argparse.ArgumentParser(description="Track 1: Pure LLM QLoRA Fine-Tuning")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="llm-graph-ontology/data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="llm-graph-ontology/data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="llm-graph-ontology/models/qwen-1.5b-legal-graph-adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()

def load_jsonl_dataset(filepath: str, tokenizer) -> Dataset:
    """Loads JSONL chat records and formats them using the tokenizer's chat template."""
    with open(filepath, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    formatted_texts = []
    for r in records:
        messages = r["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted_texts.append({"text": text, "task_type": r.get("task_type", "unknown")})

    return Dataset.from_list(formatted_texts)

def main():
    args = parse_args()

    print("=" * 65)
    print("🚀 TRACK 1: PURE LLM FINE-TUNING PIPELINE (QLoRA + SFT)")
    print("=" * 65)
    print(f"Base Model        : {args.model_id}")
    print(f"Train File        : {args.train_file}")
    print(f"Val File          : {args.val_file}")
    print(f"Output Adapter    : {args.output_dir}")
    print(f"LoRA Rank (r)     : {args.lora_r}")
    print(f"LoRA Alpha        : {args.lora_alpha}")
    print(f"Epochs            : {args.epochs}")
    print(f"Batch Size (eff.) : {args.batch_size * args.grad_accum} ({args.batch_size} x {args.grad_accum})")
    print("=" * 65)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Datasets
    print("\nLoading and formatting datasets...")
    train_dataset = load_jsonl_dataset(args.train_file, tokenizer)
    val_dataset = load_jsonl_dataset(args.val_file, tokenizer)
    print(f"[OK] Train samples loaded: {len(train_dataset)}")
    print(f"[OK] Val samples loaded  : {len(val_dataset)}")

    # 3. Model Loading with 4-Bit NF4 Quantization
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print("[WARN] CUDA not detected. Loading FP32 on CPU (slow execution).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
            device_map="auto"
        )

    # 4. LoRA Configuration targeting all linear layers
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()
    print(f"[OK] Trainable Parameters : {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        fp16=is_cuda,
        optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
        report_to="none"
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args
    )

    print("\n🚀 Starting SFT Training...")
    train_result = trainer.train()

    # 7. Save Adapter & Artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[SUCCESS] Adapter weights successfully saved to: {args.output_dir}")
    print(f"[SUCCESS] Training metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
