"""
SFT + QLoRA Training Pipeline Runner
Handles dataset loading, training arguments setup, SFTTrainer execution,
and experiment metrics tracking (JSON & CSV).
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import get_peft_model

from src.training.lora_config import get_lora_config, print_trainable_parameters
from src.training.qlora_config import get_qlora_bnb_config, prepare_qlora_model
from src.utils.logger import setup_logger

logger = setup_logger("training_pipeline")

def load_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_sft_training(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """
    Executes end-to-end SFT + QLoRA training run.
    Saves adapter weights and metrics file.
    """
    cfg = load_config(config_path)
    model_id = cfg["model"]["default_model_id"]
    output_dir = cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        use_fast=cfg["model"].get("use_fast_tokenizer", True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check CUDA
    device_map = "auto" if torch.cuda.is_available() else None
    
    # 4-bit Quantization Config
    if torch.cuda.is_available() and cfg["qlora"].get("load_in_4bit", True):
        logger.info("Configuring 4-bit BitsAndBytes NF4 Quantization...")
        bnb_config = get_qlora_bnb_config(
            compute_dtype=cfg["qlora"].get("bnb_4bit_compute_dtype", "float16"),
            use_double_quant=cfg["qlora"].get("bnb_4bit_use_double_quant", True),
            quant_type=cfg["qlora"].get("bnb_4bit_quant_type", "nf4")
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=cfg["model"].get("trust_remote_code", False),
            torch_dtype=torch.float16
        )
        base_model = prepare_qlora_model(
            base_model,
            use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True)
        )
    else:
        logger.info("Loading model in standard precision (CPU / Non-quantized)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=cfg["model"].get("trust_remote_code", False),
            torch_dtype=torch.float32
        )

    # Attach LoRA adapter
    lora_cfg = get_lora_config(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"]
    )
    model = get_peft_model(base_model, lora_cfg)
    param_stats = print_trainable_parameters(model)

    # Load dataset
    data_files = {
        "train": cfg["data"]["train_file"],
        "val": cfg["data"]["val_file"]
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    logger.info(f"Loaded train ({len(raw_datasets['train'])}) and val ({len(raw_datasets['val'])}) samples.")

    # Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy=cfg["training"].get("eval_strategy", "steps"),
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        fp16=cfg["training"]["fp16"] if torch.cuda.is_available() else False,
        bf16=cfg["training"]["bf16"] if torch.cuda.is_available() else False,
        optim=cfg["training"]["optim"] if torch.cuda.is_available() else "adamw_torch",
        report_to=cfg["training"]["report_to"],
        seed=cfg["project"]["seed"]
    )

    # Import SFTTrainer safely
    try:
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["val"],
            tokenizer=tokenizer,
            max_seq_length=cfg["data"]["max_seq_length"]
        )
    except Exception as e:
        logger.warning(f"Using standard Transformers Trainer fallback due to: {e}")
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["val"],
            tokenizer=tokenizer
        )

    logger.info("Starting training execution...")
    train_result = trainer.train()

    # Save adapter & tokenizer
    logger.info(f"Saving fine-tuned adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metrics
    metrics_path = Path("reports/training_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        "model_id": model_id,
        "train_loss": train_result.training_loss,
        "train_runtime_seconds": train_result.metrics.get("train_runtime", 0.0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0),
        "trainable_parameters": param_stats,
        "hyperparameters": cfg
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info(f"Training metrics saved to {metrics_path}")

    return metrics_data

if __name__ == "__main__":
    run_sft_training()
