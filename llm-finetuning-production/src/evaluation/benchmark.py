"""
Evaluation Benchmark Suite
Runs evaluation on held-out test split for Base vs. Fine-Tuned models.
"""
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from src.evaluation.metrics import evaluate_predictions
from src.data.formatter import format_chat_prompt, apply_template_to_messages
from src.utils.logger import setup_logger

logger = setup_logger("benchmark_runner")

def load_test_dataset(test_file: str = "data/processed/test.jsonl") -> List[Dict[str, Any]]:
    """Loads test JSONL dataset."""
    test_path = Path(test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_file}. Generate it first using src/data/dataset_generator.py.")
    
    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples

def run_evaluation_benchmark(
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path: Optional[str] = None,
    test_file: str = "data/processed/test.jsonl",
    max_new_tokens: int = 128,
    limit_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Runs batch inference and computes evaluation metrics on held-out test split."""
    test_samples = load_test_dataset(test_file)
    if limit_samples:
        test_samples = test_samples[:limit_samples]

    logger.info(f"Loaded {len(test_samples)} test samples from {test_file}.")
    logger.info(f"Loading tokenizer: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    logger.info(f"Loading model: {base_model_id}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None
    )

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Attaching LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info("Running in ZERO-SHOT BASE MODEL mode.")

    model.eval()

    raw_predictions = []
    ground_truths = [s["output"] for s in test_samples]
    inputs = [s["input"] for s in test_samples]

    start_time = time.time()
    for idx, sample in enumerate(test_samples):
        messages = format_chat_prompt(sample["input"])
        prompt_text = apply_template_to_messages(messages, tokenizer, add_generation_prompt=True)

        inputs_tensor = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Extract only the generated completion tokens
        input_len = inputs_tensor["input_ids"].shape[1]
        completion = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        raw_predictions.append(completion)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(test_samples):
            logger.info(f"Processed {idx + 1}/{len(test_samples)} samples...")

    total_time = time.time() - start_time
    avg_latency_ms = (total_time / len(test_samples)) * 1000

    metrics, sample_results = evaluate_predictions(raw_predictions, ground_truths)
    metrics["benchmark_time_seconds"] = round(total_time, 2)
    metrics["avg_latency_ms"] = round(avg_latency_ms, 2)
    metrics["model_evaluated"] = f"{base_model_id} + {adapter_path}" if adapter_path else f"{base_model_id} (Base)"

    logger.info("=== BENCHMARK RESULTS ===")
    logger.info(f"Model               : {metrics['model_evaluated']}")
    logger.info(f"JSON Validity Rate  : {metrics['json_validity_rate'] * 100:.2f}%")
    logger.info(f"Exact Match Rate    : {metrics['exact_match_rate'] * 100:.2f}%")
    logger.info(f"Avg Field Accuracy  : {metrics['avg_field_accuracy'] * 100:.2f}%")
    logger.info(f"Field Accuracies    : {metrics['field_accuracies']}")
    logger.info(f"Avg Latency         : {metrics['avg_latency_ms']} ms/sample")

    return {
        "metrics": metrics,
        "sample_evaluations": [
            {
                "input": inp,
                "ground_truth": gt,
                "prediction": pred,
                "eval": res
            }
            for inp, gt, pred, res in zip(inputs, ground_truths, raw_predictions, sample_results)
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Base vs. Fine-Tuned Model on Test Split.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--test_file", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--out_json", type=str, default="reports/evaluation_results.json")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    results = run_evaluation_benchmark(
        base_model_id=args.base_model,
        adapter_path=args.adapter,
        test_file=args.test_file,
        limit_samples=args.limit
    )

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_p}")
