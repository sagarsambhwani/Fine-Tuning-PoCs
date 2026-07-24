import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import f1_score, hamming_loss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = ["billing", "tech_support", "urgency", "feedback", "feature_request"]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_synthetic_multilabel_dataset(train_samples: int, eval_samples: int, seed: int) -> DatasetDict:
    sample_data = [
        ("My invoice is completely wrong and I demand a refund right now!", [1, 0, 1, 0, 0]),
        ("The website is throwing 500 server errors and I cannot log into my dashboard.", [0, 1, 1, 0, 0]),
        ("I love the new dark mode theme, but can you add keyboard shortcuts for navigation?", [0, 0, 0, 1, 1]),
        ("Urgent: API endpoint latency is failing production builds!", [0, 1, 1, 0, 0]),
        ("Where can I update my credit card payment details?", [1, 0, 0, 0, 0]),
        ("Your customer service was super helpful today, thanks!", [0, 0, 0, 1, 0]),
        ("Please add CSV export option to the reporting analytics tab.", [0, 0, 0, 0, 1]),
        ("Billing issue: double charge on my monthly subscription bill emergency!", [1, 0, 1, 0, 0]),
        ("Database connection timeout occurs every time I run export query.", [0, 1, 0, 0, 0]),
        ("Great app overall! Would be awesome to support multi-language localizations.", [0, 0, 0, 1, 1]),
        ("Cannot reset password, email verification code never arrives.", [0, 1, 0, 0, 0]),
        ("Overcharged by $50 on current cycle, please fix asap!", [1, 0, 1, 0, 0]),
    ]
    
    # Expand data synthetically for training
    rng = random.Random(seed)
    rows = []
    for _ in range(max(10, train_samples + eval_samples)):
        text, label_vec = rng.choice(sample_data)
        rows.append({"text": text, "labels": [float(x) for x in label_vec]})
    
    rng.shuffle(rows)
    split_index = int(0.8 * len(rows))
    train_rows = rows[:split_index]
    eval_rows = rows[split_index:]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows[:train_samples]),
            "validation": Dataset.from_list(eval_rows[:eval_samples]),
        }
    )


def build_dataset(train_samples: int, eval_samples: int, seed: int) -> DatasetDict:
    try:
        raw_ds = load_dataset("google/go_emotions", "simplified")
        print("Successfully loaded google/go_emotions dataset.")
        train_subset = raw_ds["train"].shuffle(seed=seed).select(range(min(train_samples, len(raw_ds["train"]))))
        eval_subset = raw_ds["validation"].shuffle(seed=seed).select(range(min(eval_samples, len(raw_ds["validation"]))))
        
        num_labels = 28
        def convert_labels(example):
            vec = [0.0] * num_labels
            for idx in example["labels"]:
                vec[idx] = 1.0
            return {"labels": vec}
        
        train_subset = train_subset.map(convert_labels)
        eval_subset = eval_subset.map(convert_labels)
        return DatasetDict({"train": train_subset, "validation": eval_subset})
    except Exception as exc:
        print(f"Using synthetic multi-label dataset (Offline/Fallback mode).")
        return build_synthetic_multilabel_dataset(train_samples, eval_samples, seed)


def tokenize_dataset(tokenizer, dataset: DatasetDict, max_length: int = 128) -> DatasetDict:
    def tokenize_batch(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        result["labels"] = examples["labels"]
        return result

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probs >= 0.5).astype(int)
    
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    h_loss = hamming_loss(labels, predictions)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "hamming_loss": float(h_loss),
    }


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.train_samples, args.eval_samples, args.seed)
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)

    num_labels = len(dataset["train"][0]["labels"])
    id2label = {i: f"label_{i}" for i in range(num_labels)} if num_labels != len(LABEL_NAMES) else ID2LABEL
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_micro_f1",
        save_total_limit=1,
        logging_dir=str(output_dir / "logs"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir / "final-model"))
    tokenizer.save_pretrained(str(output_dir / "final-model"))

    metadata = {
        "model_name": args.model_name,
        "metrics": metrics,
        "id2label": id2label,
        "num_labels": num_labels,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a multi-label text classifier")
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased", help="Model checkpoint")
    parser.add_argument("--output-dir", default="multilabel-classification/models", help="Output directory")
    parser.add_argument("--train-samples", type=int, default=100, help="Training samples count")
    parser.add_argument("--eval-samples", type=int, default=30, help="Evaluation samples count")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_model(args)
    print("Training Complete. Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
