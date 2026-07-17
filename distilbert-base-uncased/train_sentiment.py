import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.isAvailable():
        torch.cuda.manual_seed_all(seed)


def build_dataset(train_samples: int, eval_samples: int, seed: int) -> DatasetDict:
    try:
        dataset = load_dataset("imdb")
        train_subset = dataset["train"].shuffle(seed=seed).select(range(min(train_samples, len(dataset["train"]))))
        eval_subset = dataset["test"].shuffle(seed=seed).select(range(min(eval_samples, len(dataset["test"]))))
        return DatasetDict({"train": train_subset, "validation": eval_subset})
    except Exception as exc:
        print(f"Falling back to a synthetic sentiment dataset because IMDb download failed: {exc}")

        positive_examples = [
            "I absolutely loved this movie and would watch it again.",
            "This was a fantastic experience with great pacing and wonderful acting.",
            "Amazing production quality and an uplifting story from start to finish.",
            "The soundtrack was beautiful and the characters felt real.",
            "A perfect blend of humor, emotion, and suspense.",
            "I enjoyed every minute and recommend it to everyone.",
        ]
        negative_examples = [
            "This movie was dull, slow, and badly written.",
            "I hated the pacing and the ending felt rushed.",
            "The acting was poor and the plot made no sense.",
            "It was a frustrating experience from beginning to end.",
            "The film felt cheap and uninteresting.",
            "I regret watching it and would not recommend it.",
        ]
        rows = [{"text": text, "label": 1} for text in positive_examples] + [{"text": text, "label": 0} for text in negative_examples]
        random.Random(seed).shuffle(rows)
        split_index = int(0.8 * len(rows))
        train_rows = rows[:split_index]
        eval_rows = rows[split_index:]
        return DatasetDict(
            {
                "train": Dataset.from_list(train_rows[:train_samples]),
                "validation": Dataset.from_list(eval_rows[:eval_samples]),
            }
        )


def tokenize_dataset(tokenizer, dataset: DatasetDict, max_length: int = 128) -> DatasetDict:
    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def train_model(args: argparse.Namespace) -> Dict[str, float]:
    set_global_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args.train_samples, args.eval_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_dataset = tokenize_dataset(tokenizer, dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        save_total_limit=1,
        logging_dir=str(output_dir / "logs"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir / "final-model"))
    tokenizer.save_pretrained(str(output_dir / "final-model"))

    metadata = {"model_name": args.model_name, "metrics": metrics, "labels": ["negative", "positive"]}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a sentiment classifier")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model to fine-tune")
    parser.add_argument("--output-dir", default="models/sentiment", help="Directory for the trained model")
    parser.add_argument("--train-samples", type=int, default=120, help="Number of training examples")
    parser.add_argument("--eval-samples", type=int, default=40, help="Number of evaluation examples")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_model(args)
    print(json.dumps(metrics, indent=2))
