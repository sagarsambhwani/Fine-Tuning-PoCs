import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = ["normal_operation", "critical_incident"]
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

CUSTOM_TOKENS = [
    "[LOG_CRITICAL]",
    "[ENV_PROD]",
    "[ENV_STAGING]",
    "[SYSTEM_ALERT]",
    "[INCIDENT_SECURITY]",
    "[INCIDENT_PERFORMANCE]",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_domain_dataset(seed: int) -> DatasetDict:
    """
    Creates a domain-specific log classification dataset using custom system tags.
    Label 0: normal_operation
    Label 1: critical_incident
    """
    sample_data: List[Tuple[str, int]] = [
        ("[LOG_CRITICAL] [ENV_PROD] [INCIDENT_SECURITY] Unauthorized root access detected on host 10.0.4.12!", 1),
        ("[LOG_CRITICAL] [ENV_PROD] [INCIDENT_PERFORMANCE] Database deadlock in transaction pool after timeout.", 1),
        ("[SYSTEM_ALERT] [ENV_PROD] Memory usage spiked above 98% threshold on container primary-0.", 1),
        ("[LOG_CRITICAL] [ENV_PROD] SSL certificate validation failure on external payment gateway API.", 1),
        ("[SYSTEM_ALERT] [ENV_STAGING] High latency detected during performance load test execution.", 1),
        ("[ENV_PROD] System health check succeeded: all 12 worker processes reporting status OK.", 0),
        ("[ENV_STAGING] User login flow test executed successfully with 0 errors.", 0),
        ("[ENV_PROD] Scheduled batch backup completed in 42 seconds. Bytes written: 4.2GB.", 0),
        ("[ENV_STAGING] Cache warm-up job finished successfully for region us-east-1.", 0),
        ("[ENV_PROD] Routine garbage collection finished, memory reclaimed: 250MB.", 0),
        ("[LOG_CRITICAL] [ENV_PROD] [INCIDENT_SECURITY] DDoS attack pattern recognized on gateway router!", 1),
        ("[SYSTEM_ALERT] [ENV_PROD] [INCIDENT_PERFORMANCE] CPU throttling active due to thermal limits.", 1),
        ("[ENV_PROD] Cron job sync_user_metrics executed without errors.", 0),
        ("[ENV_STAGING] Feature flag sync finished: 14 flags active.", 0),
    ]

    rng = random.Random(seed)
    rows = []
    # Expand data to 120 samples synthetically
    for _ in range(120):
        text, label = rng.choice(sample_data)
        rows.append({"text": text, "label": label})

    rng.shuffle(rows)
    split_index = int(0.8 * len(rows))
    train_rows = rows[:split_index]
    eval_rows = rows[split_index:]

    return DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(eval_rows),
    })


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"accuracy": acc, "f1": f1}


def initialize_new_token_embeddings(model, old_vocab_size: int, new_vocab_size: int) -> None:
    """
    Initializes newly added token embedding weights to the mean vector of existing embeddings.
    Prevents large initial gradient spikes during fine-tuning.
    """
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        old_weights = embeddings.weight[:old_vocab_size]
        mean_embedding = old_weights.mean(dim=0)

        for i in range(old_vocab_size, new_vocab_size):
            embeddings.weight[i] = mean_embedding.clone()


def main():
    parser = argparse.ArgumentParser(description="PoC 1.3: Train Model with Custom Vocabulary")
    parser.add_argument("--model-name", type=str, default="distilbert/distilbert-base-uncased", help="Base HF model")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="tokenization-custom-vocab/models/custom_distilbert", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PoC 1.3: Custom Vocabulary Fine-Tuning Pipeline")
    print("=" * 60)

    # 1. Load Tokenizer & Add Custom Tokens
    print(f"\n1. Loading base tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    old_vocab_size = len(tokenizer)
    print(f"   Original vocabulary size: {old_vocab_size}")

    print(f"   Adding custom tokens: {CUSTOM_TOKENS}")
    num_added = tokenizer.add_tokens(CUSTOM_TOKENS)
    new_vocab_size = len(tokenizer)
    print(f"   Added {num_added} tokens. New vocabulary size: {new_vocab_size}")

    # 2. Load Model & Resize Embeddings
    print(f"\n2. Loading base model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("   Resizing model token embeddings...")
    model.resize_token_embeddings(new_vocab_size)
    print(f"   Resized embedding shape: {model.get_input_embeddings().weight.shape}")

    print("   Initializing new token embeddings with mean vector...")
    initialize_new_token_embeddings(model, old_vocab_size, new_vocab_size)

    # 3. Prepare Dataset
    print("\n3. Generating domain-specific dataset with custom tokens...")
    dataset = create_synthetic_domain_dataset(args.seed)
    print(f"   Train samples: {len(dataset['train'])}, Validation samples: {len(dataset['validation'])}")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Training Setup
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
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

    # 5. Execute Training
    print("\n4. Starting Training...")
    trainer.train()

    # 6. Evaluate Best Model
    print("\n5. Evaluating Final Model...")
    eval_results = trainer.evaluate()
    print(f"   Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   Validation F1 Score: {eval_results['eval_f1']:.4f}")

    # 7. Save Model & Custom Tokenizer
    print(f"\n6. Saving fine-tuned model and custom tokenizer to: {output_path.resolve()}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\n" + "=" * 60)
    print("Training complete! Model and custom tokenizer saved successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
