import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertTokenizer


def load_multilabel_model(model_dir: str):
    model_path = Path(model_dir)
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    metadata_path = model_path.parent / "metadata.json"
    id2label = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in metadata.get("id2label", {}).items()}
    else:
        id2label = model.config.id2label

    return model, tokenizer, id2label


def predict_labels(
    text: str, model, tokenizer, id2label: Dict[int, str], threshold: float = 0.5
) -> List[Tuple[str, float]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
        # Apply Sigmoid activation (independent probability per class)
        probabilities = torch.sigmoid(logits).tolist()

    results = []
    for idx, prob in enumerate(probabilities):
        label_name = id2label.get(idx, f"label_{idx}")
        if prob >= threshold:
            results.append((label_name, round(prob, 4)))

    # Sort descending by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on multi-label classifier")
    parser.add_argument(
        "--model-dir",
        default="multilabel-classification/models/final-model",
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Input sentence for classification",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold (0.0 to 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, id2label = load_multilabel_model(args.model_dir)
    active_labels = predict_labels(args.text, model, tokenizer, id2label, threshold=args.threshold)

    print(f"\nText: '{args.text}'")
    print(f"Threshold: {args.threshold}")
    print("Predicted Active Categories:")
    if not active_labels:
        print("  (No categories exceeded the threshold)")
    else:
        for label, prob in active_labels:
            print(f"  - {label}: {prob * 100:.2f}%")
