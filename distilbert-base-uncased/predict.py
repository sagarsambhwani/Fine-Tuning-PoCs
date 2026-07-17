import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_sentiment(text: str, model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()
    label_id = int(torch.argmax(outputs.logits, dim=-1).item())
    label_name = model.config.id2label.get(label_id, str(label_id))
    return label_name, probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment for a given text")
    parser.add_argument("--model-dir", default="models/sentiment/final-model")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    label, probs = predict_sentiment(args.text, args.model_dir)
    print(json.dumps({"label": label, "probabilities": probs}, indent=2))
