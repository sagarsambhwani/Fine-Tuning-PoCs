import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="PoC 1.3: Run Inference with Custom Vocabulary Classifier")
    parser.add_argument(
        "--text",
        type=str,
        default="[LOG_CRITICAL] [ENV_PROD] Connection pool exhausted on primary server node",
        help="Input text containing custom tokens",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="tokenization-custom-vocab/models/custom_distilbert",
        help="Path to saved model directory",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path)
    if not model_dir.exists():
        print(f"Error: Saved model directory '{args.model_path}' not found.")
        print("Please train the model first by running:")
        print("   python tokenization-custom-vocab/train_custom_vocab.py")
        sys.exit(1)

    print("=" * 60)
    print("PoC 1.3: Inference with Custom Vocabulary")
    print("=" * 60)
    print(f"\n1. Loading model & custom tokenizer from: '{model_dir}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    text = args.text
    print(f"\n2. Input Text: \"{text}\"")

    # Inspect tokenization
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    print(f"   Tokenized Output: {tokens}")
    print(f"   Token IDs:        {token_ids}")

    # Prepare tensor
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    predicted_label = model.config.id2label[predicted_class_id]
    confidence = probs[predicted_class_id]

    print("\n3. Classification Results:")
    print(f"   Predicted Class: {predicted_label.upper()}")
    print(f"   Confidence:      {confidence * 100:.2f}%")
    print("\n   Probabilities Breakdown:")
    for class_id, prob in enumerate(probs):
        label_name = model.config.id2label[class_id]
        print(f"     - {label_name:<20}: {prob * 100:6.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
