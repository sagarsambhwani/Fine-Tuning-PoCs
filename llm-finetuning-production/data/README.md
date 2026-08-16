# Dataset Directory

This directory stores raw and processed data for domain-specific fine-tuning (Natural Language $\to$ Structured JSON Extraction).

## Directory Structure

```
data/
├── raw/               # Raw generation logs, uncurated samples
└── processed/         # Formatted, validated, deduplicated splits
    ├── train.jsonl    # Training split (1,200 examples)
    ├── val.jsonl      # Validation split (150 examples)
    └── test.jsonl     # Held-out Test split (150 examples)
```

## Schema Format

Each line in `train.jsonl`, `val.jsonl`, and `test.jsonl` is a valid JSON object containing conversational messages formatted for Causal LM training:

```json
{
  "id": "order_0001",
  "instruction": "Extract the order details into structured JSON.",
  "input": "John ordered 3 laptops for $2400 and wants delivery on Friday.",
  "output": {
    "customer": "John",
    "quantity": 3,
    "product": "laptops",
    "amount": 2400.0,
    "delivery_day": "Friday"
  },
  "messages": [
    {
      "role": "system",
      "content": "You are an expert entity extraction system. Extract structured order information into valid JSON according to the schema: {\"customer\": string, \"quantity\": integer, \"product\": string, \"amount\": float, \"delivery_day\": string}."
    },
    {
      "role": "user",
      "content": "John ordered 3 laptops for $2400 and wants delivery on Friday."
    },
    {
      "role": "assistant",
      "content": "{\"customer\": \"John\", \"quantity\": 3, \"product\": \"laptops\", \"amount\": 2400.0, \"delivery_day\": \"Friday\"}"
    }
  ]
}
```

## Reproducibility & Anti-Leakage
- Splits are generated deterministically using seed `42`.
- Deduplication and template stratification ensure zero data leakage between train, validation, and test partitions.
- Generate or re-generate splits using:
  ```bash
  python src/data/dataset_generator.py --seed 42 --train 1200 --val 150 --test 150
  ```
