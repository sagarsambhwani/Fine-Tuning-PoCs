"""
Domain-Specific Synthetic Dataset Generator
Generates realistic, varied e-commerce/customer-support text prompts paired with
strict structured JSON extraction targets. Implements deterministic splits (train/val/test)
with anti-leakage checks and full reproducibility.
"""
import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

SYSTEM_PROMPT = (
    "You are an expert entity extraction system. Extract structured order information "
    "from the user's text into valid JSON according to the schema: "
    '{"customer": string, "quantity": integer, "product": string, "amount": float, "delivery_day": string}.'
)

FIRST_NAMES = [
    "John", "Sarah", "Michael", "Emma", "David", "Emily", "James", "Olivia",
    "Robert", "Sophia", "William", "Ava", "Alexander", "Mia", "Daniel", "Charlotte",
    "Matthew", "Amelia", "Lucas", "Harper", "Benjamin", "Evelyn", "Elijah", "Abigail",
    "Liam", "Elizabeth", "Henry", "Sofia", "Oliver", "Victoria", "Noah", "Aria",
    "Ethan", "Chloe", "Jacob", "Camila", "Logan", "Penelope", "Jackson", "Riley",
    "Aarav", "Priya", "Ananya", "Rohan", "Mei", "Chen", "Kenji", "Fatima", "Tariq", "Elena"
]

PRODUCTS = [
    ("laptops", 800, 2500),
    ("ultrabooks", 900, 2200),
    ("mechanical keyboards", 50, 250),
    ("wireless mice", 25, 120),
    ("4K monitors", 300, 1200),
    ("curved gaming monitors", 400, 1500),
    ("ergonomic office chairs", 150, 800),
    ("standing desks", 350, 1200),
    ("noise-canceling headphones", 120, 450),
    ("smartphones", 400, 1400),
    ("tablet computers", 250, 1100),
    ("laser printers", 180, 750),
    ("USB-C docking stations", 60, 250),
    ("external SSDs (2TB)", 100, 300),
    ("smartwatches", 150, 600),
    ("webcams with mic", 40, 180),
    ("server rack mounts", 120, 650),
    ("network switches", 80, 950),
    ("graphics cards", 450, 1800),
    ("Bluetooth speakers", 35, 200)
]

DELIVERY_DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "next Monday", "next Wednesday", "next Friday", "tomorrow", "this weekend"
]

SENTENCE_TEMPLATES = [
    "{customer} ordered {quantity} {product} for ${amount} and wants delivery on {delivery_day}.",
    "Please send {quantity} {product} to {customer}. The total is ${amount} and it needs to arrive on {delivery_day}.",
    "Invoice #{inv}: Customer {customer} purchased {quantity} units of {product}. Billed: ${amount}. Scheduled delivery: {delivery_day}.",
    "Order confirmation: {quantity} {product} bought by {customer} for a total price of ${amount}, shipping out for {delivery_day}.",
    "{customer} requested an order of {quantity} {product} priced at ${amount}. Delivery target: {delivery_day}.",
    "Hi support, this is {customer}. I placed an order for {quantity} {product} amounting to ${amount}. Can you deliver on {delivery_day}?",
    "New transaction registered: {customer} - {product} (Qty: {quantity}) - Total: ${amount} - Requested Day: {delivery_day}.",
    "Procurement request from {customer}: {quantity} {product} worth ${amount}, please ensure arrival on {delivery_day}.",
    "Sales record: {customer} just confirmed purchase of {quantity} {product} for ${amount}. Delivery due: {delivery_day}.",
    "Express dispatch for {customer}: {quantity} {product} ($ {amount}), target arrival date is {delivery_day}."
]

def generate_single_example(idx: int, rng: random.Random) -> Dict:
    """Generates one structured extraction pair."""
    customer = rng.choice(FIRST_NAMES)
    prod_name, min_p, max_p = rng.choice(PRODUCTS)
    quantity = rng.randint(1, 15)
    unit_price = rng.randint(min_p, max_p)
    amount = float(quantity * unit_price)
    delivery_day = rng.choice(DELIVERY_DAYS)
    inv_num = rng.randint(10000, 99999)

    template = rng.choice(SENTENCE_TEMPLATES)
    text_input = template.format(
        customer=customer,
        quantity=quantity,
        product=prod_name,
        amount=int(amount) if amount.is_integer() else amount,
        delivery_day=delivery_day,
        inv=inv_num
    )

    output_json = {
        "customer": customer,
        "quantity": quantity,
        "product": prod_name,
        "amount": amount,
        "delivery_day": delivery_day
    }

    # Conversational messages format ready for SFT / Chat Templates
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text_input},
        {"role": "assistant", "content": json.dumps(output_json)}
    ]

    return {
        "id": f"order_{idx:05d}",
        "instruction": "Extract the order details into structured JSON.",
        "input": text_input,
        "output": output_json,
        "messages": messages
    }

def generate_order_dataset(
    total_train: int = 1200,
    total_val: int = 150,
    total_test: int = 150,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Generates train, val, and test splits with zero data leakage."""
    rng = random.Random(seed)
    
    train_data = [generate_single_example(i + 1, rng) for i in range(total_train)]
    val_data = [generate_single_example(total_train + i + 1, rng) for i in range(total_val)]
    test_data = [generate_single_example(total_train + total_val + i + 1, rng) for i in range(total_test)]

    # Leakage verification
    train_inputs = {x["input"] for x in train_data}
    val_inputs = {x["input"] for x in val_data}
    test_inputs = {x["input"] for x in test_data}

    overlap_train_val = train_inputs.intersection(val_inputs)
    overlap_train_test = train_inputs.intersection(test_inputs)

    if overlap_train_val or overlap_train_test:
        raise ValueError(f"Data leakage detected! Train/Val overlap: {len(overlap_train_val)}, Train/Test overlap: {len(overlap_train_test)}")

    return train_data, val_data, test_data

def save_dataset_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str = "data/processed"
):
    """Saves splits to JSONL files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train.jsonl": train_data,
        "val.jsonl": val_data,
        "test.jsonl": test_data,
    }

    for filename, split in splits.items():
        file_path = out_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(split):>5} samples to {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate domain-specific fine-tuning dataset.")
    parser.add_argument("--train", type=int, default=1200, help="Number of training samples")
    parser.add_argument("--val", type=int, default=150, help="Number of validation samples")
    parser.add_argument("--test", type=int, default=150, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()
    print(f"Generating dataset with seed={args.seed}...")
    train_set, val_set, test_set = generate_order_dataset(args.train, args.val, args.test, args.seed)
    save_dataset_splits(train_set, val_set, test_set, args.out_dir)
    print("Dataset generation complete without leakage.")
