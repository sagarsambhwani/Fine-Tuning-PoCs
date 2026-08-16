"""
Unit tests for data generation, formatting, and schema validation.
"""
import random
import pytest
from src.data.dataset_generator import generate_order_dataset, generate_single_example
from src.data.validator import parse_and_validate_extraction, validate_json_schema, extract_json_substring
from src.data.formatter import format_chat_prompt

def test_generate_single_example():
    rng = random.Random(42)
    sample = generate_single_example(1, rng)
    
    assert "id" in sample
    assert "input" in sample
    assert "output" in sample
    assert "messages" in sample
    assert len(sample["messages"]) == 3
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert sample["messages"][2]["role"] == "assistant"

    # Validate output schema
    out = sample["output"]
    assert isinstance(out["customer"], str)
    assert isinstance(out["quantity"], int)
    assert isinstance(out["product"], str)
    assert isinstance(out["amount"], (int, float))
    assert isinstance(out["delivery_day"], str)

def test_dataset_anti_leakage_and_splits():
    train, val, test = generate_order_dataset(total_train=50, total_val=10, total_test=10, seed=123)
    assert len(train) == 50
    assert len(val) == 10
    assert len(test) == 10

    train_inputs = {x["input"] for x in train}
    val_inputs = {x["input"] for x in val}
    test_inputs = {x["input"] for x in test}

    assert len(train_inputs.intersection(val_inputs)) == 0, "Train and Validation have overlapping inputs!"
    assert len(train_inputs.intersection(test_inputs)) == 0, "Train and Test have overlapping inputs!"
    assert len(val_inputs.intersection(test_inputs)) == 0, "Validation and Test have overlapping inputs!"

def test_json_validator_clean():
    clean_json = '{"customer": "John", "quantity": 3, "product": "laptops", "amount": 2400.0, "delivery_day": "Friday"}'
    is_valid, parsed, err = parse_and_validate_extraction(clean_json)
    
    assert is_valid is True
    assert err is None
    assert parsed["customer"] == "John"
    assert parsed["quantity"] == 3
    assert parsed["amount"] == 2400.0

def test_json_validator_markdown_wrapped():
    markdown_json = "Here is the extracted information:\n```json\n{\"customer\": \"Sarah\", \"quantity\": 2, \"product\": \"4K monitors\", \"amount\": 800.0, \"delivery_day\": \"Monday\"}\n```\nHope that helps!"
    is_valid, parsed, err = parse_and_validate_extraction(markdown_json)
    
    assert is_valid is True
    assert err is None
    assert parsed["customer"] == "Sarah"
    assert parsed["product"] == "4K monitors"

def test_json_validator_missing_fields():
    incomplete_json = '{"customer": "John", "quantity": 3}'
    is_valid, parsed, err = parse_and_validate_extraction(incomplete_json)
    
    assert is_valid is False
    assert "Missing required schema keys" in err

def test_json_validator_invalid_syntax():
    broken_json = '{"customer": "John", quantity: 3, product: }'
    is_valid, parsed, err = parse_and_validate_extraction(broken_json)
    
    assert is_valid is False
    assert "JSONDecodeError" in err
